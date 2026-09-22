# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse-MLA + TurboQuant KV cache backend.

Port of promisezackr's `mla-sparse-tq-do-all` branch onto the v0.30 sparse
backend stack. Combines two existing patterns:
- `TritonMLASparseBackend`: sparse-MLA via DSA indexer + topk gather +
  split-decode Triton kernel (single-arena flat-row cache addressing).
- `TritonMLATurboQuantBackend` (fork): TurboQuant byte-packed KV cache +
  fused dequant inside the attention kernel.

SM80 (CMP 170HX) runs fp8 presets through e5m2 stores/bitcast-dequant
(Triton cannot load fp8e4nv on SM<89); SM89+ keeps e4m3.

Cache layout per slot (uint8 bytes):
    [ kv_c_packed (L bytes for k8v4) | k_pe (2*R bf16 or R+2 fp8) ]
NoPE models (GLM-5.3-Flash sparse layers, qk_rope_head_dim=0) store no
k_pe segment: the slot is just the packed kv_c latent.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from vllm import envs
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
    TritonMLASparseImpl,
    TritonMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.triton_mla_sparse_tq_kernel import (
    KV_SPLITS_CANDIDATES,
    _PRESET_K8V4,
    _PRESET_MSE_2BIT,
    _PRESET_MSE_3BIT,
    _PRESET_MSE_4BIT,
    triton_sparse_tq_mla_attention,
)

_PRESET_STR = {
    "turboquant_k8v4": ("k8v4", _PRESET_K8V4),
    "turboquant_4bit_nc": ("mse_4bit", _PRESET_MSE_4BIT),
    "turboquant_k3v4_nc": ("mse_3bit", _PRESET_MSE_3BIT),
    "turboquant_3bit_nc": ("mse_3bit", _PRESET_MSE_3BIT),
    "turboquant_2bit_nc": ("mse_2bit", _PRESET_MSE_2BIT),
}


class TritonMLASparseTurboQuantMetadataBuilder(TritonMLASparseMetadataBuilder):
    # Same CG support as the bf16 sparse path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLASparseTurboQuantImpl(TritonMLASparseImpl):
    """Sparse-MLA impl with TurboQuant byte-packed KV cache.

    Mirrors TritonMLATurboQuantImpl's store/setup machinery (Hadamard
    buffers, Lloyd-Max centroids, Pi fold into W_UK_T / W_UV for MSE
    presets) on top of the v0.30 Triton sparse forward path, with
    TQ-aware dequant in the inner loop.
    """

    def __init__(self, *args, **kwargs) -> None:
        # The sparse impl base takes qk_rope_head_dim via **mla_args but
        # never assigns it to self. We need it on self for the TQ config
        # and kernel launch, so capture it before super() consumes the
        # kwargs.
        self.qk_rope_head_dim = kwargs.get("qk_rope_head_dim")
        if self.qk_rope_head_dim is None:
            head_size = args[1] if len(args) > 1 else kwargs.get("head_size")
            kv_lora_rank = kwargs.get("kv_lora_rank")
            if head_size is not None and kv_lora_rank is not None:
                self.qk_rope_head_dim = head_size - kv_lora_rank
        super().__init__(*args, **kwargs)
        if self.qk_rope_head_dim is None:
            raise RuntimeError(
                "TritonMLASparseTurboQuantImpl could not resolve "
                "qk_rope_head_dim from constructor args"
            )

        L = self.kv_lora_rank
        assert L > 0 and (L & (L - 1)) == 0, (
            f"TritonMLASparseTurboQuant requires kv_lora_rank to be a power "
            f"of 2 (Sylvester Hadamard); got {L}"
        )
        assert self.qk_rope_head_dim >= 0
        k_pe_fp8 = envs.VLLM_TQ_KPE_FP8
        self.tq_config: TurboQuantConfig = TurboQuantConfig.from_cache_dtype(
            self.kv_cache_dtype,
            head_dim=L,
            rope_head_dim=self.qk_rope_head_dim,
            k_pe_fp8=k_pe_fp8,
        )
        self._kpe_fp8 = self.tq_config.k_pe_fp8
        self._kv_c_bytes = self.tq_config.key_packed_size
        self._k_pe_bytes = self.tq_config.k_pe_bytes
        self._packed_bytes = self.tq_config.mla_packed_bytes

        preset_pair = _PRESET_STR.get(self.kv_cache_dtype)
        assert preset_pair is not None, (
            f"unknown TurboQuant preset {self.kv_cache_dtype!r}"
        )
        self._preset_str, self._preset_id = preset_pair

        # Never let vLLM quantize the query to match the cache dtype —
        # Triton on SM80/86 can't compile fp8e4nv loads of the query, and
        # the MSE path needs bf16 q anyway (the Pi rotation is folded into
        # W_UK_T so q comes in pre-rotated).
        self.supports_quant_query_input = False

        # Pick the fp8 type used for kv_c/k_pe based on platform. Triton on
        # SM<89 can't load tl.float8e4nv (E4M3), so we store and read as
        # tl.float8e5 (E5M2). On SM89+ (Ada/Hopper) we keep E4M3 for its
        # better mantissa precision on small key magnitudes. Store and read
        # paths must use the same type — both threaded via `fp8_e5m2`.
        from vllm.platforms import current_platform

        cap = current_platform.get_device_capability()
        self._fp8_e5m2: bool = bool(
            cap is not None and (cap.major < 8 or (cap.major == 8 and cap.minor < 9))
        )

        # Parent __init__ ran _warmup_autotune before the TQ fields above
        # existed (it early-returns in that state); warm the TQ kernel now.
        self._warmup_autotune()

    # ---------------- autotune warmup ----------------
    def _warmup_autotune(self) -> None:
        if self.topk_indices_buffer is None or not hasattr(self, "_preset_str"):
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        dim_qk = self.kv_lora_rank + self.qk_rope_head_dim
        q = torch.empty(1, self.num_heads, dim_qk, dtype=torch.bfloat16, device=device)
        cache = torch.zeros(
            64, 1, self._packed_bytes, dtype=torch.uint8, device=device
        )
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        if self.tq_config.key_fp8:
            k_scale = torch.ones(1, dtype=torch.float32, device=device)
            centroids = None
        else:
            k_scale = None
            centroids = self._get_buffers(device)["centroids_bf16"]
        for splits in KV_SPLITS_CANDIDATES:
            triton_sparse_tq_mla_attention(
                q,
                cache,
                indices,
                sm_scale=self.softmax_scale,
                preset=self._preset_str,
                L=self.kv_lora_rank,
                R=self.qk_rope_head_dim,
                kpe_fp8=self._kpe_fp8,
                k_scale=k_scale,
                centroids=centroids,
                norm_correction=self.tq_config.norm_correction,
                fp8_e5m2=self._fp8_e5m2,
                num_kv_splits=splits,
                sm_count=self._sm_count,
            )

    # ---------------- buffers (Hadamard + centroids) ----------------
    # MSE presets only; depends on L + device + bit-width, not on the
    # dense-vs-sparse forward path. k8v4 never reaches these.
    def _ensure_buffers(self, device: torch.device) -> None:
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            get_centroids,
        )
        from vllm.v1.attention.backends.mla.triton_mla_tq import _build_hadamard

        key = str(device)
        if not hasattr(self, "_tq_buffers"):
            self._tq_buffers: dict[str, dict] = {}
        if key in self._tq_buffers:
            return
        D = self.kv_lora_rank
        H = _build_hadamard(D, str(device))  # symmetric → Pi == PiT
        _BF16 = torch.bfloat16
        buf: dict = {"Pi": H, "PiT": H, "Pi_bf16": H.to(_BF16)}
        if not self.tq_config.key_fp8:
            cents = get_centroids(D, self.tq_config.key_mse_bits).to(
                device=device, dtype=torch.float32
            )
            c_sorted, _ = cents.sort()
            buf["centroids"] = cents
            buf["centroids_bf16"] = cents.to(_BF16)
            buf["midpoints"] = (c_sorted[:-1] + c_sorted[1:]) / 2
        self._tq_buffers[key] = buf

    def _get_buffers(self, device: torch.device) -> dict:
        self._ensure_buffers(device)
        return self._tq_buffers[str(device)]

    def _maybe_fold_pi_into_layer(self, layer) -> None:
        """Fold Pi into layer.W_UK_T and layer.W_UV once. MSE-only —
        FP8 (k8v4) has no Hadamard rotation."""
        if self.tq_config.key_fp8:
            return
        if getattr(layer, "_tq_pi_folded", False):
            return
        if not (hasattr(layer, "W_UK_T") and hasattr(layer, "W_UV")):
            return
        Pi = self._get_buffers(layer.W_UK_T.device)["Pi_bf16"]
        Pi_f32 = Pi.to(torch.float32)
        wuk = layer.W_UK_T
        wuk_new = (wuk.to(torch.float32) @ Pi_f32).to(wuk.dtype)
        layer.W_UK_T = wuk_new
        wuv = layer.W_UV
        wuv_new = (Pi_f32 @ wuv.to(torch.float32)).to(wuv.dtype)
        layer.W_UV = wuv_new
        layer._tq_pi_folded = True

    # ---------------- store (write-path) ----------------
    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Reuse the fused MLA TQ store kernels (_mla_fused_store_fp8 /
        _mse). Storage layout is dense-vs-sparse agnostic."""
        from vllm.v1.attention.ops.triton_turboquant_store import (
            _mla_fused_store_fp8,
            _mla_fused_store_mse,
        )

        torch.cuda.nvtx.range_push("sparse_tq_store")
        if kv_cache.numel() == 0:
            torch.cuda.nvtx.range_pop()
            return
        N = slot_mapping.shape[0]
        if N <= 0:
            torch.cuda.nvtx.range_pop()
            return

        k_pe_ = k_pe.squeeze(1) if k_pe.dim() == 3 else k_pe
        kv_c_ = kv_c_normed[:N]
        k_pe_ = k_pe_[:N]
        if self.qk_rope_head_dim == 0:
            # NoPE: the kernel compiles the k_pe path out (R == 0) but still
            # takes a pointer arg; hand it a real tensor, never a 0-numel one.
            k_pe_ = kv_c_

        if not hasattr(self, "_tq_k_scale_f32"):
            self._tq_k_scale_f32 = k_scale.to(torch.float32).contiguous()

        if self.tq_config.key_fp8:
            _mla_fused_store_fp8(
                kv_c_, k_pe_, kv_cache, slot_mapping, self._tq_k_scale_f32,
                self.kv_lora_rank, self.qk_rope_head_dim, self._kpe_fp8,
                fp8_e5m2=self._fp8_e5m2,
            )
        else:
            device = kv_cache.device
            buf = self._get_buffers(device)
            _mla_fused_store_mse(
                kv_c_, k_pe_, kv_cache, slot_mapping,
                buf["PiT"], buf["midpoints"],
                self.tq_config.key_mse_bits,
                self.kv_lora_rank, self.qk_rope_head_dim, self._kpe_fp8,
                fp8_e5m2=self._fp8_e5m2,
            )

        torch.cuda.nvtx.range_pop()

    # ---------------- forward (read-path) ----------------
    def forward_mqa(
        self,
        q,
        kv_c_and_k_pe_cache,
        attn_metadata,
        layer,
    ):
        """Sparse-MLA forward with TQ-aware dequant. Follows the v0.30
        parent's single-arena addressing (flat_kv_row_view + row-stride
        aware index conversion), then hands off to the sparse TQ kernel."""
        # One-time per-layer Pi fold (MSE only).
        self._maybe_fold_pi_into_layer(layer)

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            # The kpool indexer widens the buffer past index_topk; extra
            # slots are -1 and masked out by the kernel.
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        cache_view = kv_rows.view(-1, 1, kv_rows.shape[-1])
        topk_indices_global = topk_indices_global.view(num_actual_toks, 1, -1)

        # Force k_scale to fp32 — vLLM sometimes stores layer._k_scale as
        # fp8 itself, and Triton can't load fp8e4nv on SM80/SM86.
        k_scale_fp32 = None
        centroids_bf16 = None
        if self.tq_config.key_fp8:
            k_scale_fp32 = layer._k_scale
            if k_scale_fp32.dtype != torch.float32:
                k_scale_fp32 = k_scale_fp32.float()
        else:
            centroids_bf16 = self._get_buffers(cache_view.device)["centroids_bf16"]

        output = triton_sparse_tq_mla_attention(
            q,
            cache_view,
            topk_indices_global,
            sm_scale=self.softmax_scale,
            preset=self._preset_str,
            L=self.kv_lora_rank,
            R=self.qk_rope_head_dim,
            kpe_fp8=self._kpe_fp8,
            k_scale=k_scale_fp32,
            centroids=centroids_bf16,
            norm_correction=self.tq_config.norm_correction,
            fp8_e5m2=self._fp8_e5m2,
            sm_count=self._sm_count,
        )
        return output[:, : self.num_heads, :], None


class TritonMLASparseTurboQuantBackend(TritonMLASparseBackend):
    """Sparse-MLA + TurboQuant backend (SM80+).

    Selectable for GLM-5.3 / DSv3.2 / DSv4 family models that use DSA
    sparse attention with a turboquant_* kv-cache-dtype. bf16 stays on
    TritonMLASparseBackend.
    """

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_TURBOQUANT"

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseTurboQuantMetadataBuilder"]:
        return TritonMLASparseTurboQuantMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseTurboQuantImpl"]:
        return TritonMLASparseTurboQuantImpl
