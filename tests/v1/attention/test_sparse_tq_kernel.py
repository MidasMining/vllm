# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the sparse-MLA TurboQuant path on SM80 (GLM NoPE k8v4).

Guards the three port-specific behaviors on top of the fork's kernels:
  1. R == 0 (NoPE) compiles the k_pe path out of both store and read
     kernels without touching bytes past the kv_c region.
  2. The store kernel addresses the paged cache through its strides
     (single-arena layout), matching concat_and_cache_mla semantics.
  3. e5m2 store + bitcast dequant roundtrips within fp8 tolerance of a
     bf16 attention reference.

GPU-only; skipped without CUDA. Run standalone (not during serving —
concurrent CUDA processes wedge the CMP 170HX GSP):
    .venv/bin/python -m pytest --confcutdir=tests/v1/attention \
        tests/v1/attention/test_sparse_tq_kernel.py -q
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from vllm.v1.attention.ops.triton_mla_sparse_tq_kernel import (
    triton_sparse_tq_mla_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import _mla_fused_store_fp8

L = 512
BLOCK_SIZE = 64
NUM_HEADS = 32
TOPK = 128  # multiple of 16


def _is_sm_lt_89() -> bool:
    cap = torch.cuda.get_device_capability()
    return cap[0] < 8 or (cap[0] == 8 and cap[1] < 9)


@torch.inference_mode()
def test_k8v4_nope_store_read_roundtrip_strided():
    device = "cuda"
    torch.manual_seed(7)
    fp8_e5m2 = _is_sm_lt_89()

    num_blocks = 8
    # Interleaved arena: rows between this layer's blocks belong to other
    # layers. block stride = 3 pages.
    arena = torch.full(
        (num_blocks * 3, BLOCK_SIZE, L), 0xAB, dtype=torch.uint8, device=device
    )
    kv_cache = arena[::3]  # [num_blocks, BLOCK_SIZE, L], stride(0) = 3*64*512
    assert kv_cache.stride(0) == 3 * BLOCK_SIZE * L

    num_tokens = 3 * BLOCK_SIZE  # fill blocks 0..2
    kv_c = torch.randn(num_tokens, L, dtype=torch.bfloat16, device=device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)

    _mla_fused_store_fp8(
        kv_c,
        kv_c,  # NoPE dummy k_pe pointer; R == 0 never reads it
        kv_cache,
        slot_mapping,
        k_scale,
        kv_lora_rank=L,
        qk_rope_head_dim=0,
        k_pe_fp8=False,
        fp8_e5m2=fp8_e5m2,
    )

    # 1+2: only this layer's pages were written; interleaved pages intact.
    other = torch.cat([arena[1::3], arena[2::3]])
    assert (other == 0xAB).all(), "store leaked outside the layer's pages"
    written = arena[0::3][:3]
    assert not (written == 0xAB).all(), "store wrote nothing"

    # 3: sparse TQ attention vs bf16 reference over the dequantized cache.
    fp8_t = torch.float8_e5m2 if fp8_e5m2 else torch.float8_e4m3fn
    deq = (
        kv_cache[:3]
        .reshape(num_tokens, L)
        .view(fp8_t)
        .to(torch.float32)
    )
    # fp8 quantization error alone should be small on randn data
    # (e5m2 keeps 2 mantissa bits: rel err <= 12.5%, so ~0.6 near |x|=4.5)
    q_err = (deq - kv_c.float()).abs().max().item()
    assert q_err < 0.8, f"fp8 store error too large: {q_err}"

    q = torch.randn(2, NUM_HEADS, L, dtype=torch.bfloat16, device=device)
    indices = torch.randint(
        0, num_tokens, (2, 1, TOPK), dtype=torch.int32, device=device
    )
    indices[0, 0, -5:] = -1  # padding must be masked out
    sm_scale = L**-0.5

    # kernel view: flat rows over the strided arena (row = block*3*64 + off)
    rows = kv_cache.as_strided(
        ((num_blocks - 1) * 3 * BLOCK_SIZE + BLOCK_SIZE, L), (L, 1)
    )
    row_idx = torch.where(
        indices < 0,
        indices,
        (indices // BLOCK_SIZE) * (3 * BLOCK_SIZE) + indices % BLOCK_SIZE,
    )
    out = triton_sparse_tq_mla_attention(
        q,
        rows.view(-1, 1, L),
        row_idx,
        sm_scale=sm_scale,
        preset="k8v4",
        L=L,
        R=0,
        kpe_fp8=False,
        k_scale=k_scale,
        fp8_e5m2=fp8_e5m2,
    )

    # bf16 reference with the same dequantized K=V
    for t in range(2):
        idx = indices[t, 0]
        valid = idx >= 0
        kv = deq[idx[valid].long()]  # [n, L]
        logits = (q[t].float() @ kv.T) * sm_scale
        p = torch.softmax(logits, dim=-1)
        ref = p @ kv  # [heads, L]
        got = out[t, :NUM_HEADS].float()
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
