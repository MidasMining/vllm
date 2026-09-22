# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA attention with TurboQuant byte-packed KV cache.

Combines two existing patterns:
- haosdent's sparse-MLA split-KV decode (triton_mla_sparse_kernel.py):
  per-query topk gather over a sparse-attention indexer, online-softmax
  loop over the gathered slice, single cache buffer holds K (lanes [0:L])
  + K_pe (lanes [L:L+R]) + V (= K because MLA reconstructs V via W_UV).
- shanyulu's dense TQ decode (triton_turboquant_mla_decode.py): cache
  stored as packed uint8 bytes per token, dequant fused inside the
  attention loop (bit-unpack + centroid gather + vec_norm scale for MSE
  presets; uint8 → fp8 bitcast + k_scale for the k8v4 preset).

Per-token cache slot layout (uint8 bytes):
    [ kv_c_packed (KV_C_BYTES) | k_pe (KPE_BYTES) ]

KV_C_BYTES:
    k8v4 (fp8):     L         (1 byte / elem, no vec_norm)
    MSE 4-bit:      L/2 + 2   (packed nibbles + fp16 vec_norm)
    MSE 3-bit:      ceil(L*3/8) + 2

KPE_BYTES:
    bf16:           2 * R
    fp8 + scale:    R + 2     (R fp8 bytes + fp16 per-token scale)

This file currently implements the k8v4 preset only. MSE presets share
the same outer kernel scaffold; the dequant body branches on a
TQ_PRESET constexpr that will be widened when the MSE paths land.
"""

from __future__ import annotations

import functools

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton
from vllm.utils.platform_utils import num_compute_units

# DeepSeek-V3.2 / GLM-5 sparse MLA shape constants. These match
# triton_mla_sparse_kernel.py — the TQ variant runs on the same models.
_BLOCK_DMODEL = 512  # L (kv_lora_rank)
_BLOCK_DPE = 64  # R (qk_rope_head_dim)
_BLOCK_DV = 512  # V output dim — V reconstructed from kv_c so = L

_BLOCK_H = 16
_MIN_BLOCK_N = 16
_MERGE_BLOCK_H = 1
_MERGE_BLOCK_DV_TILE = 128
assert _BLOCK_DV % _MERGE_BLOCK_DV_TILE == 0
_NUM_MERGE_DV_TILES = _BLOCK_DV // _MERGE_BLOCK_DV_TILE

# TQ_PRESET enum (must stay in lockstep with the Python wrapper's dispatch).
_PRESET_K8V4 = 0
_PRESET_MSE_4BIT = 1
_PRESET_MSE_3BIT = 2
_PRESET_MSE_2BIT = 3

_FINAL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 16}, num_warps=nw, num_stages=ns)
    for nw in (2, 4)
    for ns in (2, 4)
]
_SPLIT_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=ns) for ns in (2, 4)
]

KV_SPLITS_CANDIDATES = (1, 2, 4, 8, 16)

_MIN_TOPK_PER_SPLIT = 128
_SPLIT_MAX_OCCUPANCY = 4

# Preset packed-byte arithmetic — kept in Python so the kernels can take
# byte offsets as compile-time constants.
import math as _math


def _kv_c_bytes(preset: int, L: int) -> int:
    if preset == _PRESET_K8V4:
        return L
    if preset == _PRESET_MSE_4BIT:
        return L // 2 + 2
    if preset == _PRESET_MSE_3BIT:
        return _math.ceil(L * 3 / 8) + 2
    if preset == _PRESET_MSE_2BIT:
        return L // 4 + 2
    raise ValueError(f"unknown TQ preset {preset}")


def _kpe_bytes(R: int, kpe_fp8: bool) -> int:
    return R + 2 if kpe_fp8 else 2 * R


@triton.jit
def _sparse_tq_mla_compute_tile(
    q_buffer,
    cache_buffer,  # uint8 packed cache.
    indices_ptr,
    k_scale_ptr,  # fp32 scalar — used by k8v4 path; ignored otherwise.
    centroids_ptr,  # bf16 [2**MSE_BITS] — used by MSE path; ignored for k8v4.
    cur_q,
    cur_head,
    cur_kv_head_id,
    mask_h,
    split_start,
    split_end,
    seq_kv,
    stride_q_token,
    stride_q_head,
    stride_cache_token,  # = packed_bytes per slot
    stride_cache_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    KV_C_BYTES: tl.constexpr,
    KPE_BYTES_OFFS: tl.constexpr,  # byte offset where k_pe starts (= KV_C_BYTES)
    VEC_NORM_OFFS: tl.constexpr,  # byte offset of fp16 vec_norm (MSE: KV_C_BYTES - 2)
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    MSE_BITS: tl.constexpr,  # 0=fp8 (k8v4), 4=MSE 4bit, 3=MSE 3bit
    NORM_CORRECTION: tl.constexpr,  # MSE only; re-normalize y_hat to unit norm
    KPE_FP8: tl.constexpr,
    FP8_E5M2: tl.constexpr,  # 1 = read fp8 as e5m2 (SM<89); 0 = e4nv (Hopper+).
):
    """Sparse-MLA tile compute with inline TQ dequant.

    Mirrors haosdent's `_sparse_mla_compute_tile`. Loads K/Kpe/V from a
    TurboQuant-packed uint8 cache and dequantizes each loaded tile to
    q.dtype before the dot products.

    MSE_BITS dispatches dequant strategy:
        0 = k8v4   (fp8 bitcast + k_scale; no centroids)
        4 = MSE 4-bit (nibble unpack + centroid gather + post-scale by vec_norm)
        3 = MSE 3-bit (3-bit unpack + centroid gather + post-scale by vec_norm)

    For MSE the K1-a optimization is used: qk = dot(q, y_hat); qk *=
    vec_norm.  V uses the same y_hat (transposed) and pre-scales p by
    vec_norm before the V dot — mathematically equivalent to scaling V.
    """
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)

    if MSE_BITS == 0:  # k8v4
        k_scale = tl.load(k_scale_ptr).to(tl.float32)
    else:
        k_scale = 1.0

    q = tl.load(
        q_buffer
        + cur_q * stride_q_token
        + cur_head[:, None] * stride_q_head
        + offs_d[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )
    # NoPE models (GLM-5.3-Flash: qk_rope_head_dim=0) have no PE lanes;
    # BLOCK_DPE == 0 compiles the PE loads/dots out entirely.
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < BLOCK_DMODEL + BLOCK_DPE
        qpe = tl.load(
            q_buffer
            + cur_q * stride_q_token
            + cur_head[:, None] * stride_q_head
            + offs_dpe[None, :],
            mask=(mask_h[:, None]) & (mask_dpe[None, :]),
            other=0.0,
        )

    NEG_LARGE = -1.0e30
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) + NEG_LARGE
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_indice in range(split_start, split_end, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < split_end
        indices = tl.load(
            indices_ptr
            + cur_q * stride_indices_token
            + cur_kv_head_id * stride_indices_head
            + offs_indice,
            mask=mask_indice,
            other=-1,
        )
        mask_kv = (indices >= 0) & (indices < seq_kv)
        # int64 offsets: with >2^31 byte offsets (large KV caches), int32
        # `indices * stride` overflows and faults (Xid 31) even though the
        # index VALUES pass the mask above.
        indices = indices.to(tl.int64)

        # ----- Dequant kv_c into y (BLOCK_DMODEL, BLOCK_N) bf16 -----
        if MSE_BITS == 0:  # k8v4 — 1 byte per elem, fp8e4m3 bitcast
            offs_kbytes = (
                indices[None, :] * stride_cache_token
                + cur_kv_head_id * stride_cache_head
                + offs_d[:, None]
            )
            k_u8 = tl.load(
                cache_buffer + offs_kbytes, mask=mask_kv[None, :], other=0,
            )
            if FP8_E5M2:
                k_fp8 = k_u8.to(tl.float8e5, bitcast=True)
            else:
                k_fp8 = k_u8.to(tl.float8e4nv, bitcast=True)
            y = (k_fp8.to(tl.float32) * k_scale).to(q.dtype)
            # k8v4 has no vec_norm — set to 1.0 so post-scaling is a no-op.
            vec_norm = tl.full([BLOCK_N], 1.0, dtype=tl.float32)
        elif MSE_BITS == 4:  # MSE 4-bit — 2 indices per byte
            # Each elem d ∈ [0, L) lives at byte (d // 2) within the slot,
            # in the low nibble (d % 2 == 0) or high nibble.
            byte_offs = (offs_d // 2)[:, None]  # (BLOCK_DMODEL, 1)
            offs_kbytes = (
                indices[None, :] * stride_cache_token
                + cur_kv_head_id * stride_cache_head
                + byte_offs
            )
            packed = tl.load(
                cache_buffer + offs_kbytes, mask=mask_kv[None, :], other=0,
            ).to(tl.int32)
            # Low nibble for even d, high nibble for odd d.
            shift = tl.where((offs_d % 2)[:, None] == 0, 0, 4)
            idx = (packed >> shift) & 0xF  # (BLOCK_DMODEL, BLOCK_N) int32
            # Gather centroids — addressable load, one bf16 lookup per (d, t).
            y = tl.load(centroids_ptr + idx)  # (BLOCK_DMODEL, BLOCK_N) bf16
            if NORM_CORRECTION:
                y_f32 = y.to(tl.float32)
                # Per-token norm of the centroid-reconstructed vector.
                col_norm = tl.sqrt(tl.sum(y_f32 * y_f32, axis=0))  # (BLOCK_N,)
                y = (y_f32 / (col_norm[None, :] + 1e-8)).to(q.dtype)
            # vec_norm: 2 bytes at VEC_NORM_OFFS within each slot, fp16 LE.
            vn_addr = (
                indices * stride_cache_token
                + cur_kv_head_id * stride_cache_head
                + VEC_NORM_OFFS
            )
            vn_lo = tl.load(cache_buffer + vn_addr, mask=mask_kv, other=0).to(
                tl.uint16
            )
            vn_hi = tl.load(cache_buffer + vn_addr + 1, mask=mask_kv, other=0).to(
                tl.uint16
            )
            vn_u16 = (vn_hi << 8) | vn_lo
            vec_norm = vn_u16.to(tl.float16, bitcast=True).to(tl.float32)
        elif MSE_BITS == 2:  # MSE 2-bit — 4 indices per byte
            # Each elem d ∈ [0, L) lives at byte (d // 4) within the slot,
            # at intra-byte shift (d % 4) * 2.
            byte_offs = (offs_d // 4)[:, None]
            intra_shift = ((offs_d % 4) * 2)[:, None]
            offs_kbytes = (
                indices[None, :] * stride_cache_token
                + cur_kv_head_id * stride_cache_head
                + byte_offs
            )
            packed = tl.load(
                cache_buffer + offs_kbytes, mask=mask_kv[None, :], other=0,
            ).to(tl.int32)
            idx = (packed >> intra_shift) & 0x3  # 4 centroids
            y = tl.load(centroids_ptr + idx)
            if NORM_CORRECTION:
                y_f32 = y.to(tl.float32)
                col_norm = tl.sqrt(tl.sum(y_f32 * y_f32, axis=0))
                y = (y_f32 / (col_norm[None, :] + 1e-8)).to(q.dtype)
            vn_addr = (
                indices * stride_cache_token
                + cur_kv_head_id * stride_cache_head
                + VEC_NORM_OFFS
            )
            vn_lo = tl.load(cache_buffer + vn_addr, mask=mask_kv, other=0).to(
                tl.uint16
            )
            vn_hi = tl.load(cache_buffer + vn_addr + 1, mask=mask_kv, other=0).to(
                tl.uint16
            )
            vn_u16 = (vn_hi << 8) | vn_lo
            vec_norm = vn_u16.to(tl.float16, bitcast=True).to(tl.float32)
        else:  # MSE 3-bit — indices straddle bytes (3 doesn't divide 8).
            # For element d, the 3-bit index starts at bit position d*3.
            # That straddles 1 or 2 bytes depending on intra_byte position.
            # Always read 2 consecutive bytes and shift — the +1 byte is
            # within the slot (either another kv_c byte or the start of the
            # fp16 vec_norm, both valid memory). The mask &0x7 ensures we
            # only consume the 3 bits we want.
            bit_pos = offs_d * 3  # (BLOCK_DMODEL,)
            byte_offs_d = (bit_pos // 8)[:, None]  # (BLOCK_DMODEL, 1)
            intra_bit = (bit_pos % 8)[:, None]  # (BLOCK_DMODEL, 1)
            offs_kbytes = (
                indices[None, :] * stride_cache_token
                + cur_kv_head_id * stride_cache_head
                + byte_offs_d
            )
            b0 = tl.load(
                cache_buffer + offs_kbytes, mask=mask_kv[None, :], other=0,
            ).to(tl.int32)
            b1 = tl.load(
                cache_buffer + offs_kbytes + 1, mask=mask_kv[None, :], other=0,
            ).to(tl.int32)
            combined = (b1 << 8) | b0  # uint16 stitched as int32
            idx = (combined >> intra_bit) & 0x7  # (BLOCK_DMODEL, BLOCK_N)
            y = tl.load(centroids_ptr + idx)  # (BLOCK_DMODEL, BLOCK_N) bf16
            if NORM_CORRECTION:
                y_f32 = y.to(tl.float32)
                col_norm = tl.sqrt(tl.sum(y_f32 * y_f32, axis=0))
                y = (y_f32 / (col_norm[None, :] + 1e-8)).to(q.dtype)
            vn_addr = (
                indices * stride_cache_token
                + cur_kv_head_id * stride_cache_head
                + VEC_NORM_OFFS
            )
            vn_lo = tl.load(cache_buffer + vn_addr, mask=mask_kv, other=0).to(
                tl.uint16
            )
            vn_hi = tl.load(cache_buffer + vn_addr + 1, mask=mask_kv, other=0).to(
                tl.uint16
            )
            vn_u16 = (vn_hi << 8) | vn_lo
            vec_norm = vn_u16.to(tl.float16, bitcast=True).to(tl.float32)

        # K1-a: dot first, then scalar post-multiply by vec_norm.
        # (For k8v4, vec_norm is 1.0 — pass-through; for MSE the scale
        # recovers the true K = y_hat * vec_norm magnitude.)
        qk = tl.dot(q, y)
        if MSE_BITS != 0:
            qk = qk * vec_norm[None, :]

        # ----- k_pe (positional encoding) — bf16 or fp8 + scale -----
        # Compiled out entirely for NoPE models (BLOCK_DPE == 0).
        if BLOCK_DPE > 0:
            offs_kpe_byte_start = (
                indices[None, :] * stride_cache_token
                + cur_kv_head_id * stride_cache_head
                + KPE_BYTES_OFFS
            )
            if not KPE_FP8:
                offs_kpe_lo = (
                    offs_kpe_byte_start + (offs_dpe - BLOCK_DMODEL)[:, None] * 2
                )
                kpe_lo = tl.load(
                    cache_buffer + offs_kpe_lo,
                    mask=(mask_kv[None, :]) & (mask_dpe[:, None]),
                    other=0,
                ).to(tl.uint16)
                kpe_hi = tl.load(
                    cache_buffer + offs_kpe_lo + 1,
                    mask=(mask_kv[None, :]) & (mask_dpe[:, None]),
                    other=0,
                ).to(tl.uint16)
                kpe_u16 = (kpe_hi << 8) | kpe_lo
                kpe = kpe_u16.to(tl.bfloat16, bitcast=True).to(q.dtype)
            else:
                offs_kpe_b = offs_kpe_byte_start + (offs_dpe - BLOCK_DMODEL)[:, None]
                kpe_u8 = tl.load(
                    cache_buffer + offs_kpe_b,
                    mask=(mask_kv[None, :]) & (mask_dpe[:, None]),
                    other=0,
                )
                if FP8_E5M2:
                    kpe_fp8 = kpe_u8.to(tl.float8e5, bitcast=True)
                else:
                    kpe_fp8 = kpe_u8.to(tl.float8e4nv, bitcast=True)
                # Scale is one per-token fp16 at slot byte KPE_BYTES_OFFS+BLOCK_DPE.
                # Compute the address as 1D (BLOCK_N,) so the loaded scale
                # broadcasts cleanly against kpe_fp8 (BLOCK_DPE, BLOCK_N) at
                # the multiply.
                scale_addr = (
                    indices * stride_cache_token
                    + cur_kv_head_id * stride_cache_head
                    + KPE_BYTES_OFFS
                    + BLOCK_DPE
                )
                kpe_scale_lo = tl.load(
                    cache_buffer + scale_addr, mask=mask_kv, other=0,
                ).to(tl.uint16)
                kpe_scale_hi = tl.load(
                    cache_buffer + scale_addr + 1, mask=mask_kv, other=0,
                ).to(tl.uint16)
                kpe_scale_u16 = (kpe_scale_hi << 8) | kpe_scale_lo  # (BLOCK_N,)
                kpe_scale = kpe_scale_u16.to(tl.float16, bitcast=True).to(
                    tl.float32
                )
                kpe = (kpe_fp8.to(tl.float32) * kpe_scale[None, :]).to(q.dtype)

            qk += tl.dot(qpe, kpe)
        qk *= sm_scale
        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, NEG_LARGE)

        # ----- V dequant + weighted sum -----
        # V is the same data as K in MLA (kv_c bytes). For k8v4, reload as
        # transposed access. For MSE, reuse y (transpose to (BLOCK_N, BLOCK_DV))
        # — avoids a second dequant pass.
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]

        if MSE_BITS == 0:  # k8v4 — reload V as separate tile
            offs_vbytes = (
                indices[:, None] * stride_cache_token
                + cur_kv_head_id * stride_cache_head
                + offs_dv[None, :]
            )
            v_u8 = tl.load(
                cache_buffer + offs_vbytes, mask=mask_kv[:, None], other=0,
            )
            if FP8_E5M2:
                v_fp8 = v_u8.to(tl.float8e5, bitcast=True)
            else:
                v_fp8 = v_u8.to(tl.float8e4nv, bitcast=True)
            v = (v_fp8.to(tl.float32) * k_scale).to(q.dtype)
            acc += tl.dot(p.to(v.dtype), v)
        else:
            # MSE: V = y_hat (already dequant'd), needs vec_norm scale.
            # Pre-scale p by vec_norm to avoid materializing scaled V.
            v = tl.trans(y)  # (BLOCK_N, BLOCK_DMODEL == BLOCK_DV)
            p_scaled = p * vec_norm[None, :]
            acc += tl.dot(p_scaled.to(v.dtype), v)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    return acc, e_max, e_sum


@triton.autotune(
    configs=_FINAL_AUTOTUNE_CONFIGS,
    key=["index_topk", "kv_group_num", "MSE_BITS", "NORM_CORRECTION", "KPE_FP8"],
)
@triton.jit
def _sparse_tq_mla_kernel_final(
    q_buffer,
    cache_buffer,
    indices_ptr,
    k_scale_ptr,
    centroids_ptr,
    out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_cache_token,
    stride_cache_head,
    stride_out_token,
    stride_out_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    kv_group_num: tl.constexpr,
    KV_C_BYTES: tl.constexpr,
    KPE_BYTES_OFFS: tl.constexpr,
    VEC_NORM_OFFS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    MSE_BITS: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    KPE_FP8: tl.constexpr,
    FP8_E5M2: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    acc, e_max, e_sum = _sparse_tq_mla_compute_tile(
        q_buffer,
        cache_buffer,
        indices_ptr,
        k_scale_ptr,
        centroids_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        0,
        index_topk,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_cache_token,
        stride_cache_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        KV_C_BYTES,
        KPE_BYTES_OFFS,
        VEC_NORM_OFFS,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
        MSE_BITS,
        NORM_CORRECTION,
        KPE_FP8,
        FP8_E5M2,
    )

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None],
    )


@triton.autotune(
    configs=_SPLIT_AUTOTUNE_CONFIGS,
    key=[
        "index_topk", "NUM_KV_SPLITS", "kv_group_num",
        "MSE_BITS", "NORM_CORRECTION", "KPE_FP8",
    ],
)
@triton.jit
def _sparse_tq_mla_kernel_split(
    q_buffer,
    cache_buffer,
    indices_ptr,
    k_scale_ptr,
    centroids_ptr,
    mid_out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_cache_token,
    stride_cache_head,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    KV_C_BYTES: tl.constexpr,
    KPE_BYTES_OFFS: tl.constexpr,
    VEC_NORM_OFFS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    LOGE2: tl.constexpr,
    MSE_BITS: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
    KPE_FP8: tl.constexpr,
    FP8_E5M2: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    split_topk: tl.constexpr = tl.cdiv(index_topk, NUM_KV_SPLITS)
    split_start = split_kv_id * split_topk
    split_end = tl.minimum(split_start + split_topk, index_topk)

    acc, e_max, e_sum = _sparse_tq_mla_compute_tile(
        q_buffer,
        cache_buffer,
        indices_ptr,
        k_scale_ptr,
        centroids_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        split_start,
        split_end,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_cache_token,
        stride_cache_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        KV_C_BYTES,
        KPE_BYTES_OFFS,
        VEC_NORM_OFFS,
        BLOCK_H,
        BLOCK_N,
        BLOCK_DV,
        BLOCK_DMODEL,
        BLOCK_DPE,
        MSE_BITS,
        NORM_CORRECTION,
        KPE_FP8,
        FP8_E5M2,
    )

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_dv = tl.arange(0, BLOCK_DV)
    mid_base_2d = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head[:, None] * stride_mid_head
        + split_kv_id * stride_mid_split
    )
    tl.store(
        mid_base_2d + offs_dv[None, :],
        acc / e_sum_safe[:, None],
        mask=mask_h[:, None],
    )
    mid_lse_ptr = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head * stride_mid_head
        + split_kv_id * stride_mid_split
        + BLOCK_DV
    )
    tl.store(mid_lse_ptr, (e_max + tl.log2(e_sum)) * LOGE2, mask=mask_h)


# Merge kernel is identical to haosdent's — no KV access during merge,
# just online-softmax reduction of per-split (acc, lse) buffers.
@triton.jit
def _sparse_tq_mla_merge_kernel(
    mid_out_ptr,
    out_ptr,
    h_q,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_out_token,
    stride_out_head,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DV_TILE: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_dv_tile = tl.program_id(2)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    offs_dv = cur_dv_tile * BLOCK_DV_TILE + tl.arange(0, BLOCK_DV_TILE)
    mask_dv = offs_dv < BLOCK_DV
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV_TILE], dtype=tl.float32)

    mid_base_2d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head[:, None] * stride_mid_head
    )
    mid_lse_1d = (
        mid_out_ptr + cur_q * stride_mid_token + cur_head * stride_mid_head + BLOCK_DV
    )

    for split_kv_id in range(NUM_KV_SPLITS):
        tv = tl.load(
            mid_base_2d + split_kv_id * stride_mid_split + offs_dv[None, :],
            mask=mask_h[:, None] & mask_dv[None, :],
            other=0.0,
        )
        tlogic = tl.load(
            mid_lse_1d + split_kv_id * stride_mid_split,
            mask=mask_h,
            other=-float("inf"),
        )
        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        exp_logic = tl.exp(tlogic - n_e_max)
        acc = acc * old_scale[:, None] + exp_logic[:, None] * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    tl.store(
        out_ptr
        + cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :],
        (acc / e_sum_safe[:, None]).to(tl.bfloat16),
        mask=mask_h[:, None] & mask_dv[None, :],
    )


@functools.lru_cache(maxsize=256)
def _choose_num_kv_splits(
    num_tokens: int, num_head_groups: int, index_topk: int, sm_count: int
) -> int:
    baseline = num_tokens * num_head_groups
    if baseline == 0 or baseline * _SPLIT_MAX_OCCUPANCY >= sm_count:
        return 1
    ideal = triton.next_power_of_2(max(1, index_topk // _MIN_TOPK_PER_SPLIT))
    max_splits = max(1, sm_count // baseline)
    max_splits = 1 << (max_splits.bit_length() - 1)
    num_kv_splits = min(ideal, max_splits)
    while num_kv_splits > 1 and index_topk % num_kv_splits != 0:
        num_kv_splits //= 2
    return max(1, num_kv_splits)


def triton_sparse_tq_mla_attention(
    q: torch.Tensor,
    cache: torch.Tensor,  # uint8, (seq_kv, num_kv_heads=1, packed_bytes)
    indices: torch.Tensor,
    sm_scale: float,
    preset: str,  # 'k8v4' | 'mse_4bit' | 'mse_3bit'
    L: int,
    R: int,
    kpe_fp8: bool,
    k_scale: torch.Tensor | None = None,
    centroids: torch.Tensor | None = None,
    norm_correction: bool = True,
    fp8_e5m2: bool = False,
    num_kv_splits: int | None = None,
    sm_count: int | None = None,
) -> torch.Tensor:
    """Sparse MLA attention over a TurboQuant-packed uint8 cache.

    Args:
        q:        [num_tokens, num_heads_q, L+R] bf16. For MSE presets q
                  is already Hadamard-rotated (Pi folded into W_UK_T at
                  init); for k8v4 there is no rotation.
        cache:    [seq_kv, num_kv_heads=1, packed_bytes] uint8 packed
                  per-token slots.
        indices:  [num_tokens, num_kv_heads=1, topk] int32 sparse topk.
        sm_scale: softmax scale.
        preset:   one of 'k8v4', 'mse_4bit', 'mse_3bit'.
        L:        kv_lora_rank (BLOCK_DMODEL constexpr).
        R:        qk_rope_head_dim (BLOCK_DPE constexpr).
        kpe_fp8:  True if k_pe is stored as R fp8 bytes + 2-byte fp16
                  per-token scale; False if k_pe is 2*R bf16 bytes.
        k_scale:  fp32 1-element tensor for k8v4 fp8 dequant. Required
                  when preset='k8v4'; ignored otherwise.
        centroids: bf16 [2**MSE_BITS] Lloyd-Max codebook. Required for
                   MSE presets; ignored for k8v4.
        norm_correction: re-normalize centroid-reconstructed vector to
                         unit norm before applying vec_norm. MSE only.

    Returns:
        out: [num_tokens, num_heads_q, _BLOCK_DV] bf16.
    """
    preset_id = {"k8v4": _PRESET_K8V4, "mse_4bit": _PRESET_MSE_4BIT,
                 "mse_3bit": _PRESET_MSE_3BIT, "mse_2bit": _PRESET_MSE_2BIT}.get(preset)
    if preset_id is None:
        raise ValueError(f"unknown TQ preset {preset!r}")
    # Per-preset compile-time integers fed to the kernel as constexprs.
    if preset_id == _PRESET_K8V4:
        mse_bits = 0
    elif preset_id == _PRESET_MSE_4BIT:
        mse_bits = 4
    elif preset_id == _PRESET_MSE_3BIT:
        mse_bits = 3
    else:  # _PRESET_MSE_2BIT
        mse_bits = 2

    num_tokens, num_heads_q, dim_qk = q.shape
    assert dim_qk == L + R, (
        f"sparse TQ MLA kernel expected dim_qk={L + R}, got {dim_qk}"
    )
    assert cache.shape[1] == 1
    packed_bytes = cache.shape[2]
    expected_packed = _kv_c_bytes(preset_id, L) + _kpe_bytes(R, kpe_fp8)
    assert packed_bytes == expected_packed, (
        f"cache slot is {packed_bytes} bytes; expected {expected_packed} "
        f"for preset={preset} L={L} R={R} kpe_fp8={kpe_fp8}"
    )
    index_topk = indices.shape[2]
    assert index_topk % _MIN_BLOCK_N == 0, (
        f"topk ({index_topk}) must be a multiple of {_MIN_BLOCK_N}"
    )

    if preset_id == _PRESET_K8V4:
        assert k_scale is not None and k_scale.numel() == 1, (
            "k8v4 requires a 1-element k_scale tensor"
        )
        k_scale_arg = k_scale
        centroids_arg = cache  # unused; dummy pointer
        # k8v4 has no vec_norm — set offset to 0 (kernel won't read with mse_bits=0).
        vec_norm_offs = 0
    else:  # MSE preset
        assert centroids is not None, (
            f"MSE preset {preset!r} requires a centroids tensor"
        )
        expected_n_cents = 1 << mse_bits  # 16 for 4-bit, 8 for 3-bit
        assert centroids.numel() == expected_n_cents, (
            f"centroids must have {expected_n_cents} entries for "
            f"{mse_bits}-bit MSE; got {centroids.numel()}"
        )
        centroids_arg = centroids
        # k_scale is unused on MSE; pass cache as dummy pointer.
        k_scale_arg = cache
        # vec_norm sits in the last 2 bytes of the kv_c region.
        vec_norm_offs = _kv_c_bytes(preset_id, L) - 2

    kv_group_num = num_heads_q
    num_head_groups = triton.cdiv(num_heads_q, min(_BLOCK_H, kv_group_num))

    if num_kv_splits is None or num_kv_splits == 0:
        if sm_count is None:
            sm_count = num_compute_units(q.device.index)
        num_kv_splits = _choose_num_kv_splits(
            num_tokens, num_head_groups, index_topk, sm_count
        )

    out = torch.empty(
        (num_tokens, num_heads_q, _BLOCK_DV),
        dtype=torch.bfloat16,
        device=q.device,
    )

    kv_c_bytes_val = _kv_c_bytes(preset_id, L)
    norm_corr_int = 1 if norm_correction else 0

    if num_kv_splits == 1:
        _sparse_tq_mla_kernel_final[(num_tokens, num_head_groups)](
            q_buffer=q,
            cache_buffer=cache,
            indices_ptr=indices,
            k_scale_ptr=k_scale_arg,
            centroids_ptr=centroids_arg,
            out_ptr=out,
            seq_kv=cache.shape[0],
            h_q=num_heads_q,
            stride_q_token=q.stride(0),
            stride_q_head=q.stride(1),
            stride_cache_token=cache.stride(0),
            stride_cache_head=cache.stride(1),
            stride_out_token=out.stride(0),
            stride_out_head=out.stride(1),
            stride_indices_token=indices.stride(0),
            stride_indices_head=indices.stride(1),
            sm_scale=sm_scale * LOG2E,
            index_topk=index_topk,
            kv_group_num=kv_group_num,
            KV_C_BYTES=kv_c_bytes_val,
            KPE_BYTES_OFFS=kv_c_bytes_val,
            VEC_NORM_OFFS=vec_norm_offs,
            BLOCK_H=_BLOCK_H,
            BLOCK_DV=_BLOCK_DV,
            BLOCK_DMODEL=L,
            BLOCK_DPE=R,
            MSE_BITS=mse_bits,
            NORM_CORRECTION=norm_corr_int,
            KPE_FP8=kpe_fp8,
            FP8_E5M2=1 if fp8_e5m2 else 0,
        )
        return out

    mid_out = torch.empty(
        (num_tokens, num_heads_q, num_kv_splits, _BLOCK_DV + 1),
        dtype=torch.float32,
        device=q.device,
    )
    _sparse_tq_mla_kernel_split[(num_tokens, num_head_groups, num_kv_splits)](
        q_buffer=q,
        cache_buffer=cache,
        indices_ptr=indices,
        k_scale_ptr=k_scale_arg,
        centroids_ptr=centroids_arg,
        mid_out_ptr=mid_out,
        seq_kv=cache.shape[0],
        h_q=num_heads_q,
        stride_q_token=q.stride(0),
        stride_q_head=q.stride(1),
        stride_cache_token=cache.stride(0),
        stride_cache_head=cache.stride(1),
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_indices_token=indices.stride(0),
        stride_indices_head=indices.stride(1),
        sm_scale=sm_scale * LOG2E,
        index_topk=index_topk,
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        KV_C_BYTES=kv_c_bytes_val,
        KPE_BYTES_OFFS=kv_c_bytes_val,
        VEC_NORM_OFFS=vec_norm_offs,
        BLOCK_H=_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DMODEL=L,
        BLOCK_DPE=R,
        LOGE2=LOGE2,
        MSE_BITS=mse_bits,
        NORM_CORRECTION=norm_corr_int,
        KPE_FP8=kpe_fp8,
        FP8_E5M2=1 if fp8_e5m2 else 0,
    )

    _sparse_tq_mla_merge_kernel[(num_tokens, num_heads_q, _NUM_MERGE_DV_TILES)](
        mid_out_ptr=mid_out,
        out_ptr=out,
        h_q=num_heads_q,
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_MERGE_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DV_TILE=_MERGE_BLOCK_DV_TILE,
        num_warps=2,
    )
    return out
