#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <numeric>
#include <sycl/sycl.hpp>
#include "../reference/tq_cpu_attention.h"

// ============================================================================
// TurboQuant Phase 2 Stage 1b: Paged KV Cache Microbenchmark
//
// Tests whether the Stage 1 TQ speedup (0.66x of FP16) survives under:
//   1. Paged KV cache with scattered blocks
//   2. Variable sequence lengths per batch element
//   3. Batch > 1 (multiple concurrent decode tokens)
//   4. GQA head mapping (Qwen3.6: 16 Q heads, 2 KV heads for full attn)
//
// Qwen3.6-35B-A3B head configuration:
//   Full attention (every 4th layer): n_q_heads=16, n_kv_heads=2, head_dim=256
//   Linear attention (3/4 layers):    n_k_heads=16, n_v_heads=32, head_dim=128
//
// For TurboQuant microbench we use head_dim=128 with configurable GQA ratio.
// ============================================================================

// ============================================================================
// Paged K-score kernel: TQ path
// One thread per (batch, q_head, token) — sequential 128-dim dot product
// Reads K via block table indirection
// ============================================================================

// KV block layout in device memory (TQ path):
//   d_tq_pool[block_idx * n_kv_heads * block_size * 68 + kv_head * block_size * 68 + tok_in_block * 68]
//   = one block_turbo4_0 (68 bytes)
//
// KV block layout in device memory (FP16 path):
//   d_fp16_pool[block_idx * n_kv_heads * block_size * 128 + kv_head * block_size * 128 + tok_in_block * 128]
//   = 128 half values

template <int WG_SIZE>
void launch_paged_k_scores_tq(sycl::queue& q,
    const uint8_t* d_tq_pool,         // raw TQ block pool
    const int* d_block_tables,         // [max_batch, max_blocks_per_seq]
    const int* d_seq_lengths,          // [max_batch]
    const float* d_q_rot,              // [batch * n_q_heads * 128]
    const float* d_tc4,                // [16] codebook
    float* d_scores,                   // [batch * n_q_heads * max_tokens]
    int batch_size, int n_q_heads, int n_kv_heads,
    int max_seq_len, int max_blocks_per_seq,
    int block_size)
{
    int gqa_ratio = n_q_heads / n_kv_heads;
    int wgs_per_head = (max_seq_len + WG_SIZE - 1) / WG_SIZE;
    int total_wgs = batch_size * n_q_heads * wgs_per_head;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int lid = item.get_local_id(0);
                auto grp = item.get_group();

                int b = wg_id / (n_q_heads * wgs_per_head);
                int rem = wg_id % (n_q_heads * wgs_per_head);
                int q_head = rem / wgs_per_head;
                int kv_head = q_head / gqa_ratio;
                int tile_start = (rem % wgs_per_head) * WG_SIZE;
                int tok = tile_start + lid;

                int this_seq_len = d_seq_lengths[b];

                // Cooperatively load q_rot into SLM
                auto slm_q = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                for (int i = lid; i < 128; i += WG_SIZE) {
                    (*slm_q)[i] = d_q_rot[(b * n_q_heads + q_head) * 128 + i];
                }
                sycl::group_barrier(grp);

                if (tok >= this_seq_len) return;

                // Block table lookup
                int block_idx = d_block_tables[b * max_blocks_per_seq + tok / block_size];
                int tok_in_block = tok % block_size;

                // Read TQ block from pool
                const uint8_t* blk_ptr = d_tq_pool +
                    ((size_t)block_idx * n_kv_heads * block_size + kv_head * block_size + tok_in_block) * 68;

                // Parse block_turbo4_0: norm(2) + rnorm(2) + qs[64]
                uint16_t nb;
                __builtin_memcpy(&nb, blk_ptr, 2);
                const uint8_t* qs = blk_ptr + 4;

                // Sequential 128-dim dot product
                float dot = 0.0f;
                for (int d = 0; d < 64; d++) {
                    uint8_t byte = qs[d];
                    uint8_t lo = byte & 0xF;
                    uint8_t hi = (byte >> 4) & 0xF;
                    dot += (*slm_q)[d * 2]     * d_tc4[lo];
                    dot += (*slm_q)[d * 2 + 1] * d_tc4[hi];
                }

                // Decode norm
                uint32_t sb = (uint32_t)(nb & 0x8000) << 16;
                uint32_t eb = (nb >> 10) & 0x1F;
                uint32_t mb = nb & 0x3FF;
                float nv;
                uint32_t r = (eb == 0) ? sb : (sb | ((eb + 112) << 23) | (mb << 13));
                __builtin_memcpy(&nv, &r, 4);

                d_scores[(b * n_q_heads + q_head) * max_seq_len + tok] = dot * nv;
            });
    });
}

// ============================================================================
// Paged K-score kernel: FP16 baseline
// ============================================================================

template <int WG_SIZE>
void launch_paged_k_scores_fp16(sycl::queue& q,
    const sycl::half* d_fp16_pool,
    const int* d_block_tables,
    const int* d_seq_lengths,
    const float* d_q,                  // [batch * n_q_heads * 128] (NOT rotated for FP16)
    float* d_scores,
    int batch_size, int n_q_heads, int n_kv_heads,
    int max_seq_len, int max_blocks_per_seq,
    int block_size)
{
    int gqa_ratio = n_q_heads / n_kv_heads;
    int wgs_per_head = (max_seq_len + WG_SIZE - 1) / WG_SIZE;
    int total_wgs = batch_size * n_q_heads * wgs_per_head;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int lid = item.get_local_id(0);
                auto grp = item.get_group();

                int b = wg_id / (n_q_heads * wgs_per_head);
                int rem = wg_id % (n_q_heads * wgs_per_head);
                int q_head = rem / wgs_per_head;
                int kv_head = q_head / gqa_ratio;
                int tile_start = (rem % wgs_per_head) * WG_SIZE;
                int tok = tile_start + lid;

                int this_seq_len = d_seq_lengths[b];

                auto slm_q = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                for (int i = lid; i < 128; i += WG_SIZE) {
                    (*slm_q)[i] = d_q[(b * n_q_heads + q_head) * 128 + i];
                }
                sycl::group_barrier(grp);

                if (tok >= this_seq_len) return;

                int block_idx = d_block_tables[b * max_blocks_per_seq + tok / block_size];
                int tok_in_block = tok % block_size;

                const sycl::half* kv = d_fp16_pool +
                    ((size_t)block_idx * n_kv_heads * block_size + kv_head * block_size + tok_in_block) * 128;

                float dot = 0.0f;
                for (int d = 0; d < 128; d++) {
                    dot += (*slm_q)[d] * static_cast<float>(kv[d]);
                }

                d_scores[(b * n_q_heads + q_head) * max_seq_len + tok] = dot;
            });
    });
}

// ============================================================================
// Paged softmax: one WG per (batch, q_head)
// ============================================================================

void launch_paged_softmax(sycl::queue& q,
    const float* d_scores,
    float* d_weights,
    const int* d_seq_lengths,
    int batch_size, int n_q_heads, int max_seq_len)
{
    constexpr int SM_WG = 256;
    int total_wgs = batch_size * n_q_heads;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * SM_WG, SM_WG),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int lid = item.get_local_id(0);
                auto grp = item.get_group();

                int b = wg_id / n_q_heads;
                int head = wg_id % n_q_heads;
                int n_tokens = d_seq_lengths[b];

                const float* s = d_scores + (b * n_q_heads + head) * max_seq_len;
                float* w = d_weights + (b * n_q_heads + head) * max_seq_len;

                // Max reduction
                float local_max = -1e30f;
                for (int t = lid; t < n_tokens; t += SM_WG) {
                    float v = s[t];
                    if (v > local_max) local_max = v;
                }

                auto slm = sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(grp);
                (*slm)[lid] = local_max;
                sycl::group_barrier(grp);
                for (int stride = SM_WG / 2; stride > 0; stride >>= 1) {
                    if (lid < (unsigned)stride) {
                        float a = (*slm)[lid], bv = (*slm)[lid + stride];
                        (*slm)[lid] = (a > bv) ? a : bv;
                    }
                    sycl::group_barrier(grp);
                }
                float global_max = (*slm)[0];
                sycl::group_barrier(grp);

                // Exp + sum
                float local_sum = 0.0f;
                for (int t = lid; t < n_tokens; t += SM_WG) {
                    float e = sycl::exp(s[t] - global_max);
                    w[t] = e;
                    local_sum += e;
                }
                (*slm)[lid] = local_sum;
                sycl::group_barrier(grp);
                for (int stride = SM_WG / 2; stride > 0; stride >>= 1) {
                    if (lid < (unsigned)stride) (*slm)[lid] += (*slm)[lid + stride];
                    sycl::group_barrier(grp);
                }
                float global_sum = (*slm)[0];
                sycl::group_barrier(grp);

                float inv_sum = 1.0f / global_sum;
                for (int t = lid; t < n_tokens; t += SM_WG) {
                    w[t] *= inv_sum;
                }
            });
    });
}

// ============================================================================
// Paged V-accumulate: TQ path (tiled, one thread per dimension)
// ============================================================================

template <int TILE_SIZE>
void launch_paged_v_accum_tq(sycl::queue& q,
    const uint8_t* d_tq_pool,
    const int* d_block_tables,
    const int* d_seq_lengths,
    const float* d_weights,
    const float* d_tc4,
    float* d_partial_acc,              // [batch * n_q_heads * max_tiles * 128]
    int batch_size, int n_q_heads, int n_kv_heads,
    int max_seq_len, int max_blocks_per_seq,
    int block_size, int max_tiles)
{
    int gqa_ratio = n_q_heads / n_kv_heads;
    int total_wgs = batch_size * n_q_heads * max_tiles;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * 128, 128),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int t = item.get_local_id(0);

                int b = wg_id / (n_q_heads * max_tiles);
                int rem = wg_id % (n_q_heads * max_tiles);
                int q_head = rem / max_tiles;
                int kv_head = q_head / gqa_ratio;
                int tile = rem % max_tiles;
                int tile_start = tile * TILE_SIZE;
                int tile_end = sycl::min(tile_start + TILE_SIZE, d_seq_lengths[b]);

                if (tile_start >= d_seq_lengths[b]) {
                    // This tile is beyond the sequence — write zero
                    d_partial_acc[((b * n_q_heads + q_head) * max_tiles + tile) * 128 + t] = 0.0f;
                    return;
                }

                float acc = 0.0f;
                for (int tok = tile_start; tok < tile_end; tok++) {
                    int block_idx = d_block_tables[b * max_blocks_per_seq + tok / block_size];
                    int tok_in_block = tok % block_size;

                    const uint8_t* blk_ptr = d_tq_pool +
                        ((size_t)block_idx * n_kv_heads * block_size + kv_head * block_size + tok_in_block) * 68;

                    uint16_t nb;
                    __builtin_memcpy(&nb, blk_ptr, 2);
                    const uint8_t* qs = blk_ptr + 4;

                    uint8_t idx = (t % 2 == 0) ? (qs[t/2] & 0xF) : ((qs[t/2] >> 4) & 0xF);
                    float centroid = d_tc4[idx];

                    uint32_t sb = (uint32_t)(nb & 0x8000) << 16;
                    uint32_t eb = (nb >> 10) & 0x1F;
                    uint32_t mb = nb & 0x3FF;
                    float nv;
                    uint32_t r = (eb == 0) ? sb : (sb | ((eb + 112) << 23) | (mb << 13));
                    __builtin_memcpy(&nv, &r, 4);

                    float w = d_weights[(b * n_q_heads + q_head) * max_seq_len + tok] * nv;
                    acc += w * centroid;
                }

                d_partial_acc[((b * n_q_heads + q_head) * max_tiles + tile) * 128 + t] = acc;
            });
    });
}

// ============================================================================
// Paged V-accumulate: FP16 baseline
// ============================================================================

template <int TILE_SIZE>
void launch_paged_v_accum_fp16(sycl::queue& q,
    const sycl::half* d_fp16_pool,
    const int* d_block_tables,
    const int* d_seq_lengths,
    const float* d_weights,
    float* d_partial_acc,
    int batch_size, int n_q_heads, int n_kv_heads,
    int max_seq_len, int max_blocks_per_seq,
    int block_size, int max_tiles)
{
    int gqa_ratio = n_q_heads / n_kv_heads;
    int total_wgs = batch_size * n_q_heads * max_tiles;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * 128, 128),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int t = item.get_local_id(0);

                int b = wg_id / (n_q_heads * max_tiles);
                int rem = wg_id % (n_q_heads * max_tiles);
                int q_head = rem / max_tiles;
                int kv_head = q_head / gqa_ratio;
                int tile = rem % max_tiles;
                int tile_start = tile * TILE_SIZE;
                int tile_end = sycl::min(tile_start + TILE_SIZE, d_seq_lengths[b]);

                if (tile_start >= d_seq_lengths[b]) {
                    d_partial_acc[((b * n_q_heads + q_head) * max_tiles + tile) * 128 + t] = 0.0f;
                    return;
                }

                float acc = 0.0f;
                for (int tok = tile_start; tok < tile_end; tok++) {
                    int block_idx = d_block_tables[b * max_blocks_per_seq + tok / block_size];
                    int tok_in_block = tok % block_size;

                    const sycl::half* v = d_fp16_pool +
                        ((size_t)block_idx * n_kv_heads * block_size + kv_head * block_size + tok_in_block) * 128;

                    float w = d_weights[(b * n_q_heads + q_head) * max_seq_len + tok];
                    acc += w * static_cast<float>(v[t]);
                }

                d_partial_acc[((b * n_q_heads + q_head) * max_tiles + tile) * 128 + t] = acc;
            });
    });
}

// ============================================================================
// Reduce tiles + inverse WHT (TQ path)
// ============================================================================

void launch_paged_reduce_wht(sycl::queue& q,
    const float* d_partial_acc,
    float* d_output,
    const float* d_ts1, const float* d_ts2,
    int batch_size, int n_q_heads, int max_tiles, int n_tiles_actual)
{
    int total_wgs = batch_size * n_q_heads;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * 128, 128),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int t = item.get_local_id(0);
                auto grp = item.get_group();

                int b = wg_id / n_q_heads;
                int head = wg_id % n_q_heads;

                float acc = 0.0f;
                for (int tile = 0; tile < n_tiles_actual; tile++)
                    acc += d_partial_acc[((b * n_q_heads + head) * max_tiles + tile) * 128 + t];

                // Inverse RHT
                acc *= d_ts2[t] * 0.08838834764831845f;

                auto local = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                (*local)[t] = acc;
                sycl::group_barrier(grp);
                for (int hv = 1; hv < 128; hv *= 2) {
                    if ((t % (2 * hv)) < hv) {
                        float a = (*local)[t], bv = (*local)[t + hv];
                        (*local)[t] = a + bv;
                        (*local)[t + hv] = a - bv;
                    }
                    sycl::group_barrier(grp);
                }

                d_output[(b * n_q_heads + head) * 128 + t] = (*local)[t] * d_ts1[t];
            });
    });
}

// ============================================================================
// Reduce tiles only (FP16 path — no WHT)
// ============================================================================

void launch_paged_reduce_only(sycl::queue& q,
    const float* d_partial_acc,
    float* d_output,
    int batch_size, int n_q_heads, int max_tiles, int n_tiles_actual)
{
    int total_wgs = batch_size * n_q_heads;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * 128, 128),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int t = item.get_local_id(0);

                int b = wg_id / n_q_heads;
                int head = wg_id % n_q_heads;

                float acc = 0.0f;
                for (int tile = 0; tile < n_tiles_actual; tile++)
                    acc += d_partial_acc[((b * n_q_heads + head) * max_tiles + tile) * 128 + t];

                d_output[(b * n_q_heads + head) * 128 + t] = acc;
            });
    });
}

// ============================================================================
// Helper: pack contiguous TQ blocks into paged pool with scattered block tables
// ============================================================================

struct PagedSetup {
    std::vector<uint8_t> tq_pool;      // raw bytes, laid out as [n_phys_blocks][n_kv_heads][block_size][68]
    std::vector<sycl::half> fp16_pool;  // [n_phys_blocks][n_kv_heads][block_size][128]
    std::vector<int> block_tables;      // [batch][max_blocks_per_seq]
    std::vector<int> seq_lengths;       // [batch]
    int n_physical_blocks;
    int max_blocks_per_seq;
};

PagedSetup create_paged_layout(
    const std::vector<block_turbo4_0>& k_blocks_contiguous,  // [n_kv_heads * total_tokens]
    const std::vector<sycl::half>& k_fp16_contiguous,        // [n_kv_heads * total_tokens * 128]
    const std::vector<int>& seq_lens,
    int n_kv_heads, int block_size, std::mt19937& rng)
{
    PagedSetup setup;
    int batch = seq_lens.size();
    setup.seq_lengths = seq_lens;

    // Calculate total blocks needed
    int total_blocks = 0;
    int max_blocks = 0;
    for (int s : seq_lens) {
        int blocks_for_seq = (s + block_size - 1) / block_size;
        total_blocks += blocks_for_seq;
        max_blocks = std::max(max_blocks, blocks_for_seq);
    }
    setup.max_blocks_per_seq = max_blocks;
    setup.n_physical_blocks = total_blocks;

    // Allocate and shuffle physical blocks
    std::vector<int> block_pool(total_blocks);
    std::iota(block_pool.begin(), block_pool.end(), 0);
    std::shuffle(block_pool.begin(), block_pool.end(), rng);

    // Build block tables
    setup.block_tables.resize(batch * max_blocks, -1);
    int pool_idx = 0;
    for (int b = 0; b < batch; b++) {
        int blocks_for_seq = (seq_lens[b] + block_size - 1) / block_size;
        for (int i = 0; i < blocks_for_seq; i++) {
            setup.block_tables[b * max_blocks + i] = block_pool[pool_idx++];
        }
    }

    // Allocate pools
    size_t tq_block_bytes = 68;
    size_t tq_pool_size = (size_t)total_blocks * n_kv_heads * block_size * tq_block_bytes;
    size_t fp16_pool_size = (size_t)total_blocks * n_kv_heads * block_size * 128;
    setup.tq_pool.resize(tq_pool_size, 0);
    setup.fp16_pool.resize(fp16_pool_size, sycl::half(0.0f));

    // Copy contiguous data into scattered paged layout
    // contiguous layout: k_blocks_contiguous[kv_head * total_tokens_for_all_seqs + token_offset]
    // But we generate per-sequence, so we track offsets
    int token_offset = 0;
    for (int b = 0; b < batch; b++) {
        int blocks_for_seq = (seq_lens[b] + block_size - 1) / block_size;
        for (int logical_block = 0; logical_block < blocks_for_seq; logical_block++) {
            int phys_block = setup.block_tables[b * max_blocks + logical_block];
            int tok_start = logical_block * block_size;
            int tok_end = std::min(tok_start + block_size, seq_lens[b]);

            for (int kv_h = 0; kv_h < n_kv_heads; kv_h++) {
                for (int t = tok_start; t < tok_end; t++) {
                    int tok_in_block = t - tok_start;
                    int contiguous_idx = kv_h * seq_lens[b] + t;  // within this batch element

                    // TQ: copy 68 bytes
                    size_t pool_offset = ((size_t)phys_block * n_kv_heads * block_size +
                                          kv_h * block_size + tok_in_block) * tq_block_bytes;
                    const auto& src_blk = k_blocks_contiguous[token_offset * n_kv_heads + contiguous_idx];
                    std::memcpy(setup.tq_pool.data() + pool_offset, &src_blk, tq_block_bytes);

                    // FP16: copy 128 halves
                    size_t fp16_offset = ((size_t)phys_block * n_kv_heads * block_size +
                                          kv_h * block_size + tok_in_block) * 128;
                    size_t fp16_src = (token_offset * n_kv_heads + contiguous_idx) * 128;
                    std::memcpy(setup.fp16_pool.data() + fp16_offset,
                               k_fp16_contiguous.data() + fp16_src,
                               128 * sizeof(sycl::half));
                }
            }
        }
        token_offset += seq_lens[b];
    }

    return setup;
}

// ============================================================================
// Correctness test: paged vs contiguous
// ============================================================================

bool run_correctness_tests() {
    std::cout << "\n=== Phase 2 Stage 1b Correctness Tests ===" << std::endl;
    bool all_pass = true;

    std::mt19937 rng(456);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const int n_q_heads = 8;
    const int n_kv_heads = 2;  // GQA 4:1
    const int n_tokens = 96;
    const int block_size = 16;
    const int batch = 1;

    // Generate queries
    std::vector<float> queries(batch * n_q_heads * 128);
    for (auto& v : queries) v = dist(rng);

    // Rotated queries for TQ path
    std::vector<float> q_rot(batch * n_q_heads * 128);
    for (int i = 0; i < batch * n_q_heads; i++) {
        std::memcpy(q_rot.data() + i * 128, queries.data() + i * 128, 128 * sizeof(float));
        tq_ref::forward_rht(q_rot.data() + i * 128);
    }

    // Generate K,V blocks (contiguous, per KV head)
    int total_kv = n_kv_heads * n_tokens;
    std::vector<block_turbo4_0> k_blocks(total_kv);
    std::vector<block_turbo4_0> v_blocks(total_kv);
    std::vector<sycl::half> k_fp16(total_kv * 128);
    std::vector<sycl::half> v_fp16(total_kv * 128);

    for (int i = 0; i < total_kv; i++) {
        float vec[128];
        float scale = 0.5f + (rng() % 1000) / 1000.0f;
        for (int d = 0; d < 128; d++) vec[d] = dist(rng) * scale;
        tq_ref::quantize_vector_turbo4(vec, k_blocks[i]);
        // Dequant to FP16
        float tmp[128];
        tq_ref::dequantize_vector_turbo4(k_blocks[i], tmp);
        for (int d = 0; d < 128; d++) k_fp16[i * 128 + d] = sycl::half(tmp[d]);

        for (int d = 0; d < 128; d++) vec[d] = dist(rng) * scale;
        tq_ref::quantize_vector_turbo4(vec, v_blocks[i]);
        tq_ref::dequantize_vector_turbo4(v_blocks[i], tmp);
        for (int d = 0; d < 128; d++) v_fp16[i * 128 + d] = sycl::half(tmp[d]);
    }

    // CPU reference: TQ rotated-space attention with GQA
    std::vector<float> cpu_output(batch * n_q_heads * 128, 0.0f);
    for (int b = 0; b < batch; b++) {
        for (int qh = 0; qh < n_q_heads; qh++) {
            int kv_h = qh / (n_q_heads / n_kv_heads);
            float* qr = q_rot.data() + (b * n_q_heads + qh) * 128;

            // K-scores in rotated space
            std::vector<float> scores(n_tokens);
            for (int t = 0; t < n_tokens; t++) {
                const auto& blk = k_blocks[kv_h * n_tokens + t];
                uint8_t indices[128];
                tq_ref::unpack_turbo4(blk, indices);
                float dot = 0.0f;
                for (int d = 0; d < 128; d++) dot += qr[d] * TC4[indices[d]];
                scores[t] = dot * fp16_to_fp32(blk.norm);
            }

            // Softmax
            float mx = *std::max_element(scores.begin(), scores.end());
            float sum = 0.0f;
            std::vector<float> weights(n_tokens);
            for (int t = 0; t < n_tokens; t++) {
                weights[t] = std::exp(scores[t] - mx);
                sum += weights[t];
            }
            for (int t = 0; t < n_tokens; t++) weights[t] /= sum;

            // V-accum in rotated space
            float acc[128] = {};
            for (int t = 0; t < n_tokens; t++) {
                const auto& blk = v_blocks[kv_h * n_tokens + t];
                uint8_t indices[128];
                tq_ref::unpack_turbo4(blk, indices);
                float nv = fp16_to_fp32(blk.norm);
                float w = weights[t] * nv;
                for (int d = 0; d < 128; d++) acc[d] += w * TC4[indices[d]];
            }

            // Inverse RHT
            tq_ref::inverse_rht(acc);
            std::memcpy(cpu_output.data() + (b * n_q_heads + qh) * 128, acc, 128 * sizeof(float));
        }
    }

    // Create paged layout
    std::vector<int> seq_lens = {n_tokens};
    PagedSetup k_paged = create_paged_layout(k_blocks, k_fp16, seq_lens, n_kv_heads, block_size, rng);
    PagedSetup v_paged = create_paged_layout(v_blocks, v_fp16, seq_lens, n_kv_heads, block_size, rng);

    sycl::queue q_dev(sycl::gpu_selector_v, sycl::property::queue::in_order());

    // Test 1: Paged TQ attention vs CPU reference
    {
        auto* d_tq_k = sycl::malloc_device<uint8_t>(k_paged.tq_pool.size(), q_dev);
        auto* d_tq_v = sycl::malloc_device<uint8_t>(v_paged.tq_pool.size(), q_dev);
        auto* d_bt_k = sycl::malloc_device<int>(k_paged.block_tables.size(), q_dev);
        auto* d_bt_v = sycl::malloc_device<int>(v_paged.block_tables.size(), q_dev);
        auto* d_sl = sycl::malloc_device<int>(batch, q_dev);
        auto* d_tc4 = sycl::malloc_device<float>(16, q_dev);
        auto* d_ts1 = sycl::malloc_device<float>(128, q_dev);
        auto* d_ts2 = sycl::malloc_device<float>(128, q_dev);
        auto* d_qr = sycl::malloc_device<float>(batch * n_q_heads * 128, q_dev);
        int max_seq = n_tokens;
        auto* d_scores = sycl::malloc_device<float>(batch * n_q_heads * max_seq, q_dev);
        auto* d_weights = sycl::malloc_device<float>(batch * n_q_heads * max_seq, q_dev);
        constexpr int TILE = 64;
        int max_tiles = (max_seq + TILE - 1) / TILE;
        auto* d_partial = sycl::malloc_device<float>(batch * n_q_heads * max_tiles * 128, q_dev);
        auto* d_output = sycl::malloc_device<float>(batch * n_q_heads * 128, q_dev);

        q_dev.memcpy(d_tq_k, k_paged.tq_pool.data(), k_paged.tq_pool.size());
        q_dev.memcpy(d_tq_v, v_paged.tq_pool.data(), v_paged.tq_pool.size());
        q_dev.memcpy(d_bt_k, k_paged.block_tables.data(), k_paged.block_tables.size() * sizeof(int));
        q_dev.memcpy(d_bt_v, v_paged.block_tables.data(), v_paged.block_tables.size() * sizeof(int));
        q_dev.memcpy(d_sl, seq_lens.data(), batch * sizeof(int));
        q_dev.memcpy(d_tc4, TC4, 16 * sizeof(float));
        q_dev.memcpy(d_ts1, TS1, 128 * sizeof(float));
        q_dev.memcpy(d_ts2, TS2, 128 * sizeof(float));
        q_dev.memcpy(d_qr, q_rot.data(), batch * n_q_heads * 128 * sizeof(float));
        q_dev.wait();

        launch_paged_k_scores_tq<256>(q_dev, d_tq_k, d_bt_k, d_sl, d_qr, d_tc4, d_scores,
            batch, n_q_heads, n_kv_heads, max_seq, k_paged.max_blocks_per_seq, block_size);
        q_dev.wait();
        launch_paged_softmax(q_dev, d_scores, d_weights, d_sl, batch, n_q_heads, max_seq);
        q_dev.wait();
        launch_paged_v_accum_tq<TILE>(q_dev, d_tq_v, d_bt_v, d_sl, d_weights, d_tc4, d_partial,
            batch, n_q_heads, n_kv_heads, max_seq, v_paged.max_blocks_per_seq, block_size, max_tiles);
        q_dev.wait();
        int n_tiles_actual = (n_tokens + TILE - 1) / TILE;
        launch_paged_reduce_wht(q_dev, d_partial, d_output, d_ts1, d_ts2,
            batch, n_q_heads, max_tiles, n_tiles_actual);
        q_dev.wait();

        std::vector<float> gpu_output(batch * n_q_heads * 128);
        q_dev.memcpy(gpu_output.data(), d_output, gpu_output.size() * sizeof(float)).wait();

        float max_err = 0.0f, max_val = 0.0f;
        for (size_t i = 0; i < gpu_output.size(); i++) {
            max_err = std::max(max_err, std::fabs(gpu_output[i] - cpu_output[i]));
            max_val = std::max(max_val, std::fabs(cpu_output[i]));
        }
        float rel_err = (max_val > 1e-10f) ? (max_err / max_val) : max_err;

        bool pass = rel_err < 1e-3f;
        std::cout << "Test 1: Paged TQ attention vs CPU reference (GQA " << n_q_heads << ":" << n_kv_heads << ")" << std::endl;
        std::cout << "  Max abs error: " << std::scientific << max_err << std::endl;
        std::cout << "  Max rel error: " << rel_err << std::endl;
        std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;

        sycl::free(d_tq_k, q_dev); sycl::free(d_tq_v, q_dev);
        sycl::free(d_bt_k, q_dev); sycl::free(d_bt_v, q_dev);
        sycl::free(d_sl, q_dev); sycl::free(d_tc4, q_dev);
        sycl::free(d_ts1, q_dev); sycl::free(d_ts2, q_dev);
        sycl::free(d_qr, q_dev); sycl::free(d_scores, q_dev);
        sycl::free(d_weights, q_dev); sycl::free(d_partial, q_dev);
        sycl::free(d_output, q_dev);
    }

    std::cout << "\n=== All " << (all_pass ? "PASSED" : "FAILED") << " ===" << std::endl;
    return all_pass;
}

// ============================================================================
// Benchmark runner
// ============================================================================

struct BenchConfig {
    std::string label;
    int block_size;
    int batch_size;
    std::vector<int> seq_lengths;
    int n_q_heads;
    int n_kv_heads;
};

struct BenchResult {
    std::string label;
    double tq_us;
    double fp16_us;
    double ratio;
    int total_tokens;
};

BenchResult run_one_bench(sycl::queue& q_dev, const BenchConfig& cfg, int warmup, int iters,
                           std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);

    int batch = cfg.batch_size;
    int n_q_heads = cfg.n_q_heads;
    int n_kv_heads = cfg.n_kv_heads;
    int block_size = cfg.block_size;
    int max_seq = *std::max_element(cfg.seq_lengths.begin(), cfg.seq_lengths.end());
    int total_tokens = 0;
    for (int s : cfg.seq_lengths) total_tokens += s;

    // Generate data per batch element
    std::vector<block_turbo4_0> all_k_blocks;
    std::vector<block_turbo4_0> all_v_blocks;
    std::vector<sycl::half> all_k_fp16;
    std::vector<sycl::half> all_v_fp16;

    for (int b = 0; b < batch; b++) {
        int n_tok = cfg.seq_lengths[b];
        for (int i = 0; i < n_kv_heads * n_tok; i++) {
            float vec[128];
            float scale = 0.5f + (rng() % 1000) / 1000.0f;
            for (int d = 0; d < 128; d++) vec[d] = dist(rng) * scale;
            block_turbo4_0 blk;
            tq_ref::quantize_vector_turbo4(vec, blk);
            all_k_blocks.push_back(blk);
            float tmp[128];
            tq_ref::dequantize_vector_turbo4(blk, tmp);
            for (int d = 0; d < 128; d++) all_k_fp16.push_back(sycl::half(tmp[d]));

            for (int d = 0; d < 128; d++) vec[d] = dist(rng) * scale;
            block_turbo4_0 vblk;
            tq_ref::quantize_vector_turbo4(vec, vblk);
            all_v_blocks.push_back(vblk);
            tq_ref::dequantize_vector_turbo4(vblk, tmp);
            for (int d = 0; d < 128; d++) all_v_fp16.push_back(sycl::half(tmp[d]));
        }
    }

    // Create paged layouts (K and V get independent block tables)
    // We need per-batch-element contiguous blocks, so split and create per element
    // Actually create_paged_layout handles multi-seq already
    PagedSetup k_paged = create_paged_layout(all_k_blocks, all_k_fp16, cfg.seq_lengths, n_kv_heads, block_size, rng);
    PagedSetup v_paged = create_paged_layout(all_v_blocks, all_v_fp16, cfg.seq_lengths, n_kv_heads, block_size, rng);

    // Queries
    std::vector<float> queries(batch * n_q_heads * 128);
    for (auto& v : queries) v = dist(rng);
    std::vector<float> q_rot(batch * n_q_heads * 128);
    for (int i = 0; i < batch * n_q_heads; i++) {
        std::memcpy(q_rot.data() + i * 128, queries.data() + i * 128, 128 * sizeof(float));
        tq_ref::forward_rht(q_rot.data() + i * 128);
    }

    // Device allocations
    auto* d_tq_k = sycl::malloc_device<uint8_t>(k_paged.tq_pool.size(), q_dev);
    auto* d_tq_v = sycl::malloc_device<uint8_t>(v_paged.tq_pool.size(), q_dev);
    auto* d_fp16_k = sycl::malloc_device<sycl::half>(k_paged.fp16_pool.size(), q_dev);
    auto* d_fp16_v = sycl::malloc_device<sycl::half>(v_paged.fp16_pool.size(), q_dev);
    auto* d_bt_k = sycl::malloc_device<int>(k_paged.block_tables.size(), q_dev);
    auto* d_bt_v = sycl::malloc_device<int>(v_paged.block_tables.size(), q_dev);
    auto* d_sl = sycl::malloc_device<int>(batch, q_dev);
    auto* d_tc4 = sycl::malloc_device<float>(16, q_dev);
    auto* d_ts1 = sycl::malloc_device<float>(128, q_dev);
    auto* d_ts2 = sycl::malloc_device<float>(128, q_dev);
    auto* d_qr = sycl::malloc_device<float>(batch * n_q_heads * 128, q_dev);
    auto* d_qp = sycl::malloc_device<float>(batch * n_q_heads * 128, q_dev);
    auto* d_scores = sycl::malloc_device<float>(batch * n_q_heads * max_seq, q_dev);
    auto* d_weights = sycl::malloc_device<float>(batch * n_q_heads * max_seq, q_dev);
    constexpr int TILE = 64;
    int max_tiles = (max_seq + TILE - 1) / TILE;
    auto* d_partial = sycl::malloc_device<float>(batch * n_q_heads * max_tiles * 128, q_dev);
    auto* d_output = sycl::malloc_device<float>(batch * n_q_heads * 128, q_dev);

    q_dev.memcpy(d_tq_k, k_paged.tq_pool.data(), k_paged.tq_pool.size());
    q_dev.memcpy(d_tq_v, v_paged.tq_pool.data(), v_paged.tq_pool.size());
    q_dev.memcpy(d_fp16_k, k_paged.fp16_pool.data(), k_paged.fp16_pool.size() * sizeof(sycl::half));
    q_dev.memcpy(d_fp16_v, v_paged.fp16_pool.data(), v_paged.fp16_pool.size() * sizeof(sycl::half));
    q_dev.memcpy(d_bt_k, k_paged.block_tables.data(), k_paged.block_tables.size() * sizeof(int));
    q_dev.memcpy(d_bt_v, v_paged.block_tables.data(), v_paged.block_tables.size() * sizeof(int));
    q_dev.memcpy(d_sl, cfg.seq_lengths.data(), batch * sizeof(int));
    q_dev.memcpy(d_tc4, TC4, 16 * sizeof(float));
    q_dev.memcpy(d_ts1, TS1, 128 * sizeof(float));
    q_dev.memcpy(d_ts2, TS2, 128 * sizeof(float));
    q_dev.memcpy(d_qr, q_rot.data(), batch * n_q_heads * 128 * sizeof(float));
    q_dev.memcpy(d_qp, queries.data(), batch * n_q_heads * 128 * sizeof(float));
    q_dev.wait();

    int n_tiles_max = max_tiles;  // for reduce kernel

    // --- TQ pipeline ---
    auto run_tq = [&]() {
        launch_paged_k_scores_tq<256>(q_dev, d_tq_k, d_bt_k, d_sl, d_qr, d_tc4, d_scores,
            batch, n_q_heads, n_kv_heads, max_seq, k_paged.max_blocks_per_seq, block_size);
        q_dev.wait();
        launch_paged_softmax(q_dev, d_scores, d_weights, d_sl, batch, n_q_heads, max_seq);
        q_dev.wait();
        launch_paged_v_accum_tq<TILE>(q_dev, d_tq_v, d_bt_v, d_sl, d_weights, d_tc4, d_partial,
            batch, n_q_heads, n_kv_heads, max_seq, v_paged.max_blocks_per_seq, block_size, n_tiles_max);
        q_dev.wait();
        // Use max_seq for n_tiles_actual per batch element — conservative but correct
        int n_tiles_actual = (max_seq + TILE - 1) / TILE;
        launch_paged_reduce_wht(q_dev, d_partial, d_output, d_ts1, d_ts2,
            batch, n_q_heads, n_tiles_max, n_tiles_actual);
        q_dev.wait();
    };

    for (int i = 0; i < warmup; i++) run_tq();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) run_tq();
    auto t1 = std::chrono::high_resolution_clock::now();
    double tq_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    // --- FP16 pipeline ---
    auto run_fp16 = [&]() {
        launch_paged_k_scores_fp16<256>(q_dev, d_fp16_k, d_bt_k, d_sl, d_qp, d_scores,
            batch, n_q_heads, n_kv_heads, max_seq, k_paged.max_blocks_per_seq, block_size);
        q_dev.wait();
        launch_paged_softmax(q_dev, d_scores, d_weights, d_sl, batch, n_q_heads, max_seq);
        q_dev.wait();
        launch_paged_v_accum_fp16<TILE>(q_dev, d_fp16_v, d_bt_v, d_sl, d_weights, d_partial,
            batch, n_q_heads, n_kv_heads, max_seq, v_paged.max_blocks_per_seq, block_size, n_tiles_max);
        q_dev.wait();
        int n_tiles_actual = (max_seq + TILE - 1) / TILE;
        launch_paged_reduce_only(q_dev, d_partial, d_output, batch, n_q_heads, n_tiles_max, n_tiles_actual);
        q_dev.wait();
    };

    for (int i = 0; i < warmup; i++) run_fp16();

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) run_fp16();
    t1 = std::chrono::high_resolution_clock::now();
    double fp16_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    // Cleanup
    sycl::free(d_tq_k, q_dev); sycl::free(d_tq_v, q_dev);
    sycl::free(d_fp16_k, q_dev); sycl::free(d_fp16_v, q_dev);
    sycl::free(d_bt_k, q_dev); sycl::free(d_bt_v, q_dev);
    sycl::free(d_sl, q_dev); sycl::free(d_tc4, q_dev);
    sycl::free(d_ts1, q_dev); sycl::free(d_ts2, q_dev);
    sycl::free(d_qr, q_dev); sycl::free(d_qp, q_dev);
    sycl::free(d_scores, q_dev); sycl::free(d_weights, q_dev);
    sycl::free(d_partial, q_dev); sycl::free(d_output, q_dev);

    BenchResult r;
    r.label = cfg.label;
    r.tq_us = tq_us;
    r.fp16_us = fp16_us;
    r.ratio = (fp16_us > 0) ? tq_us / fp16_us : 0;
    r.total_tokens = total_tokens;
    return r;
}

void run_benchmarks() {
    std::cout << "\n=== Phase 2 Stage 1b Benchmarks ===" << std::endl;

    sycl::queue q_dev(sycl::gpu_selector_v, sycl::property::queue::in_order());
    auto dev = q_dev.get_device();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Global mem: " << dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024) << " MB\n";

    // Qwen3.6-35B-A3B linear attention config (head_dim=128)
    // Using 16 Q heads, 4 KV heads as a realistic GQA config for TQ
    // (actual model has complex mix of linear/full attn, we test GQA overhead)
    const int n_q_heads = 16;
    const int n_kv_heads = 4;  // GQA 4:1

    const int warmup = 5;
    const int iters = 50;
    std::mt19937 rng(42);

    std::vector<BenchResult> all_results;
    std::vector<BenchConfig> configs;

    // --- Block size sweep at 14K, batch=1 ---
    configs.push_back({"bs16_14K_b1", 16, 1, {14000}, n_q_heads, n_kv_heads});
    configs.push_back({"bs32_14K_b1", 32, 1, {14000}, n_q_heads, n_kv_heads});

    // --- Block size sweep at 60K, batch=1 ---
    configs.push_back({"bs16_60K_b1", 16, 1, {60000}, n_q_heads, n_kv_heads});
    configs.push_back({"bs32_60K_b1", 32, 1, {60000}, n_q_heads, n_kv_heads});

    // --- Batch size sweep at block_size=16, 14K tokens per seq ---
    configs.push_back({"bs16_14K_b2", 16, 2, {14000, 14000}, n_q_heads, n_kv_heads});
    configs.push_back({"bs16_14K_b4", 16, 4, {14000, 14000, 14000, 14000}, n_q_heads, n_kv_heads});
    configs.push_back({"bs16_14K_b8", 16, 8, {14000, 14000, 14000, 14000, 14000, 14000, 14000, 14000}, n_q_heads, n_kv_heads});

    // --- Mixed sequence lengths, batch=4, block_size=16 ---
    configs.push_back({"bs16_mixed1_b4", 16, 4, {14000, 8000, 14000, 2000}, n_q_heads, n_kv_heads});
    configs.push_back({"bs16_mixed2_b4", 16, 4, {60000, 4000, 60000, 4000}, n_q_heads, n_kv_heads});

    std::cout << "\nGQA config: " << n_q_heads << " Q heads, " << n_kv_heads << " KV heads (ratio "
              << n_q_heads/n_kv_heads << ":1)\n" << std::endl;

    std::cout << std::setw(22) << "config"
              << std::setw(10) << "total_tok"
              << std::setw(12) << "TQ(us)"
              << std::setw(12) << "FP16(us)"
              << std::setw(10) << "ratio"
              << std::setw(12) << "OH_14K(ms)"
              << std::endl;
    std::cout << std::string(78, '-') << std::endl;

    for (auto& cfg : configs) {
        // Check memory feasibility
        int total_tokens = 0;
        for (int s : cfg.seq_lengths) total_tokens += s;
        size_t mem_est = (size_t)total_tokens * n_kv_heads * 2 * (68 + 256);  // K+V, TQ+FP16
        size_t dev_mem = dev.get_info<sycl::info::device::global_mem_size>();
        if (mem_est > dev_mem * 0.7) {
            std::cout << std::setw(22) << cfg.label << "  SKIP (need ~"
                      << mem_est/(1024*1024) << " MB)" << std::endl;
            continue;
        }

        BenchResult r = run_one_bench(q_dev, cfg, warmup, iters, rng);
        all_results.push_back(r);

        double oh_14k = (r.tq_us - r.fp16_us) * 14000.0 / r.total_tokens / 1000.0;

        std::cout << std::setw(22) << r.label
                  << std::setw(10) << r.total_tokens
                  << std::fixed
                  << std::setw(12) << std::setprecision(1) << r.tq_us
                  << std::setw(12) << std::setprecision(1) << r.fp16_us
                  << std::setw(10) << std::setprecision(3) << r.ratio << "x"
                  << std::setw(12) << std::setprecision(3) << oh_14k
                  << std::endl;
    }

    // Summary tables
    std::cout << "\n============================================" << std::endl;
    std::cout << "BLOCK SIZE SWEEP (batch=1)" << std::endl;
    std::cout << "============================================" << std::endl;
    for (const auto& r : all_results) {
        if (r.label.find("_b1") != std::string::npos) {
            std::cout << "  " << std::setw(16) << r.label
                      << "  TQ=" << std::setprecision(1) << r.tq_us << " us"
                      << "  FP16=" << r.fp16_us << " us"
                      << "  ratio=" << std::setprecision(3) << r.ratio << "x"
                      << std::endl;
        }
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "BATCH SIZE SWEEP (block_size=16, 14K/seq)" << std::endl;
    std::cout << "============================================" << std::endl;
    for (const auto& r : all_results) {
        if (r.label.find("bs16_14K") != std::string::npos) {
            std::cout << "  " << std::setw(16) << r.label
                      << "  TQ=" << std::setprecision(1) << r.tq_us << " us"
                      << "  FP16=" << r.fp16_us << " us"
                      << "  ratio=" << std::setprecision(3) << r.ratio << "x"
                      << std::endl;
        }
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "MIXED SEQUENCE LENGTHS (block_size=16, batch=4)" << std::endl;
    std::cout << "============================================" << std::endl;
    for (const auto& r : all_results) {
        if (r.label.find("mixed") != std::string::npos) {
            std::cout << "  " << std::setw(16) << r.label
                      << "  TQ=" << std::setprecision(1) << r.tq_us << " us"
                      << "  FP16=" << r.fp16_us << " us"
                      << "  ratio=" << std::setprecision(3) << r.ratio << "x"
                      << std::endl;
        }
    }

    // Stage 1 vs 1b comparison
    std::cout << "\n============================================" << std::endl;
    std::cout << "CONTIGUOUS (Stage 1) vs PAGED (Stage 1b)" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Stage 1 reference (contiguous, n_heads=40, no GQA):" << std::endl;
    std::cout << "  14K: TQ=482.9 us, FP16=736.2 us, ratio=0.66x" << std::endl;
    std::cout << "  60K: TQ=1979.6 us, FP16=3074.2 us, ratio=0.64x" << std::endl;
    std::cout << "Stage 1b paged (block_size=16, GQA " << n_q_heads << ":" << n_kv_heads << "):" << std::endl;
    for (const auto& r : all_results) {
        if (r.label == "bs16_14K_b1" || r.label == "bs16_60K_b1") {
            std::cout << "  " << r.total_tokens/1000 << "K: TQ=" << std::setprecision(1) << r.tq_us
                      << " us, FP16=" << r.fp16_us << " us, ratio=" << std::setprecision(3) << r.ratio << "x"
                      << std::endl;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "============================================" << std::endl;
    std::cout << "TurboQuant Phase 2 Stage 1b: Paged KV Cache" << std::endl;
    std::cout << "============================================" << std::endl;

    bool skip_bench = false;
    bool skip_correctness = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--skip-bench") skip_bench = true;
        else if (arg == "--skip-correctness") skip_correctness = true;
    }

    if (!skip_correctness) {
        bool ok = run_correctness_tests();
        if (!ok) {
            std::cerr << "Correctness tests failed, aborting benchmark." << std::endl;
            return 1;
        }
    }

    if (!skip_bench) {
        run_benchmarks();
    }

    return 0;
}
