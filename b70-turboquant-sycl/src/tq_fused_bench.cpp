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
// TurboQuant Phase 2 Stage 1: Fused Rotated-Space Attention
//
// Key difference from Phase 0.5:
//   Phase 0.5 K-score: 128 threads per token, tree reduction (7 barriers)
//   Phase 2 K-score: 1 thread per token, sequential 128-dim dot product
//
// Kernels:
//   1. TQ K-score fused: each thread handles one (head, token), sequential dot
//   2. GPU softmax: two-pass (max → exp/sum → normalize), one WG per head
//   3. TQ V-accumulate: tiled, one thread per dimension (reuse Phase 0.5)
//   4. Reduce + WHT: reduce tile accumulators + inverse WHT
//   5. FP16 baseline versions for overhead measurement
// ============================================================================

// ============================================================================
// Kernel 1: TQ K-score — fused sequential dot product
// WG_SIZE threads per WG, each thread handles one token for one head.
// q_rot is loaded into SLM cooperatively, then each thread loops d=0..63,
// reads one byte (two nibbles), two codebook lookups, two FMAs.
// ============================================================================

template <int WG_SIZE>
void launch_fused_k_scores_tq(sycl::queue& q,
    const block_turbo4_0* d_k_blocks,  // [n_heads * n_tokens]
    const float* d_q_rot,              // [n_heads * 128]
    const float* d_tc4,                // [16] codebook
    float* d_scores,                   // [n_heads * n_tokens]
    int n_heads, int n_tokens)
{
    int wgs_per_head = (n_tokens + WG_SIZE - 1) / WG_SIZE;
    int total_wgs = n_heads * wgs_per_head;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int lid = item.get_local_id(0);
                auto grp = item.get_group();

                int head = wg_id / wgs_per_head;
                int tile_start = (wg_id % wgs_per_head) * WG_SIZE;
                int tok = tile_start + lid;

                // Cooperatively load q_rot[head*128..+128] into SLM
                auto slm_q = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                // Each thread loads ceil(128/WG_SIZE) elements
                for (int i = lid; i < 128; i += WG_SIZE) {
                    (*slm_q)[i] = d_q_rot[head * 128 + i];
                }
                sycl::group_barrier(grp);

                if (tok >= n_tokens) return;

                // Sequential 128-element dot product
                const auto& blk = d_k_blocks[head * n_tokens + tok];
                float dot = 0.0f;

                for (int d = 0; d < 64; d++) {
                    uint8_t byte = blk.qs[d];
                    uint8_t lo = byte & 0xF;
                    uint8_t hi = (byte >> 4) & 0xF;
                    dot += (*slm_q)[d * 2]     * d_tc4[lo];
                    dot += (*slm_q)[d * 2 + 1] * d_tc4[hi];
                }

                // Multiply by norm
                uint16_t nb = blk.norm;
                uint32_t sb = (uint32_t)(nb & 0x8000) << 16;
                uint32_t eb = (nb >> 10) & 0x1F;
                uint32_t mb = nb & 0x3FF;
                float nv;
                uint32_t r = (eb == 0) ? sb : (sb | ((eb + 112) << 23) | (mb << 13));
                __builtin_memcpy(&nv, &r, 4);

                d_scores[head * n_tokens + tok] = dot * nv;
            });
    });
}

// ============================================================================
// Kernel 1b: FP16 K-score baseline — same structure, reads FP16 KV
// K stored as sycl::half[n_heads * n_tokens * 128]
// ============================================================================

template <int WG_SIZE>
void launch_fused_k_scores_fp16(sycl::queue& q,
    const sycl::half* d_k_fp16,       // [n_heads * n_tokens * 128]
    const float* d_q_rot,              // [n_heads * 128]
    float* d_scores,                   // [n_heads * n_tokens]
    int n_heads, int n_tokens)
{
    int wgs_per_head = (n_tokens + WG_SIZE - 1) / WG_SIZE;
    int total_wgs = n_heads * wgs_per_head;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int lid = item.get_local_id(0);
                auto grp = item.get_group();

                int head = wg_id / wgs_per_head;
                int tile_start = (wg_id % wgs_per_head) * WG_SIZE;
                int tok = tile_start + lid;

                // Cooperatively load q_rot into SLM
                auto slm_q = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                for (int i = lid; i < 128; i += WG_SIZE) {
                    (*slm_q)[i] = d_q_rot[head * 128 + i];
                }
                sycl::group_barrier(grp);

                if (tok >= n_tokens) return;

                // Sequential 128-element dot product reading FP16
                const sycl::half* kv = d_k_fp16 + (head * n_tokens + tok) * 128;
                float dot = 0.0f;
                for (int d = 0; d < 128; d++) {
                    dot += (*slm_q)[d] * static_cast<float>(kv[d]);
                }

                d_scores[head * n_tokens + tok] = dot;
            });
    });
}

// ============================================================================
// Kernel 2: GPU Softmax — two-pass, one WG per head
// Pass 1: find max score (tree reduction)
// Pass 2: exp(score - max), sum, normalize
// ============================================================================

void launch_softmax_gpu(sycl::queue& q,
    const float* d_scores,   // [n_heads * n_tokens]
    float* d_weights,        // [n_heads * n_tokens]
    int n_heads, int n_tokens)
{
    // Use 256 threads per WG, each processes ceil(n_tokens/256) elements
    constexpr int SM_WG = 256;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(n_heads * SM_WG, SM_WG),
            [=](sycl::nd_item<1> item) {
                int head = item.get_group(0);
                int lid = item.get_local_id(0);
                auto grp = item.get_group();

                const float* s = d_scores + head * n_tokens;
                float* w = d_weights + head * n_tokens;

                // Pass 1: thread-local max
                float local_max = -1e30f;
                for (int t = lid; t < n_tokens; t += SM_WG) {
                    float v = s[t];
                    if (v > local_max) local_max = v;
                }

                // Tree reduction for max
                auto slm = sycl::ext::oneapi::group_local_memory_for_overwrite<float[256]>(grp);
                (*slm)[lid] = local_max;
                sycl::group_barrier(grp);

                for (int stride = SM_WG / 2; stride > 0; stride >>= 1) {
                    if (lid < (unsigned)stride) {
                        float a = (*slm)[lid], b = (*slm)[lid + stride];
                        (*slm)[lid] = (a > b) ? a : b;
                    }
                    sycl::group_barrier(grp);
                }
                float global_max = (*slm)[0];
                sycl::group_barrier(grp);

                // Pass 2: exp and local sum
                float local_sum = 0.0f;
                for (int t = lid; t < n_tokens; t += SM_WG) {
                    float e = sycl::exp(s[t] - global_max);
                    w[t] = e;
                    local_sum += e;
                }

                // Tree reduction for sum
                (*slm)[lid] = local_sum;
                sycl::group_barrier(grp);

                for (int stride = SM_WG / 2; stride > 0; stride >>= 1) {
                    if (lid < (unsigned)stride) {
                        (*slm)[lid] += (*slm)[lid + stride];
                    }
                    sycl::group_barrier(grp);
                }
                float global_sum = (*slm)[0];
                sycl::group_barrier(grp);

                // Normalize
                float inv_sum = 1.0f / global_sum;
                for (int t = lid; t < n_tokens; t += SM_WG) {
                    w[t] *= inv_sum;
                }
            });
    });
}

// ============================================================================
// Kernel 3: TQ V-accumulate — tiled, one thread per dimension
// Same as Phase 0.5 but parameterized on TILE_SIZE
// ============================================================================

template <int TILE_SIZE>
void launch_fused_v_accum_tq(sycl::queue& q,
    const block_turbo4_0* d_v_blocks,  // [n_heads * n_tokens]
    const float* d_weights,            // [n_heads * n_tokens]
    const float* d_tc4,
    float* d_partial_acc,              // [n_heads * n_tiles * 128]
    int n_heads, int n_tokens, int n_tiles)
{
    int total_wgs = n_heads * n_tiles;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * 128, 128),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int t = item.get_local_id(0);

                int head = wg_id / n_tiles;
                int tile = wg_id % n_tiles;
                int tile_start = tile * TILE_SIZE;
                int tile_end = sycl::min(tile_start + TILE_SIZE, n_tokens);

                float acc = 0.0f;
                for (int tok = tile_start; tok < tile_end; tok++) {
                    const auto& blk = d_v_blocks[head * n_tokens + tok];

                    uint8_t idx = (t % 2 == 0)
                        ? (blk.qs[t/2] & 0xF)
                        : ((blk.qs[t/2] >> 4) & 0xF);
                    float centroid = d_tc4[idx];

                    // Decode norm inline
                    uint16_t nb = blk.norm;
                    uint32_t sb = (uint32_t)(nb & 0x8000) << 16;
                    uint32_t eb = (nb >> 10) & 0x1F;
                    uint32_t mb = nb & 0x3FF;
                    float nv;
                    uint32_t r = (eb == 0) ? sb : (sb | ((eb + 112) << 23) | (mb << 13));
                    __builtin_memcpy(&nv, &r, 4);

                    float w = d_weights[head * n_tokens + tok] * nv;
                    acc += w * centroid;
                }

                d_partial_acc[(head * n_tiles + tile) * 128 + t] = acc;
            });
    });
}

// ============================================================================
// Kernel 3b: FP16 V-accumulate baseline
// ============================================================================

template <int TILE_SIZE>
void launch_fused_v_accum_fp16(sycl::queue& q,
    const sycl::half* d_v_fp16,        // [n_heads * n_tokens * 128]
    const float* d_weights,            // [n_heads * n_tokens]
    float* d_partial_acc,              // [n_heads * n_tiles * 128]
    int n_heads, int n_tokens, int n_tiles)
{
    int total_wgs = n_heads * n_tiles;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * 128, 128),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int t = item.get_local_id(0);

                int head = wg_id / n_tiles;
                int tile = wg_id % n_tiles;
                int tile_start = tile * TILE_SIZE;
                int tile_end = sycl::min(tile_start + TILE_SIZE, n_tokens);

                float acc = 0.0f;
                for (int tok = tile_start; tok < tile_end; tok++) {
                    const sycl::half* v = d_v_fp16 + (head * n_tokens + tok) * 128;
                    float w = d_weights[head * n_tokens + tok];
                    acc += w * static_cast<float>(v[t]);
                }

                d_partial_acc[(head * n_tiles + tile) * 128 + t] = acc;
            });
    });
}

// ============================================================================
// Kernel 4: Reduce tile accumulators + inverse WHT → final output
// One work-group per head (128 threads)
// ============================================================================

void launch_reduce_wht(sycl::queue& q,
    const float* d_partial_acc,  // [n_heads * n_tiles * 128]
    float* d_output,             // [n_heads * 128]
    const float* d_ts1, const float* d_ts2,
    int n_heads, int n_tiles)
{
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(n_heads * 128, 128),
            [=](sycl::nd_item<1> item) {
                int head = item.get_group(0);
                int t = item.get_local_id(0);
                auto grp = item.get_group();

                float acc = 0.0f;
                for (int tile = 0; tile < n_tiles; tile++)
                    acc += d_partial_acc[(head * n_tiles + tile) * 128 + t];

                // Inverse RHT: TS2 → TINV → butterfly → TS1
                acc *= d_ts2[t];
                acc *= 0.08838834764831845f;

                auto local = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                (*local)[t] = acc;
                sycl::group_barrier(grp);

                for (int hv = 1; hv < 128; hv *= 2) {
                    if ((t % (2 * hv)) < hv) {
                        float a = (*local)[t], b = (*local)[t + hv];
                        (*local)[t] = a + b;
                        (*local)[t + hv] = a - b;
                    }
                    sycl::group_barrier(grp);
                }

                d_output[head * 128 + t] = (*local)[t] * d_ts1[t];
            });
    });
}

// ============================================================================
// Kernel 4b: Reduce only (no WHT) — for FP16 baseline
// ============================================================================

void launch_reduce_only(sycl::queue& q,
    const float* d_partial_acc,
    float* d_output,
    int n_heads, int n_tiles)
{
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(n_heads * 128, 128),
            [=](sycl::nd_item<1> item) {
                int head = item.get_group(0);
                int t = item.get_local_id(0);

                float acc = 0.0f;
                for (int tile = 0; tile < n_tiles; tile++)
                    acc += d_partial_acc[(head * n_tiles + tile) * 128 + t];

                d_output[head * 128 + t] = acc;
            });
    });
}

// ============================================================================
// CPU-side softmax (for correctness reference)
// ============================================================================

void softmax_cpu(const float* scores, float* weights, int n_heads, int n_tokens) {
    for (int h = 0; h < n_heads; h++) {
        const float* s = scores + h * n_tokens;
        float* w = weights + h * n_tokens;
        float mx = *std::max_element(s, s + n_tokens);
        float sum = 0.0f;
        for (int t = 0; t < n_tokens; t++) {
            w[t] = std::exp(s[t] - mx);
            sum += w[t];
        }
        for (int t = 0; t < n_tokens; t++)
            w[t] /= sum;
    }
}

// ============================================================================
// Correctness Tests
// ============================================================================

bool run_correctness_tests() {
    std::cout << "\n=== Phase 2 Correctness Tests ===" << std::endl;
    bool all_pass = true;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const int n_heads = 4;
    const int n_tokens = 64;

    // Generate random queries
    std::vector<float> queries(n_heads * 128);
    for (auto& v : queries) v = dist(rng);

    // Generate random K,V blocks via quantization
    std::vector<block_turbo4_0> k_blocks(n_heads * n_tokens);
    std::vector<block_turbo4_0> v_blocks(n_heads * n_tokens);

    for (int i = 0; i < n_heads * n_tokens; i++) {
        float vec[128];
        float scale = 0.5f + (rng() % 1000) / 1000.0f;
        for (int d = 0; d < 128; d++) vec[d] = dist(rng) * scale;
        tq_ref::quantize_vector_turbo4(vec, k_blocks[i]);
        for (int d = 0; d < 128; d++) vec[d] = dist(rng) * scale;
        tq_ref::quantize_vector_turbo4(vec, v_blocks[i]);
    }

    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());

    // Test 1: Fused K-scores vs CPU reference
    {
        auto* d_k = sycl::malloc_device<block_turbo4_0>(n_heads * n_tokens, q);
        auto* d_tc4 = sycl::malloc_device<float>(16, q);
        auto* d_scores = sycl::malloc_device<float>(n_heads * n_tokens, q);

        std::vector<float> q_rot(n_heads * 128);
        for (int h = 0; h < n_heads; h++) {
            std::memcpy(q_rot.data() + h * 128, queries.data() + h * 128, 128 * sizeof(float));
            tq_ref::forward_rht(q_rot.data() + h * 128);
        }
        auto* d_q_rot = sycl::malloc_device<float>(n_heads * 128, q);

        q.memcpy(d_k, k_blocks.data(), n_heads * n_tokens * sizeof(block_turbo4_0));
        q.memcpy(d_tc4, TC4, 16 * sizeof(float));
        q.memcpy(d_q_rot, q_rot.data(), n_heads * 128 * sizeof(float));
        q.wait();

        launch_fused_k_scores_tq<64>(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
        q.wait();

        std::vector<float> gpu_scores(n_heads * n_tokens);
        q.memcpy(gpu_scores.data(), d_scores, n_heads * n_tokens * sizeof(float)).wait();

        // CPU reference
        std::vector<float> cpu_scores(n_heads * n_tokens);
        for (int h = 0; h < n_heads; h++) {
            for (int t = 0; t < n_tokens; t++) {
                uint8_t indices[128];
                tq_ref::unpack_turbo4(k_blocks[h * n_tokens + t], indices);
                float dot = 0.0f;
                for (int d = 0; d < 128; d++)
                    dot += q_rot[h * 128 + d] * TC4[indices[d]];
                float nv = fp16_to_fp32(k_blocks[h * n_tokens + t].norm);
                cpu_scores[h * n_tokens + t] = dot * nv;
            }
        }

        float max_err = 0.0f;
        for (int i = 0; i < n_heads * n_tokens; i++)
            max_err = std::max(max_err, std::fabs(gpu_scores[i] - cpu_scores[i]));

        bool pass = max_err < 1e-4f;
        std::cout << "Test 1: Fused TQ K-scores vs CPU reference" << std::endl;
        std::cout << "  Max abs error: " << std::scientific << max_err << std::endl;
        std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;

        sycl::free(d_k, q); sycl::free(d_tc4, q);
        sycl::free(d_scores, q); sycl::free(d_q_rot, q);
    }

    // Test 2: GPU softmax vs CPU softmax
    {
        // Generate random scores
        std::vector<float> h_scores(n_heads * n_tokens);
        for (auto& v : h_scores) v = dist(rng) * 5.0f;

        auto* d_scores = sycl::malloc_device<float>(n_heads * n_tokens, q);
        auto* d_weights = sycl::malloc_device<float>(n_heads * n_tokens, q);
        q.memcpy(d_scores, h_scores.data(), n_heads * n_tokens * sizeof(float)).wait();

        launch_softmax_gpu(q, d_scores, d_weights, n_heads, n_tokens);
        q.wait();

        std::vector<float> gpu_weights(n_heads * n_tokens);
        q.memcpy(gpu_weights.data(), d_weights, n_heads * n_tokens * sizeof(float)).wait();

        std::vector<float> cpu_weights(n_heads * n_tokens);
        softmax_cpu(h_scores.data(), cpu_weights.data(), n_heads, n_tokens);

        float max_err = 0.0f;
        for (int i = 0; i < n_heads * n_tokens; i++)
            max_err = std::max(max_err, std::fabs(gpu_weights[i] - cpu_weights[i]));

        bool pass = max_err < 1e-5f;
        std::cout << "Test 2: GPU softmax vs CPU softmax" << std::endl;
        std::cout << "  Max abs error: " << std::scientific << max_err << std::endl;
        std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;

        sycl::free(d_scores, q); sycl::free(d_weights, q);
    }

    // Test 3: Full fused TQ attention vs CPU standard attention
    {
        auto* d_k = sycl::malloc_device<block_turbo4_0>(n_heads * n_tokens, q);
        auto* d_v = sycl::malloc_device<block_turbo4_0>(n_heads * n_tokens, q);
        auto* d_tc4 = sycl::malloc_device<float>(16, q);
        auto* d_ts1 = sycl::malloc_device<float>(128, q);
        auto* d_ts2 = sycl::malloc_device<float>(128, q);
        auto* d_scores = sycl::malloc_device<float>(n_heads * n_tokens, q);
        auto* d_weights = sycl::malloc_device<float>(n_heads * n_tokens, q);

        constexpr int TILE = 64;
        int n_tiles = (n_tokens + TILE - 1) / TILE;
        auto* d_partial = sycl::malloc_device<float>(n_heads * n_tiles * 128, q);
        auto* d_output = sycl::malloc_device<float>(n_heads * 128, q);

        std::vector<float> q_rot(n_heads * 128);
        for (int h = 0; h < n_heads; h++) {
            std::memcpy(q_rot.data() + h * 128, queries.data() + h * 128, 128 * sizeof(float));
            tq_ref::forward_rht(q_rot.data() + h * 128);
        }
        auto* d_q_rot = sycl::malloc_device<float>(n_heads * 128, q);

        q.memcpy(d_k, k_blocks.data(), n_heads * n_tokens * sizeof(block_turbo4_0));
        q.memcpy(d_v, v_blocks.data(), n_heads * n_tokens * sizeof(block_turbo4_0));
        q.memcpy(d_tc4, TC4, 16 * sizeof(float));
        q.memcpy(d_ts1, TS1, 128 * sizeof(float));
        q.memcpy(d_ts2, TS2, 128 * sizeof(float));
        q.memcpy(d_q_rot, q_rot.data(), n_heads * 128 * sizeof(float));
        q.wait();

        // Full pipeline: K-scores → softmax → V-accum → reduce+WHT
        launch_fused_k_scores_tq<64>(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
        launch_softmax_gpu(q, d_scores, d_weights, n_heads, n_tokens);
        launch_fused_v_accum_tq<TILE>(q, d_v, d_weights, d_tc4, d_partial,
                                       n_heads, n_tokens, n_tiles);
        launch_reduce_wht(q, d_partial, d_output, d_ts1, d_ts2, n_heads, n_tiles);
        q.wait();

        std::vector<float> gpu_output(n_heads * 128);
        q.memcpy(gpu_output.data(), d_output, n_heads * 128 * sizeof(float)).wait();

        // CPU standard attention reference
        std::vector<float> cpu_output(n_heads * 128);
        tq_attn::standard_attention_turbo4(
            queries.data(), k_blocks.data(), v_blocks.data(),
            cpu_output.data(), n_heads, n_tokens);

        float max_err = 0.0f, max_val = 0.0f;
        for (int i = 0; i < n_heads * 128; i++) {
            max_err = std::max(max_err, std::fabs(gpu_output[i] - cpu_output[i]));
            max_val = std::max(max_val, std::fabs(cpu_output[i]));
        }
        float rel_err = (max_val > 1e-10f) ? (max_err / max_val) : max_err;

        bool pass = rel_err < 1e-3f;
        std::cout << "Test 3: Full fused TQ attention vs CPU standard" << std::endl;
        std::cout << "  Max abs error: " << std::scientific << max_err << std::endl;
        std::cout << "  Max rel error: " << rel_err << std::endl;
        std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;

        sycl::free(d_k, q); sycl::free(d_v, q); sycl::free(d_tc4, q);
        sycl::free(d_ts1, q); sycl::free(d_ts2, q); sycl::free(d_scores, q);
        sycl::free(d_weights, q); sycl::free(d_partial, q); sycl::free(d_output, q);
        sycl::free(d_q_rot, q);
    }

    // Test 4: FP16 baseline attention vs CPU (dequant → FP16 → attention)
    {
        // Dequant K,V to FP32, then convert to FP16 for the baseline
        std::vector<float> k_fp32(n_heads * n_tokens * 128);
        std::vector<float> v_fp32(n_heads * n_tokens * 128);
        for (int i = 0; i < n_heads * n_tokens; i++) {
            tq_ref::dequantize_vector_turbo4(k_blocks[i], k_fp32.data() + i * 128);
            tq_ref::dequantize_vector_turbo4(v_blocks[i], v_fp32.data() + i * 128);
        }
        std::vector<sycl::half> k_fp16(n_heads * n_tokens * 128);
        std::vector<sycl::half> v_fp16(n_heads * n_tokens * 128);
        for (size_t i = 0; i < k_fp32.size(); i++) {
            k_fp16[i] = sycl::half(k_fp32[i]);
            v_fp16[i] = sycl::half(v_fp32[i]);
        }

        auto* d_k16 = sycl::malloc_device<sycl::half>(n_heads * n_tokens * 128, q);
        auto* d_v16 = sycl::malloc_device<sycl::half>(n_heads * n_tokens * 128, q);
        auto* d_scores = sycl::malloc_device<float>(n_heads * n_tokens, q);
        auto* d_weights = sycl::malloc_device<float>(n_heads * n_tokens, q);

        constexpr int TILE = 64;
        int n_tiles = (n_tokens + TILE - 1) / TILE;
        auto* d_partial = sycl::malloc_device<float>(n_heads * n_tiles * 128, q);
        auto* d_output = sycl::malloc_device<float>(n_heads * 128, q);

        // For FP16 baseline, q is NOT rotated (standard space)
        auto* d_q = sycl::malloc_device<float>(n_heads * 128, q);

        q.memcpy(d_k16, k_fp16.data(), k_fp16.size() * sizeof(sycl::half));
        q.memcpy(d_v16, v_fp16.data(), v_fp16.size() * sizeof(sycl::half));
        q.memcpy(d_q, queries.data(), n_heads * 128 * sizeof(float));
        q.wait();

        launch_fused_k_scores_fp16<64>(q, d_k16, d_q, d_scores, n_heads, n_tokens);
        launch_softmax_gpu(q, d_scores, d_weights, n_heads, n_tokens);
        launch_fused_v_accum_fp16<TILE>(q, d_v16, d_weights, d_partial,
                                         n_heads, n_tokens, n_tiles);
        launch_reduce_only(q, d_partial, d_output, n_heads, n_tiles);
        q.wait();

        std::vector<float> gpu_output(n_heads * 128);
        q.memcpy(gpu_output.data(), d_output, n_heads * 128 * sizeof(float)).wait();

        // CPU reference: standard attention on dequanted vectors
        // (matches the FP16 path since dequant→FP16→attention ≈ standard attention)
        std::vector<float> cpu_output(n_heads * 128);
        tq_attn::standard_attention_turbo4(
            queries.data(), k_blocks.data(), v_blocks.data(),
            cpu_output.data(), n_heads, n_tokens);

        float max_err = 0.0f, max_val = 0.0f;
        for (int i = 0; i < n_heads * 128; i++) {
            max_err = std::max(max_err, std::fabs(gpu_output[i] - cpu_output[i]));
            max_val = std::max(max_val, std::fabs(cpu_output[i]));
        }
        float rel_err = (max_val > 1e-10f) ? (max_err / max_val) : max_err;

        // FP16 truncation causes more error, be slightly more lenient
        bool pass = rel_err < 5e-3f;
        std::cout << "Test 4: FP16 baseline attention vs CPU standard" << std::endl;
        std::cout << "  Max abs error: " << std::scientific << max_err << std::endl;
        std::cout << "  Max rel error: " << rel_err << std::endl;
        std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;

        sycl::free(d_k16, q); sycl::free(d_v16, q); sycl::free(d_scores, q);
        sycl::free(d_weights, q); sycl::free(d_partial, q); sycl::free(d_output, q);
        sycl::free(d_q, q);
    }

    std::cout << "\n=== All " << (all_pass ? "PASSED" : "FAILED") << " ===" << std::endl;
    return all_pass;
}

// ============================================================================
// Benchmark runner
// ============================================================================

struct BenchResult {
    std::string label;
    int n_tokens;
    int tile_size;
    double time_us;
    double ms_per_token_14k;
};

void run_benchmarks(int max_tokens) {
    std::cout << "\n=== Phase 2 Benchmarks ===" << std::endl;

    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    auto dev = q.get_device();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Global mem: " << dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024) << " MB\n";

    const int n_heads = 40;
    const int warmup = 10;
    const int iters = 100;

    std::vector<int> token_counts = {1000, 4000, 14000, 60000};
    std::vector<int> tile_sizes = {16, 32, 64};
    std::vector<BenchResult> all_results;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int n_tokens : token_counts) {
        if (n_tokens > max_tokens) continue;

        int total_kv = n_tokens * n_heads;

        std::cout << "\n======== n_tokens=" << n_tokens << " ========\n";

        // Generate data
        std::vector<float> queries(n_heads * 128);
        for (auto& v : queries) v = dist(rng);

        std::vector<float> q_rot(n_heads * 128);
        for (int h = 0; h < n_heads; h++) {
            std::memcpy(q_rot.data() + h * 128, queries.data() + h * 128, 128 * sizeof(float));
            tq_ref::forward_rht(q_rot.data() + h * 128);
        }

        std::vector<block_turbo4_0> k_blocks(total_kv);
        std::vector<block_turbo4_0> v_blocks(total_kv);
        for (int i = 0; i < total_kv; i++) {
            float vec[128];
            float s = 0.5f + (rng() % 1000) / 1000.0f;
            for (int d = 0; d < 128; d++) vec[d] = dist(rng) * s;
            tq_ref::quantize_vector_turbo4(vec, k_blocks[i]);
            for (int d = 0; d < 128; d++) vec[d] = dist(rng) * s;
            tq_ref::quantize_vector_turbo4(vec, v_blocks[i]);
        }

        // Prepare FP16 KV (dequant → FP16)
        std::vector<sycl::half> k_fp16(total_kv * 128);
        std::vector<sycl::half> v_fp16(total_kv * 128);
        for (int i = 0; i < total_kv; i++) {
            float tmp[128];
            tq_ref::dequantize_vector_turbo4(k_blocks[i], tmp);
            for (int d = 0; d < 128; d++) k_fp16[i * 128 + d] = sycl::half(tmp[d]);
            tq_ref::dequantize_vector_turbo4(v_blocks[i], tmp);
            for (int d = 0; d < 128; d++) v_fp16[i * 128 + d] = sycl::half(tmp[d]);
        }

        // Device allocations
        auto* d_k = sycl::malloc_device<block_turbo4_0>(total_kv, q);
        auto* d_v = sycl::malloc_device<block_turbo4_0>(total_kv, q);
        auto* d_k16 = sycl::malloc_device<sycl::half>(total_kv * 128, q);
        auto* d_v16 = sycl::malloc_device<sycl::half>(total_kv * 128, q);
        auto* d_tc4 = sycl::malloc_device<float>(16, q);
        auto* d_ts1 = sycl::malloc_device<float>(128, q);
        auto* d_ts2 = sycl::malloc_device<float>(128, q);
        auto* d_q_rot = sycl::malloc_device<float>(n_heads * 128, q);
        auto* d_q_plain = sycl::malloc_device<float>(n_heads * 128, q);
        auto* d_scores = sycl::malloc_device<float>(total_kv, q);
        auto* d_weights = sycl::malloc_device<float>(total_kv, q);
        auto* d_output = sycl::malloc_device<float>(n_heads * 128, q);

        q.memcpy(d_k, k_blocks.data(), total_kv * sizeof(block_turbo4_0));
        q.memcpy(d_v, v_blocks.data(), total_kv * sizeof(block_turbo4_0));
        q.memcpy(d_k16, k_fp16.data(), total_kv * 128 * sizeof(sycl::half));
        q.memcpy(d_v16, v_fp16.data(), total_kv * 128 * sizeof(sycl::half));
        q.memcpy(d_tc4, TC4, 16 * sizeof(float));
        q.memcpy(d_ts1, TS1, 128 * sizeof(float));
        q.memcpy(d_ts2, TS2, 128 * sizeof(float));
        q.memcpy(d_q_rot, q_rot.data(), n_heads * 128 * sizeof(float));
        q.memcpy(d_q_plain, queries.data(), n_heads * 128 * sizeof(float));
        q.wait();

        // Sweep tile sizes for full pipeline
        for (int tile_size : tile_sizes) {
            int n_tiles = (n_tokens + tile_size - 1) / tile_size;
            auto* d_partial = sycl::malloc_device<float>(n_heads * n_tiles * 128, q);

            // --- TQ fused pipeline ---
            auto run_tq = [&]() {
                launch_fused_k_scores_tq<256>(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
                launch_softmax_gpu(q, d_scores, d_weights, n_heads, n_tokens);
                if (tile_size == 16)
                    launch_fused_v_accum_tq<16>(q, d_v, d_weights, d_tc4, d_partial, n_heads, n_tokens, n_tiles);
                else if (tile_size == 32)
                    launch_fused_v_accum_tq<32>(q, d_v, d_weights, d_tc4, d_partial, n_heads, n_tokens, n_tiles);
                else
                    launch_fused_v_accum_tq<64>(q, d_v, d_weights, d_tc4, d_partial, n_heads, n_tokens, n_tiles);
                launch_reduce_wht(q, d_partial, d_output, d_ts1, d_ts2, n_heads, n_tiles);
            };

            for (int i = 0; i < warmup; i++) { run_tq(); }
            q.wait();

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) { run_tq(); }
            q.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            double tq_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
            double tq_ms_14k = tq_us * 14000.0 / n_tokens / 1000.0;

            BenchResult r_tq;
            r_tq.label = "TQ_fused";
            r_tq.n_tokens = n_tokens;
            r_tq.tile_size = tile_size;
            r_tq.time_us = tq_us;
            r_tq.ms_per_token_14k = tq_ms_14k;
            all_results.push_back(r_tq);

            std::cout << "TQ_fused  tile=" << std::setw(2) << tile_size
                      << "  " << std::fixed << std::setprecision(1) << std::setw(8) << tq_us << " us"
                      << "  est_14K=" << std::setprecision(3) << tq_ms_14k << " ms" << std::endl;

            // --- FP16 baseline pipeline ---
            auto run_fp16 = [&]() {
                launch_fused_k_scores_fp16<256>(q, d_k16, d_q_plain, d_scores, n_heads, n_tokens);
                launch_softmax_gpu(q, d_scores, d_weights, n_heads, n_tokens);
                if (tile_size == 16)
                    launch_fused_v_accum_fp16<16>(q, d_v16, d_weights, d_partial, n_heads, n_tokens, n_tiles);
                else if (tile_size == 32)
                    launch_fused_v_accum_fp16<32>(q, d_v16, d_weights, d_partial, n_heads, n_tokens, n_tiles);
                else
                    launch_fused_v_accum_fp16<64>(q, d_v16, d_weights, d_partial, n_heads, n_tokens, n_tiles);
                launch_reduce_only(q, d_partial, d_output, n_heads, n_tiles);
            };

            for (int i = 0; i < warmup; i++) { run_fp16(); }
            q.wait();

            t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) { run_fp16(); }
            q.wait();
            t1 = std::chrono::high_resolution_clock::now();
            double fp16_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
            double fp16_ms_14k = fp16_us * 14000.0 / n_tokens / 1000.0;

            BenchResult r_fp16;
            r_fp16.label = "FP16_baseline";
            r_fp16.n_tokens = n_tokens;
            r_fp16.tile_size = tile_size;
            r_fp16.time_us = fp16_us;
            r_fp16.ms_per_token_14k = fp16_ms_14k;
            all_results.push_back(r_fp16);

            double overhead_us = tq_us - fp16_us;
            double overhead_ms_14k = tq_ms_14k - fp16_ms_14k;

            std::cout << "FP16_base tile=" << std::setw(2) << tile_size
                      << "  " << std::fixed << std::setprecision(1) << std::setw(8) << fp16_us << " us"
                      << "  est_14K=" << std::setprecision(3) << fp16_ms_14k << " ms" << std::endl;
            std::cout << "  TQ overhead:  " << std::setprecision(1) << overhead_us << " us"
                      << "  est_14K=" << std::setprecision(3) << overhead_ms_14k << " ms"
                      << "  ratio=" << std::setprecision(2) << (fp16_us > 0 ? tq_us / fp16_us : 0) << "x"
                      << std::endl;
            std::cout << std::endl;

            sycl::free(d_partial, q);
        }

        // --- K-score kernel only comparison (TQ vs FP16) ---
        std::cout << "--- K-score kernel isolation ---\n";
        {
            // TQ K-scores
            for (int i = 0; i < warmup; i++)
                launch_fused_k_scores_tq<256>(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
            q.wait();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++)
                launch_fused_k_scores_tq<256>(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
            q.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            double tq_k_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            // FP16 K-scores
            for (int i = 0; i < warmup; i++)
                launch_fused_k_scores_fp16<256>(q, d_k16, d_q_plain, d_scores, n_heads, n_tokens);
            q.wait();
            t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++)
                launch_fused_k_scores_fp16<256>(q, d_k16, d_q_plain, d_scores, n_heads, n_tokens);
            q.wait();
            t1 = std::chrono::high_resolution_clock::now();
            double fp16_k_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            std::cout << "  TQ K-score:   " << std::fixed << std::setprecision(1) << tq_k_us << " us"
                      << "  (" << std::setprecision(2) << tq_k_us * 1000.0 / total_kv << " ns/vec)" << std::endl;
            std::cout << "  FP16 K-score: " << std::setprecision(1) << fp16_k_us << " us"
                      << "  (" << std::setprecision(2) << fp16_k_us * 1000.0 / total_kv << " ns/vec)" << std::endl;
            std::cout << "  TQ overhead:  " << std::setprecision(1) << tq_k_us - fp16_k_us << " us"
                      << "  ratio=" << std::setprecision(2) << (fp16_k_us > 0 ? tq_k_us / fp16_k_us : 0) << "x\n\n";
        }

        // --- Softmax kernel only ---
        std::cout << "--- Softmax kernel isolation ---\n";
        {
            // Pre-fill scores
            launch_fused_k_scores_tq<256>(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
            q.wait();

            for (int i = 0; i < warmup; i++)
                launch_softmax_gpu(q, d_scores, d_weights, n_heads, n_tokens);
            q.wait();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++)
                launch_softmax_gpu(q, d_scores, d_weights, n_heads, n_tokens);
            q.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            double sm_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
            std::cout << "  Softmax:      " << std::fixed << std::setprecision(1) << sm_us << " us\n\n";
        }

        sycl::free(d_k, q); sycl::free(d_v, q);
        sycl::free(d_k16, q); sycl::free(d_v16, q);
        sycl::free(d_tc4, q); sycl::free(d_ts1, q); sycl::free(d_ts2, q);
        sycl::free(d_q_rot, q); sycl::free(d_q_plain, q);
        sycl::free(d_scores, q); sycl::free(d_weights, q);
        sycl::free(d_output, q);
    }

    // ============================================
    // Summary
    // ============================================
    std::cout << "\n============================================" << std::endl;
    std::cout << "PHASE 2 SUMMARY: TQ vs FP16 by tile_size" << std::endl;
    std::cout << "============================================" << std::endl;

    std::cout << std::setw(8) << "tokens"
              << std::setw(6) << "tile"
              << std::setw(12) << "TQ(us)"
              << std::setw(12) << "FP16(us)"
              << std::setw(12) << "OH(us)"
              << std::setw(10) << "OH_14K(ms)"
              << std::setw(8) << "ratio" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    for (int n_tokens : token_counts) {
        for (int tile : tile_sizes) {
            const BenchResult* tq = nullptr;
            const BenchResult* fp = nullptr;
            for (const auto& r : all_results) {
                if (r.n_tokens == n_tokens && r.tile_size == tile) {
                    if (r.label == "TQ_fused") tq = &r;
                    if (r.label == "FP16_baseline") fp = &r;
                }
            }
            if (tq && fp) {
                double oh = tq->time_us - fp->time_us;
                double oh_14k = tq->ms_per_token_14k - fp->ms_per_token_14k;
                std::cout << std::setw(8) << n_tokens
                          << std::setw(6) << tile
                          << std::fixed
                          << std::setw(12) << std::setprecision(1) << tq->time_us
                          << std::setw(12) << std::setprecision(1) << fp->time_us
                          << std::setw(12) << std::setprecision(1) << oh
                          << std::setw(10) << std::setprecision(3) << oh_14k
                          << std::setw(8) << std::setprecision(2) << (fp->time_us > 0 ? tq->time_us / fp->time_us : 0) << "x"
                          << std::endl;
            }
        }
    }

    // Best tile per context
    std::cout << "\n--- Best tile_size per context ---\n";
    for (int n_tokens : token_counts) {
        if (n_tokens > max_tokens) continue;
        double best_us = 1e30;
        int best_tile = 0;
        for (const auto& r : all_results) {
            if (r.n_tokens == n_tokens && r.label == "TQ_fused" && r.time_us < best_us) {
                best_us = r.time_us;
                best_tile = r.tile_size;
            }
        }
        if (best_tile > 0) {
            std::cout << "  n_tokens=" << n_tokens << ": best tile=" << best_tile
                      << " (" << std::fixed << std::setprecision(1) << best_us << " us"
                      << ", est_14K=" << std::setprecision(3) << best_us * 14000.0 / n_tokens / 1000.0 << " ms)"
                      << std::endl;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "============================================" << std::endl;
    std::cout << "TurboQuant Phase 2: Fused Rotated-Space Attention" << std::endl;
    std::cout << "============================================" << std::endl;

    int max_tokens = 60000;
    bool skip_bench = false;
    bool skip_correctness = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (arg == "--skip-bench") skip_bench = true;
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
        run_benchmarks(max_tokens);
    }

    return 0;
}
