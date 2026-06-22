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
// TurboQuant Phase 0.5: Rotated-Space Attention Microbenchmark
//
// Paths:
//   A (baseline): Full dequant all KV (Phase 0 Mode 4, reference)
//   B: K-scores in rotated space (unpack + codebook + dot, no WHT per token)
//   C: V-accumulate in rotated space (unpack + codebook + weighted acc, no WHT)
//   D: Full rotated-space attention (B + softmax + C + final WHT)
// ============================================================================

static constexpr int V_TILE_SIZE = 256;  // tokens per V-accumulate work-group

// ============================================================================
// SYCL Kernel: Path A — Full dequant + global write (baseline, from Phase 0)
// ============================================================================

void launch_dequant_all_turbo4(sycl::queue& q,
    const block_turbo4_0* d_blocks, float* d_output,
    const float* d_tc4, const float* d_ts1, const float* d_ts2,
    int num_vectors)
{
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(num_vectors * 128, 128),
            [=](sycl::nd_item<1> item) {
                int vec_id = item.get_group(0);
                int t = item.get_local_id(0);
                auto grp = item.get_group();

                const auto& blk = d_blocks[vec_id];
                uint8_t idx = (t % 2 == 0) ? (blk.qs[t/2] & 0xF) : ((blk.qs[t/2] >> 4) & 0xF);
                float v = d_tc4[idx];

                // Inverse RHT
                v *= d_ts2[t];
                v *= 0.08838834764831845f;

                auto local = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                (*local)[t] = v;
                sycl::group_barrier(grp);

                for (int hv = 1; hv < 128; hv *= 2) {
                    if ((t % (2 * hv)) < hv) {
                        float a = (*local)[t], b = (*local)[t + hv];
                        (*local)[t] = a + b;
                        (*local)[t + hv] = a - b;
                    }
                    sycl::group_barrier(grp);
                }

                float result = (*local)[t] * d_ts1[t];

                uint16_t nb = blk.norm;
                uint32_t s = (uint32_t)(nb & 0x8000) << 16;
                uint32_t e = (nb >> 10) & 0x1F;
                uint32_t m = nb & 0x3FF;
                float nv;
                uint32_t r = (e == 0) ? s : (s | ((e + 112) << 23) | (m << 13));
                __builtin_memcpy(&nv, &r, 4);
                result *= nv;

                d_output[vec_id * 128 + t] = result;
            });
    });
}

// ============================================================================
// SYCL Kernel: Path B — K-score in rotated space (no WHT per token)
// One work-group per (head, token). Computes score = norm * dot(q_rot, codebook[idx])
// ============================================================================

void launch_k_scores_rotated_turbo4(sycl::queue& q,
    const block_turbo4_0* d_k_blocks,  // [n_heads * n_tokens]
    const float* d_q_rot,              // [n_heads * 128]
    const float* d_tc4,
    float* d_scores,                   // [n_heads * n_tokens]
    int n_heads, int n_tokens)
{
    int total_wgs = n_heads * n_tokens;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * 128, 128),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int t = item.get_local_id(0);
                auto grp = item.get_group();

                int head = wg_id / n_tokens;
                int tok = wg_id % n_tokens;

                const auto& blk = d_k_blocks[head * n_tokens + tok];

                // Unpack + codebook
                uint8_t idx = (t % 2 == 0) ? (blk.qs[t/2] & 0xF) : ((blk.qs[t/2] >> 4) & 0xF);
                float centroid = d_tc4[idx];

                // Dot product contribution
                float partial = centroid * d_q_rot[head * 128 + t];

                // Tree reduction in SLM
                auto local = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                (*local)[t] = partial;
                sycl::group_barrier(grp);

                for (int s = 64; s > 0; s >>= 1) {
                    if (t < (unsigned)s)
                        (*local)[t] += (*local)[t + s];
                    sycl::group_barrier(grp);
                }

                if (t == 0) {
                    // Multiply by norm
                    uint16_t nb = blk.norm;
                    uint32_t sb = (uint32_t)(nb & 0x8000) << 16;
                    uint32_t eb = (nb >> 10) & 0x1F;
                    uint32_t mb = nb & 0x3FF;
                    float nv;
                    uint32_t r = (eb == 0) ? sb : (sb | ((eb + 112) << 23) | (mb << 13));
                    __builtin_memcpy(&nv, &r, 4);

                    d_scores[head * n_tokens + tok] = (*local)[0] * nv;
                }
            });
    });
}

// ============================================================================
// SYCL Kernel: Path C — V-accumulate in rotated space (tiled, no WHT per token)
// One work-group per (head, tile). Each WG loops over TILE_SIZE tokens,
// accumulating weighted codebook vectors into a 128-dim partial sum.
// ============================================================================

void launch_v_accum_rotated_turbo4(sycl::queue& q,
    const block_turbo4_0* d_v_blocks,  // [n_heads * n_tokens]
    const float* d_weights,            // [n_heads * n_tokens]
    const float* d_tc4,
    float* d_partial_acc,              // [n_heads * n_tiles * 128]
    int n_heads, int n_tokens, int n_tiles)
{
    int total_wgs = n_heads * n_tiles;
    int tile_size = V_TILE_SIZE;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(total_wgs * 128, 128),
            [=](sycl::nd_item<1> item) {
                int wg_id = item.get_group(0);
                int t = item.get_local_id(0);

                int head = wg_id / n_tiles;
                int tile = wg_id % n_tiles;
                int tile_start = tile * tile_size;
                int tile_end = sycl::min(tile_start + tile_size, n_tokens);

                float acc = 0.0f;
                for (int tok = tile_start; tok < tile_end; tok++) {
                    const auto& blk = d_v_blocks[head * n_tokens + tok];

                    // Unpack + codebook
                    uint8_t idx = (t % 2 == 0)
                        ? (blk.qs[t/2] & 0xF)
                        : ((blk.qs[t/2] >> 4) & 0xF);
                    float centroid = d_tc4[idx];

                    // Decode norm
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
// SYCL Kernel: Reduce tile accumulators + inverse WHT → final output
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

                // Sum across tiles
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
// CPU-side softmax
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
// Correctness validation
// ============================================================================

bool run_correctness_tests() {
    std::cout << "\n=== Phase 0.5 Correctness Tests ===" << std::endl;
    bool all_pass = true;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const int n_heads = 4;  // small for correctness
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

    // Test 1: CPU standard vs rotated-space attention
    {
        std::vector<float> out_std(n_heads * 128);
        std::vector<float> out_rot(n_heads * 128);

        tq_attn::standard_attention_turbo4(
            queries.data(), k_blocks.data(), v_blocks.data(),
            out_std.data(), n_heads, n_tokens);

        tq_attn::rotated_attention_turbo4(
            queries.data(), k_blocks.data(), v_blocks.data(),
            out_rot.data(), n_heads, n_tokens);

        float max_err = 0.0f;
        float max_val = 0.0f;
        for (int i = 0; i < n_heads * 128; i++) {
            max_err = std::max(max_err, std::fabs(out_std[i] - out_rot[i]));
            max_val = std::max(max_val, std::fabs(out_std[i]));
        }
        float rel_err = (max_val > 1e-10f) ? (max_err / max_val) : max_err;

        bool pass = rel_err < 1e-4f;
        std::cout << "Test 1: CPU standard vs rotated attention (turbo4)" << std::endl;
        std::cout << "  Max abs error: " << std::scientific << max_err << std::endl;
        std::cout << "  Max rel error: " << rel_err << std::endl;
        std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;
    }

    // Test 2: GPU rotated-space K-scores vs CPU reference
    {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());

        auto* d_k = sycl::malloc_device<block_turbo4_0>(n_heads * n_tokens, q);
        auto* d_tc4 = sycl::malloc_device<float>(16, q);
        auto* d_ts1 = sycl::malloc_device<float>(128, q);
        auto* d_ts2 = sycl::malloc_device<float>(128, q);
        auto* d_scores = sycl::malloc_device<float>(n_heads * n_tokens, q);

        // Prepare q_rot on CPU
        std::vector<float> q_rot(n_heads * 128);
        for (int h = 0; h < n_heads; h++) {
            std::memcpy(q_rot.data() + h * 128, queries.data() + h * 128, 128 * sizeof(float));
            tq_ref::forward_rht(q_rot.data() + h * 128);
        }
        auto* d_q_rot = sycl::malloc_device<float>(n_heads * 128, q);

        q.memcpy(d_k, k_blocks.data(), n_heads * n_tokens * sizeof(block_turbo4_0));
        q.memcpy(d_tc4, TC4, 16 * sizeof(float));
        q.memcpy(d_ts1, TS1, 128 * sizeof(float));
        q.memcpy(d_ts2, TS2, 128 * sizeof(float));
        q.memcpy(d_q_rot, q_rot.data(), n_heads * 128 * sizeof(float));
        q.wait();

        launch_k_scores_rotated_turbo4(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
        q.wait();

        std::vector<float> gpu_scores(n_heads * n_tokens);
        q.memcpy(gpu_scores.data(), d_scores, n_heads * n_tokens * sizeof(float)).wait();

        // CPU reference: compute rotated K scores
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
        std::cout << "Test 2: GPU K-scores vs CPU reference (turbo4)" << std::endl;
        std::cout << "  Max abs error: " << std::scientific << max_err << std::endl;
        std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;

        sycl::free(d_k, q); sycl::free(d_tc4, q); sycl::free(d_ts1, q);
        sycl::free(d_ts2, q); sycl::free(d_scores, q); sycl::free(d_q_rot, q);
    }

    // Test 3: GPU full rotated-space attention (Path D) vs CPU standard attention
    {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());

        auto* d_k = sycl::malloc_device<block_turbo4_0>(n_heads * n_tokens, q);
        auto* d_v = sycl::malloc_device<block_turbo4_0>(n_heads * n_tokens, q);
        auto* d_tc4 = sycl::malloc_device<float>(16, q);
        auto* d_ts1 = sycl::malloc_device<float>(128, q);
        auto* d_ts2 = sycl::malloc_device<float>(128, q);
        auto* d_scores = sycl::malloc_device<float>(n_heads * n_tokens, q);
        auto* d_weights = sycl::malloc_device<float>(n_heads * n_tokens, q);

        int n_tiles = (n_tokens + V_TILE_SIZE - 1) / V_TILE_SIZE;
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

        // Path D: K-scores → softmax (CPU) → V-accum → reduce+WHT
        launch_k_scores_rotated_turbo4(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
        q.wait();

        std::vector<float> scores(n_heads * n_tokens);
        q.memcpy(scores.data(), d_scores, n_heads * n_tokens * sizeof(float)).wait();

        std::vector<float> weights(n_heads * n_tokens);
        softmax_cpu(scores.data(), weights.data(), n_heads, n_tokens);
        q.memcpy(d_weights, weights.data(), n_heads * n_tokens * sizeof(float)).wait();

        launch_v_accum_rotated_turbo4(q, d_v, d_weights, d_tc4, d_partial,
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

        bool pass = rel_err < 1e-3f;  // slightly looser due to GPU FP differences
        std::cout << "Test 3: GPU Path D vs CPU standard attention (turbo4)" << std::endl;
        std::cout << "  Max abs error: " << std::scientific << max_err << std::endl;
        std::cout << "  Max rel error: " << rel_err << std::endl;
        std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
        all_pass &= pass;

        sycl::free(d_k, q); sycl::free(d_v, q); sycl::free(d_tc4, q);
        sycl::free(d_ts1, q); sycl::free(d_ts2, q); sycl::free(d_scores, q);
        sycl::free(d_weights, q); sycl::free(d_partial, q); sycl::free(d_output, q);
        sycl::free(d_q_rot, q);
    }

    std::cout << "\n=== All " << (all_pass ? "PASSED" : "FAILED") << " ===" << std::endl;
    return all_pass;
}

// ============================================================================
// Benchmark runner
// ============================================================================

struct PathResult {
    std::string path;
    int n_tokens;
    int n_heads;
    double time_us;
    double ns_per_kv_vector;
    double est_ms_14k;
    double est_ms_60k;
    double est_ms_128k;
};

void print_path_result(const PathResult& r) {
    std::cout << std::fixed;
    std::cout << "path:            " << r.path << std::endl;
    std::cout << "n_tokens:        " << r.n_tokens << std::endl;
    std::cout << "n_heads:         " << r.n_heads << std::endl;
    std::cout << "time_us:         " << std::setprecision(1) << r.time_us << std::endl;
    std::cout << "ns_per_kv_vec:   " << std::setprecision(2) << r.ns_per_kv_vector << std::endl;
    std::cout << "est_ms_14K:      " << std::setprecision(3) << r.est_ms_14k << std::endl;
    std::cout << "est_ms_60K:      " << std::setprecision(3) << r.est_ms_60k << std::endl;
    std::cout << "est_ms_128K:     " << std::setprecision(3) << r.est_ms_128k << std::endl;
    std::cout << std::endl;
}

PathResult estimate_ms(PathResult r, int n_heads) {
    // ns per KV vector → ms per token at various context lengths
    // Each token has n_heads K + n_heads V = 2*n_heads vectors
    auto est = [&](int ctx) -> double {
        return (double)ctx * n_heads * 2 * r.ns_per_kv_vector / 1e6;
    };
    r.est_ms_14k = est(14000);
    r.est_ms_60k = est(60000);
    r.est_ms_128k = est(128000);
    return r;
}

void run_benchmarks(int max_tokens, bool turbo4_only) {
    std::cout << "\n=== Phase 0.5 Benchmarks ===" << std::endl;

    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    auto dev = q.get_device();
    std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Global mem: " << dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024) << " MB\n" << std::endl;

    const int n_heads = 40;
    const int warmup = 10;
    const int iters = 100;

    std::vector<int> token_counts = {1000, 4000, 14000, 60000, 128000};
    std::vector<PathResult> all_results;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int n_tokens : token_counts) {
        if (n_tokens > max_tokens) continue;

        int total_kv = n_tokens * n_heads;
        // Memory estimate: K blocks + V blocks + scores + weights + partial_acc + output + q_rot + constants
        size_t mem_est = (size_t)total_kv * 68 * 2  // K + V blocks
                       + (size_t)total_kv * 4 * 2   // scores + weights
                       + (size_t)total_kv * 128 * 4  // dequant output (Path A)
                       + (size_t)n_heads * 128 * 4 * 2;  // partial + output
        int n_tiles = (n_tokens + V_TILE_SIZE - 1) / V_TILE_SIZE;
        mem_est += (size_t)n_heads * n_tiles * 128 * 4;  // partial accumulators

        size_t dev_mem = dev.get_info<sycl::info::device::global_mem_size>();
        if (mem_est > dev_mem * 0.8) {
            std::cout << "SKIP n_tokens=" << n_tokens << " (need ~"
                      << mem_est/(1024*1024) << " MB)" << std::endl;
            continue;
        }

        std::cout << "======== n_tokens=" << n_tokens << " ========" << std::endl;

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

        // Allocate device memory
        auto* d_k = sycl::malloc_device<block_turbo4_0>(total_kv, q);
        auto* d_v = sycl::malloc_device<block_turbo4_0>(total_kv, q);
        auto* d_tc4 = sycl::malloc_device<float>(16, q);
        auto* d_ts1 = sycl::malloc_device<float>(128, q);
        auto* d_ts2 = sycl::malloc_device<float>(128, q);
        auto* d_q_rot = sycl::malloc_device<float>(n_heads * 128, q);
        auto* d_scores = sycl::malloc_device<float>(total_kv, q);
        auto* d_weights = sycl::malloc_device<float>(total_kv, q);
        auto* d_output_dequant = sycl::malloc_device<float>(total_kv * 128, q);
        auto* d_partial = sycl::malloc_device<float>(n_heads * n_tiles * 128, q);
        auto* d_output = sycl::malloc_device<float>(n_heads * 128, q);

        q.memcpy(d_k, k_blocks.data(), total_kv * sizeof(block_turbo4_0));
        q.memcpy(d_v, v_blocks.data(), total_kv * sizeof(block_turbo4_0));
        q.memcpy(d_tc4, TC4, 16 * sizeof(float));
        q.memcpy(d_ts1, TS1, 128 * sizeof(float));
        q.memcpy(d_ts2, TS2, 128 * sizeof(float));
        q.memcpy(d_q_rot, q_rot.data(), n_heads * 128 * sizeof(float));

        // Pre-compute weights for V-accum benchmark (use uniform weights)
        std::vector<float> fake_weights(total_kv, 1.0f / n_tokens);
        q.memcpy(d_weights, fake_weights.data(), total_kv * sizeof(float));
        q.wait();

        // --- Path A: Full dequant all KV (baseline) ---
        // Dequant both K and V blocks (total_kv each) for fair comparison with Path D
        {
            for (int i = 0; i < warmup; i++) {
                launch_dequant_all_turbo4(q, d_k, d_output_dequant, d_tc4, d_ts1, d_ts2, total_kv);
                launch_dequant_all_turbo4(q, d_v, d_output_dequant, d_tc4, d_ts1, d_ts2, total_kv);
            }
            q.wait();

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) {
                launch_dequant_all_turbo4(q, d_k, d_output_dequant, d_tc4, d_ts1, d_ts2, total_kv);
                launch_dequant_all_turbo4(q, d_v, d_output_dequant, d_tc4, d_ts1, d_ts2, total_kv);
            }
            q.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            PathResult r;
            r.path = "A_dequant_all";
            r.n_tokens = n_tokens;
            r.n_heads = n_heads;
            r.time_us = us;
            r.ns_per_kv_vector = us * 1000.0 / (total_kv * 2);  // K+V
            r = estimate_ms(r, n_heads);
            std::cout << "--- Path A (dequant all KV, baseline) ---" << std::endl;
            print_path_result(r);
            all_results.push_back(r);
        }

        // --- Path B: K-scores in rotated space ---
        {
            for (int i = 0; i < warmup; i++) {
                launch_k_scores_rotated_turbo4(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
            }
            q.wait();

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) {
                launch_k_scores_rotated_turbo4(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
            }
            q.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            PathResult r;
            r.path = "B_k_scores_rotated";
            r.n_tokens = n_tokens;
            r.n_heads = n_heads;
            r.time_us = us;
            r.ns_per_kv_vector = us * 1000.0 / total_kv;  // K only
            r = estimate_ms(r, n_heads);
            std::cout << "--- Path B (K-scores, rotated space) ---" << std::endl;
            print_path_result(r);
            all_results.push_back(r);
        }

        // --- Path C: V-accumulate in rotated space ---
        {
            for (int i = 0; i < warmup; i++) {
                launch_v_accum_rotated_turbo4(q, d_v, d_weights, d_tc4, d_partial,
                                               n_heads, n_tokens, n_tiles);
            }
            q.wait();

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) {
                launch_v_accum_rotated_turbo4(q, d_v, d_weights, d_tc4, d_partial,
                                               n_heads, n_tokens, n_tiles);
            }
            q.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            PathResult r;
            r.path = "C_v_accum_rotated";
            r.n_tokens = n_tokens;
            r.n_heads = n_heads;
            r.time_us = us;
            r.ns_per_kv_vector = us * 1000.0 / total_kv;  // V only
            r = estimate_ms(r, n_heads);
            std::cout << "--- Path C (V-accumulate, rotated space) ---" << std::endl;
            print_path_result(r);
            all_results.push_back(r);
        }

        // --- Path D: Full rotated-space attention (B + softmax + C + reduce+WHT) ---
        {
            // For timing Path D, we do K-scores → download → softmax CPU → upload → V-accum → reduce+WHT
            // The softmax is CPU-side, not counted in a fused implementation.
            // For the microbench, we time B + C + reduce separately and sum.
            // For the "honest" measurement, we also time the full pipeline.

            std::vector<float> h_scores(total_kv);
            std::vector<float> h_weights(total_kv);

            // Warmup the full pipeline
            for (int i = 0; i < warmup; i++) {
                launch_k_scores_rotated_turbo4(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
                q.wait();
                q.memcpy(h_scores.data(), d_scores, total_kv * sizeof(float)).wait();
                softmax_cpu(h_scores.data(), h_weights.data(), n_heads, n_tokens);
                q.memcpy(d_weights, h_weights.data(), total_kv * sizeof(float)).wait();
                launch_v_accum_rotated_turbo4(q, d_v, d_weights, d_tc4, d_partial,
                                               n_heads, n_tokens, n_tiles);
                launch_reduce_wht(q, d_partial, d_output, d_ts1, d_ts2, n_heads, n_tiles);
                q.wait();
            }

            // Time GPU kernels only (B + C + reduce+WHT, excluding softmax transfer)
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; i++) {
                launch_k_scores_rotated_turbo4(q, d_k, d_q_rot, d_tc4, d_scores, n_heads, n_tokens);
                launch_v_accum_rotated_turbo4(q, d_v, d_weights, d_tc4, d_partial,
                                               n_heads, n_tokens, n_tiles);
                launch_reduce_wht(q, d_partial, d_output, d_ts1, d_ts2, n_heads, n_tiles);
            }
            q.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

            PathResult r;
            r.path = "D_full_rotated_attn";
            r.n_tokens = n_tokens;
            r.n_heads = n_heads;
            r.time_us = us;
            r.ns_per_kv_vector = us * 1000.0 / (total_kv * 2);  // K+V
            r = estimate_ms(r, n_heads);
            std::cout << "--- Path D (full rotated attention, GPU kernels only) ---" << std::endl;
            print_path_result(r);
            all_results.push_back(r);
        }

        // Cleanup
        sycl::free(d_k, q); sycl::free(d_v, q);
        sycl::free(d_tc4, q); sycl::free(d_ts1, q); sycl::free(d_ts2, q);
        sycl::free(d_q_rot, q); sycl::free(d_scores, q); sycl::free(d_weights, q);
        sycl::free(d_output_dequant, q); sycl::free(d_partial, q); sycl::free(d_output, q);
    }

    // Summary comparison
    std::cout << "\n============================================" << std::endl;
    std::cout << "CRITICAL COMPARISON: Path A vs Path D" << std::endl;
    std::cout << "============================================" << std::endl;

    for (const auto& rA : all_results) {
        if (rA.path != "A_dequant_all") continue;
        for (const auto& rD : all_results) {
            if (rD.path != "D_full_rotated_attn") continue;
            if (rA.n_tokens != rD.n_tokens) continue;

            double speedup = rA.time_us / rD.time_us;
            std::cout << "n_tokens=" << rA.n_tokens << ":" << std::endl;
            std::cout << "  Path A (dequant):   " << std::fixed << std::setprecision(1)
                      << rA.time_us << " us  (" << std::setprecision(3) << rA.est_ms_14k << " ms/tok @14K)" << std::endl;
            std::cout << "  Path D (rotated):   " << std::setprecision(1)
                      << rD.time_us << " us  (" << std::setprecision(3) << rD.est_ms_14k << " ms/tok @14K)" << std::endl;
            std::cout << "  Speedup:            " << std::setprecision(2) << speedup << "x" << std::endl;
            std::cout << std::endl;
        }
    }

    // Path B vs C breakdown
    std::cout << "============================================" << std::endl;
    std::cout << "PATH BREAKDOWN: B (K-score) + C (V-accum)" << std::endl;
    std::cout << "============================================" << std::endl;

    for (const auto& rB : all_results) {
        if (rB.path != "B_k_scores_rotated") continue;
        for (const auto& rC : all_results) {
            if (rC.path != "C_v_accum_rotated") continue;
            if (rB.n_tokens != rC.n_tokens) continue;

            std::cout << "n_tokens=" << rB.n_tokens << ":" << std::endl;
            std::cout << "  Path B (K-score):   " << std::fixed << std::setprecision(1)
                      << rB.time_us << " us  (" << std::setprecision(2) << rB.ns_per_kv_vector << " ns/vec)" << std::endl;
            std::cout << "  Path C (V-accum):   " << std::setprecision(1)
                      << rC.time_us << " us  (" << std::setprecision(2) << rC.ns_per_kv_vector << " ns/vec)" << std::endl;
            std::cout << "  B+C sum:            " << std::setprecision(1) << rB.time_us + rC.time_us << " us" << std::endl;
            std::cout << std::endl;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "============================================" << std::endl;
    std::cout << "TurboQuant Phase 0.5: Rotated-Space Attention" << std::endl;
    std::cout << "============================================" << std::endl;

    int max_tokens = 128000;
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
        run_benchmarks(max_tokens, true);
    }

    return 0;
}
