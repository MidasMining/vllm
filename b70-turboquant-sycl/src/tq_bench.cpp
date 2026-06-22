#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#include <sycl/sycl.hpp>
#include "../reference/tq_cpu_reference.h"

// ============================================================================
// TurboQuant SYCL Microbenchmark Harness
// 6 modes measuring different stages of the dequant pipeline
// ============================================================================

// Benchmark modes
enum class Mode {
    UNPACK_ONLY       = 1,  // Extract indices from packed storage
    UNPACK_CODEBOOK   = 2,  // Indices → centroid lookup
    UNPACK_CB_WHT     = 3,  // Full dequant pipeline, no output write
    FULL_DEQUANT_WRITE = 4, // Dequant → write FP16 to global memory (SPLIT cost)
    FULL_DEQUANT_CSUM  = 5, // Dequant → reduce to scalar checksum (ALU-only)
    PACK_QUANTIZE     = 6,  // FP16 → WHT → codebook → pack (write path)
};

static const char* mode_name(Mode m) {
    switch (m) {
        case Mode::UNPACK_ONLY:        return "1_unpack_only";
        case Mode::UNPACK_CODEBOOK:    return "2_unpack_codebook";
        case Mode::UNPACK_CB_WHT:      return "3_unpack_cb_wht";
        case Mode::FULL_DEQUANT_WRITE: return "4_full_dequant_write";
        case Mode::FULL_DEQUANT_CSUM:  return "5_full_dequant_checksum";
        case Mode::PACK_QUANTIZE:      return "6_pack_quantize";
        default: return "unknown";
    }
}

// ============================================================================
// SYCL Kernels
// ============================================================================

// --- Turbo4 dequant kernel (modes 1-5) ---
void launch_dequant_turbo4(sycl::queue& q, const block_turbo4_0* d_blocks,
                           float* d_output, float* d_checksum,
                           const float* d_tc4, const float* d_ts1, const float* d_ts2,
                           int num_vectors, Mode mode) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(num_vectors * 128, 128),
            [=](sycl::nd_item<1> item) {
                int vec_id = item.get_group(0);
                int t = item.get_local_id(0);
                auto grp = item.get_group();

                const auto& blk = d_blocks[vec_id];

                // Step 1: Unpack (4-bit nibble)
                uint8_t idx;
                if (t % 2 == 0) {
                    idx = blk.qs[t / 2] & 0xF;
                } else {
                    idx = (blk.qs[t / 2] >> 4) & 0xF;
                }

                if (mode == Mode::UNPACK_ONLY) {
                    // Write index to output (prevents optimizer from eliminating work)
                    d_output[vec_id * 128 + t] = (float)idx;
                    return;
                }

                // Step 2: Codebook lookup
                float v = d_tc4[idx];

                if (mode == Mode::UNPACK_CODEBOOK) {
                    d_output[vec_id * 128 + t] = v;
                    return;
                }

                // Step 3: Inverse RHT
                v *= d_ts2[t];
                constexpr float tinv = 0.08838834764831845f;
                v *= tinv;

                auto local = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                (*local)[t] = v;
                sycl::group_barrier(grp);

                for (int h_val = 1; h_val < 128; h_val *= 2) {
                    if ((t % (2 * h_val)) < h_val) {
                        float a = (*local)[t];
                        float b = (*local)[t + h_val];
                        (*local)[t]         = a + b;
                        (*local)[t + h_val] = a - b;
                    }
                    sycl::group_barrier(grp);
                }

                float result = (*local)[t] * d_ts1[t];

                // Denormalize
                uint16_t norm_bits = blk.norm;
                uint32_t sign_b = (uint32_t)(norm_bits & 0x8000) << 16;
                uint32_t exp_b  = (norm_bits >> 10) & 0x1F;
                uint32_t mant_b = norm_bits & 0x3FF;
                float norm_val;
                if (exp_b == 0) {
                    uint32_t r = sign_b;
                    __builtin_memcpy(&norm_val, &r, 4);
                } else {
                    uint32_t r = sign_b | ((exp_b + 112) << 23) | (mant_b << 13);
                    __builtin_memcpy(&norm_val, &r, 4);
                }
                result *= norm_val;

                if (mode == Mode::UNPACK_CB_WHT) {
                    // No write — just ensure result is used (via a dummy write of 1 value per WG)
                    (*local)[t] = result;
                    sycl::group_barrier(grp);
                    if (t == 0) {
                        d_output[vec_id] = (*local)[0]; // minimal write to prevent DCE
                    }
                    return;
                }

                if (mode == Mode::FULL_DEQUANT_WRITE) {
                    // Convert to FP16 and write to global memory
                    // Simplified FP32→FP16 in-kernel
                    d_output[vec_id * 128 + t] = result;
                    return;
                }

                if (mode == Mode::FULL_DEQUANT_CSUM) {
                    // Reduce to per-vector checksum via local reduction
                    (*local)[t] = result;
                    sycl::group_barrier(grp);
                    for (int s = 64; s > 0; s >>= 1) {
                        if (t < (unsigned)s) {
                            (*local)[t] += (*local)[t + s];
                        }
                        sycl::group_barrier(grp);
                    }
                    if (t == 0) {
                        d_checksum[vec_id] = (*local)[0];
                    }
                    return;
                }
            });
    });
}

// --- Turbo3 dequant kernel (modes 1-5) ---
void launch_dequant_turbo3(sycl::queue& q, const block_turbo3_0* d_blocks,
                           float* d_output, float* d_checksum,
                           const float* d_tc3, const float* d_ts1, const float* d_ts2,
                           int num_vectors, Mode mode) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(num_vectors * 128, 128),
            [=](sycl::nd_item<1> item) {
                int vec_id = item.get_group(0);
                int t = item.get_local_id(0);
                auto grp = item.get_group();

                const auto& blk = d_blocks[vec_id];

                // Step 1: Unpack (3-bit: low 2 from qs, high 1 from signs)
                uint8_t low2 = (blk.qs[t / 4] >> ((t % 4) * 2)) & 0x3;
                uint8_t high1 = (blk.signs[t / 8] >> (t % 8)) & 0x1;
                uint8_t idx = low2 | (high1 << 2);

                if (mode == Mode::UNPACK_ONLY) {
                    d_output[vec_id * 128 + t] = (float)idx;
                    return;
                }

                // Step 2: Codebook lookup
                float v = d_tc3[idx];

                if (mode == Mode::UNPACK_CODEBOOK) {
                    d_output[vec_id * 128 + t] = v;
                    return;
                }

                // Step 3: Inverse RHT
                v *= d_ts2[t];
                constexpr float tinv = 0.08838834764831845f;
                v *= tinv;

                auto local = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                (*local)[t] = v;
                sycl::group_barrier(grp);

                for (int h_val = 1; h_val < 128; h_val *= 2) {
                    if ((t % (2 * h_val)) < h_val) {
                        float a = (*local)[t];
                        float b = (*local)[t + h_val];
                        (*local)[t]         = a + b;
                        (*local)[t + h_val] = a - b;
                    }
                    sycl::group_barrier(grp);
                }

                float result = (*local)[t] * d_ts1[t];

                // Denormalize
                uint16_t norm_bits = blk.norm;
                uint32_t sign_b = (uint32_t)(norm_bits & 0x8000) << 16;
                uint32_t exp_b  = (norm_bits >> 10) & 0x1F;
                uint32_t mant_b = norm_bits & 0x3FF;
                float norm_val;
                if (exp_b == 0) {
                    uint32_t r = sign_b;
                    __builtin_memcpy(&norm_val, &r, 4);
                } else {
                    uint32_t r = sign_b | ((exp_b + 112) << 23) | (mant_b << 13);
                    __builtin_memcpy(&norm_val, &r, 4);
                }
                result *= norm_val;

                if (mode == Mode::UNPACK_CB_WHT) {
                    (*local)[t] = result;
                    sycl::group_barrier(grp);
                    if (t == 0) d_output[vec_id] = (*local)[0];
                    return;
                }

                if (mode == Mode::FULL_DEQUANT_WRITE) {
                    d_output[vec_id * 128 + t] = result;
                    return;
                }

                if (mode == Mode::FULL_DEQUANT_CSUM) {
                    (*local)[t] = result;
                    sycl::group_barrier(grp);
                    for (int s = 64; s > 0; s >>= 1) {
                        if (t < (unsigned)s) (*local)[t] += (*local)[t + s];
                        sycl::group_barrier(grp);
                    }
                    if (t == 0) d_checksum[vec_id] = (*local)[0];
                    return;
                }
            });
    });
}

// --- Turbo4 pack (quantize) kernel (mode 6) ---
void launch_pack_turbo4(sycl::queue& q, const float* d_input,
                        block_turbo4_0* d_blocks,
                        const float* d_tm4, const float* d_tc4,
                        const float* d_ts1, const float* d_ts2,
                        int num_vectors) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(num_vectors * 128, 128),
            [=](sycl::nd_item<1> item) {
                int vec_id = item.get_group(0);
                int t = item.get_local_id(0);
                auto grp = item.get_group();

                auto wht = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                auto sg_acc = sycl::ext::oneapi::group_local_memory_for_overwrite<float[16]>(grp);

                // Load
                float val = d_input[vec_id * 128 + t];
                (*wht)[t] = val;
                sycl::group_barrier(grp);

                // L2 norm via tree reduction
                auto norm_local = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                (*norm_local)[t] = val * val;
                sycl::group_barrier(grp);
                for (int s = 64; s > 0; s >>= 1) {
                    if (t < (unsigned)s) (*norm_local)[t] += (*norm_local)[t + s];
                    sycl::group_barrier(grp);
                }

                float gnrm;
                if (t == 0) {
                    gnrm = sycl::sqrt((*norm_local)[0]);
                    (*sg_acc)[0] = gnrm;
                }
                sycl::group_barrier(grp);
                gnrm = (*sg_acc)[0];

                // Normalize
                float scale = (gnrm > 1e-10f) ? (1.0f / gnrm) : 0.0f;
                (*wht)[t] *= scale;
                sycl::group_barrier(grp);

                // Forward RHT: TS1 → butterfly → TINV*TS2
                (*wht)[t] *= d_ts1[t];
                sycl::group_barrier(grp);

                for (int h_val = 1; h_val < 128; h_val *= 2) {
                    if ((t % (2 * h_val)) < h_val) {
                        float a = (*wht)[t];
                        float b = (*wht)[t + h_val];
                        (*wht)[t]         = a + b;
                        (*wht)[t + h_val] = a - b;
                    }
                    sycl::group_barrier(grp);
                }

                constexpr float tinv = 0.08838834764831845f;
                float rv = (*wht)[t] * tinv * d_ts2[t];

                // Quantize to nearest of 16 centroids
                uint8_t idx = 0;
                for (int i = 0; i < 15; i++) {
                    if (rv >= d_tm4[i]) idx = i + 1;
                }

                // Pack: 2 nibbles per byte
                // Even thread writes the byte
                uint8_t my_nibble = idx & 0xF;
                (*norm_local)[t] = (float)my_nibble;  // reuse local mem
                sycl::group_barrier(grp);

                if (t % 2 == 0) {
                    uint8_t lo = (uint8_t)(*norm_local)[t];
                    uint8_t hi = (uint8_t)(*norm_local)[t + 1];
                    d_blocks[vec_id].qs[t / 2] = lo | (hi << 4);
                }

                // Reconstruction norm
                float rc = d_tc4[idx] * d_tc4[idx];
                (*norm_local)[t] = rc;
                sycl::group_barrier(grp);
                for (int s = 64; s > 0; s >>= 1) {
                    if (t < (unsigned)s) (*norm_local)[t] += (*norm_local)[t + s];
                    sycl::group_barrier(grp);
                }

                if (t == 0) {
                    float rn = sycl::sqrt((*norm_local)[0]);
                    float corrected = (rn > 1e-10f) ? (gnrm / rn) : gnrm;
                    d_blocks[vec_id].norm = fp32_to_fp16(corrected);
                    d_blocks[vec_id].rnorm = 0;
                }
            });
    });
}

// --- Turbo3 pack (quantize) kernel (mode 6) ---
void launch_pack_turbo3(sycl::queue& q, const float* d_input,
                        block_turbo3_0* d_blocks,
                        const float* d_tm3, const float* d_tc3,
                        const float* d_ts1, const float* d_ts2,
                        int num_vectors) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(num_vectors * 128, 128),
            [=](sycl::nd_item<1> item) {
                int vec_id = item.get_group(0);
                int t = item.get_local_id(0);
                auto grp = item.get_group();

                auto wht = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                auto sg_acc = sycl::ext::oneapi::group_local_memory_for_overwrite<float[16]>(grp);

                float val = d_input[vec_id * 128 + t];
                (*wht)[t] = val;
                sycl::group_barrier(grp);

                auto norm_local = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                (*norm_local)[t] = val * val;
                sycl::group_barrier(grp);
                for (int s = 64; s > 0; s >>= 1) {
                    if (t < (unsigned)s) (*norm_local)[t] += (*norm_local)[t + s];
                    sycl::group_barrier(grp);
                }

                float gnrm;
                if (t == 0) {
                    gnrm = sycl::sqrt((*norm_local)[0]);
                    (*sg_acc)[0] = gnrm;
                }
                sycl::group_barrier(grp);
                gnrm = (*sg_acc)[0];

                float scale = (gnrm > 1e-10f) ? (1.0f / gnrm) : 0.0f;
                (*wht)[t] *= scale;
                sycl::group_barrier(grp);

                (*wht)[t] *= d_ts1[t];
                sycl::group_barrier(grp);

                for (int h_val = 1; h_val < 128; h_val *= 2) {
                    if ((t % (2 * h_val)) < h_val) {
                        float a = (*wht)[t];
                        float b = (*wht)[t + h_val];
                        (*wht)[t]         = a + b;
                        (*wht)[t + h_val] = a - b;
                    }
                    sycl::group_barrier(grp);
                }

                constexpr float tinv = 0.08838834764831845f;
                float rv = (*wht)[t] * tinv * d_ts2[t];

                // Quantize: 8 centroids
                uint8_t idx;
                if      (rv < d_tm3[0]) idx = 0;
                else if (rv < d_tm3[1]) idx = 1;
                else if (rv < d_tm3[2]) idx = 2;
                else if (rv < d_tm3[3]) idx = 3;
                else if (rv < d_tm3[4]) idx = 4;
                else if (rv < d_tm3[5]) idx = 5;
                else if (rv < d_tm3[6]) idx = 6;
                else                     idx = 7;

                // Pack lower 2 bits: 4 per byte
                uint8_t low2 = idx & 0x3;
                (*norm_local)[t] = (float)low2;
                sycl::group_barrier(grp);

                if (t % 4 == 0) {
                    uint8_t byte = 0;
                    for (int k = 0; k < 4; k++) {
                        byte |= (uint8_t)(*norm_local)[t + k] << (k * 2);
                    }
                    d_blocks[vec_id].qs[t / 4] = byte;
                }

                // Pack upper 1 bit: 8 per byte
                uint8_t high1 = (idx >> 2) & 0x1;
                (*norm_local)[t] = (float)high1;
                sycl::group_barrier(grp);

                if (t % 8 == 0) {
                    uint8_t byte = 0;
                    for (int k = 0; k < 8; k++) {
                        byte |= (uint8_t)(*norm_local)[t + k] << k;
                    }
                    d_blocks[vec_id].signs[t / 8] = byte;
                }

                // Reconstruction norm
                float rc = d_tc3[idx] * d_tc3[idx];
                (*norm_local)[t] = rc;
                sycl::group_barrier(grp);
                for (int s = 64; s > 0; s >>= 1) {
                    if (t < (unsigned)s) (*norm_local)[t] += (*norm_local)[t + s];
                    sycl::group_barrier(grp);
                }

                if (t == 0) {
                    float rn = sycl::sqrt((*norm_local)[0]);
                    float corrected = (rn > 1e-10f) ? (gnrm / rn) : gnrm;
                    d_blocks[vec_id].norm = fp32_to_fp16(corrected);
                }
            });
    });
}

// ============================================================================
// Benchmark runner
// ============================================================================

struct BenchResult {
    Mode mode;
    std::string quant_type;
    int n_tokens;
    int n_heads;
    int d_head;
    int total_vectors;
    size_t packed_bytes;
    double time_us;
    double vectors_per_sec;
    double ns_per_vector;
    double gb_s_packed_read;
    double gb_s_fp16_write;
    double est_ms_14k;
    double est_ms_60k;
    double est_ms_128k;
};

void print_result(const BenchResult& r) {
    std::cout << std::fixed;
    std::cout << "mode:                   " << mode_name(r.mode) << std::endl;
    std::cout << "quant_type:             " << r.quant_type << std::endl;
    std::cout << "n_tokens:               " << r.n_tokens << std::endl;
    std::cout << "n_heads:                " << r.n_heads << std::endl;
    std::cout << "d_head:                 " << r.d_head << std::endl;
    std::cout << "total_vectors:          " << r.total_vectors << std::endl;
    std::cout << "packed_bytes:           " << r.packed_bytes << std::endl;
    std::cout << "time_us:                " << std::setprecision(1) << r.time_us << std::endl;
    std::cout << "vectors_per_second:     " << std::setprecision(0) << r.vectors_per_sec << std::endl;
    std::cout << "ns_per_vector:          " << std::setprecision(2) << r.ns_per_vector << std::endl;
    std::cout << "GB_s_packed_read:       " << std::setprecision(2) << r.gb_s_packed_read << std::endl;
    if (r.mode == Mode::FULL_DEQUANT_WRITE) {
        std::cout << "GB_s_fp16_write:        " << std::setprecision(2) << r.gb_s_fp16_write << std::endl;
    }
    std::cout << "estimated_ms_per_token_14K:  " << std::setprecision(3) << r.est_ms_14k << std::endl;
    std::cout << "estimated_ms_per_token_60K:  " << std::setprecision(3) << r.est_ms_60k << std::endl;
    std::cout << "estimated_ms_per_token_128K: " << std::setprecision(3) << r.est_ms_128k << std::endl;
    std::cout << std::endl;
}

template<typename BlockT>
BenchResult run_bench(sycl::queue& q, Mode mode, const std::string& quant_type,
                      int n_tokens, int n_heads, int d_head,
                      int warmup_iters, int bench_iters) {
    const int total_vectors = n_tokens * n_heads * 2;  // K + V
    const size_t block_size = sizeof(BlockT);
    const size_t packed_bytes = (size_t)total_vectors * block_size;

    BenchResult result;
    result.mode = mode;
    result.quant_type = quant_type;
    result.n_tokens = n_tokens;
    result.n_heads = n_heads;
    result.d_head = d_head;
    result.total_vectors = total_vectors;
    result.packed_bytes = packed_bytes;

    // Allocate and initialize packed blocks on CPU, then copy to device
    std::vector<BlockT> blocks(total_vectors);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int v = 0; v < total_vectors; v++) {
        float input[128];
        float vec_scale = 0.5f + 1.0f * (rng() % 1000) / 1000.0f;
        for (int i = 0; i < 128; i++) input[i] = dist(rng) * vec_scale;

        if constexpr (std::is_same_v<BlockT, block_turbo3_0>) {
            tq_ref::quantize_vector_turbo3(input, blocks[v]);
        } else {
            tq_ref::quantize_vector_turbo4(input, blocks[v]);
        }
    }

    auto* d_blocks = sycl::malloc_device<BlockT>(total_vectors, q);
    auto* d_output = sycl::malloc_device<float>(total_vectors * 128, q);
    auto* d_checksum = sycl::malloc_device<float>(total_vectors, q);
    q.memcpy(d_blocks, blocks.data(), total_vectors * sizeof(BlockT)).wait();

    // Constants
    auto* d_ts1 = sycl::malloc_device<float>(128, q);
    auto* d_ts2 = sycl::malloc_device<float>(128, q);
    q.memcpy(d_ts1, TS1, 128 * sizeof(float));
    q.memcpy(d_ts2, TS2, 128 * sizeof(float));

    float* d_codebook = nullptr;
    float* d_thresholds = nullptr;
    float* d_input_fp32 = nullptr;

    if constexpr (std::is_same_v<BlockT, block_turbo3_0>) {
        d_codebook = sycl::malloc_device<float>(8, q);
        q.memcpy(d_codebook, TC3, 8 * sizeof(float));
        if (mode == Mode::PACK_QUANTIZE) {
            d_thresholds = sycl::malloc_device<float>(7, q);
            q.memcpy(d_thresholds, TM3, 7 * sizeof(float));
        }
    } else {
        d_codebook = sycl::malloc_device<float>(16, q);
        q.memcpy(d_codebook, TC4, 16 * sizeof(float));
        if (mode == Mode::PACK_QUANTIZE) {
            d_thresholds = sycl::malloc_device<float>(15, q);
            q.memcpy(d_thresholds, TM4, 15 * sizeof(float));
        }
    }

    if (mode == Mode::PACK_QUANTIZE) {
        // Allocate FP32 input for quantization
        std::vector<float> input_data(total_vectors * 128);
        for (int v = 0; v < total_vectors; v++) {
            float vec_scale = 0.5f + 1.0f * (rng() % 1000) / 1000.0f;
            for (int i = 0; i < 128; i++) {
                input_data[v * 128 + i] = dist(rng) * vec_scale;
            }
        }
        d_input_fp32 = sycl::malloc_device<float>(total_vectors * 128, q);
        q.memcpy(d_input_fp32, input_data.data(), total_vectors * 128 * sizeof(float));
    }
    q.wait();

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        if (mode == Mode::PACK_QUANTIZE) {
            if constexpr (std::is_same_v<BlockT, block_turbo4_0>) {
                launch_pack_turbo4(q, d_input_fp32, d_blocks, d_thresholds, d_codebook, d_ts1, d_ts2, total_vectors);
            } else {
                launch_pack_turbo3(q, d_input_fp32, d_blocks, d_thresholds, d_codebook, d_ts1, d_ts2, total_vectors);
            }
        } else {
            if constexpr (std::is_same_v<BlockT, block_turbo3_0>) {
                launch_dequant_turbo3(q, d_blocks, d_output, d_checksum, d_codebook, d_ts1, d_ts2, total_vectors, mode);
            } else {
                launch_dequant_turbo4(q, d_blocks, d_output, d_checksum, d_codebook, d_ts1, d_ts2, total_vectors, mode);
            }
        }
    }
    q.wait();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; i++) {
        if (mode == Mode::PACK_QUANTIZE) {
            if constexpr (std::is_same_v<BlockT, block_turbo4_0>) {
                launch_pack_turbo4(q, d_input_fp32, d_blocks, d_thresholds, d_codebook, d_ts1, d_ts2, total_vectors);
            } else {
                launch_pack_turbo3(q, d_input_fp32, d_blocks, d_thresholds, d_codebook, d_ts1, d_ts2, total_vectors);
            }
        } else {
            if constexpr (std::is_same_v<BlockT, block_turbo3_0>) {
                launch_dequant_turbo3(q, d_blocks, d_output, d_checksum, d_codebook, d_ts1, d_ts2, total_vectors, mode);
            } else {
                launch_dequant_turbo4(q, d_blocks, d_output, d_checksum, d_codebook, d_ts1, d_ts2, total_vectors, mode);
            }
        }
    }
    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double avg_us = total_us / bench_iters;

    result.time_us = avg_us;
    result.vectors_per_sec = total_vectors / (avg_us * 1e-6);
    result.ns_per_vector = avg_us * 1000.0 / total_vectors;
    result.gb_s_packed_read = (packed_bytes / 1e9) / (avg_us * 1e-6);

    // FP16 write bandwidth (mode 4: writes 128 FP32 floats per vector = 512 bytes)
    // In a real implementation this would be FP16 = 256 bytes, but we write FP32 here
    size_t write_bytes = (size_t)total_vectors * 128 * sizeof(float);
    result.gb_s_fp16_write = (write_bytes / 1e9) / (avg_us * 1e-6);

    // Estimate ms/token at different context lengths
    // Each token has n_heads * 2 vectors (K + V), each vector needs dequant
    double ns_per_token_vec = result.ns_per_vector;
    int vecs_per_token = n_heads * 2;

    auto est_ms = [&](int ctx_len) -> double {
        // All tokens' KV need to be dequantized for attention
        double total_vecs = (double)ctx_len * vecs_per_token;
        return total_vecs * ns_per_token_vec / 1e6;
    };

    result.est_ms_14k = est_ms(14000);
    result.est_ms_60k = est_ms(60000);
    result.est_ms_128k = est_ms(128000);

    // Cleanup
    sycl::free(d_blocks, q);
    sycl::free(d_output, q);
    sycl::free(d_checksum, q);
    sycl::free(d_ts1, q);
    sycl::free(d_ts2, q);
    sycl::free(d_codebook, q);
    if (d_thresholds) sycl::free(d_thresholds, q);
    if (d_input_fp32) sycl::free(d_input_fp32, q);

    return result;
}

int main(int argc, char** argv) {
    std::cout << "============================================" << std::endl;
    std::cout << "TurboQuant SYCL Microbenchmark" << std::endl;
    std::cout << "============================================" << std::endl;

    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    auto dev = q.get_device();
    std::cout << "Device:         " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Max WG size:    " << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "Local mem:      " << dev.get_info<sycl::info::device::local_mem_size>() << " bytes" << std::endl;
    std::cout << "Max compute:    " << dev.get_info<sycl::info::device::max_compute_units>() << " CUs" << std::endl;
    std::cout << "Global mem:     " << dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024) << " MB" << std::endl;
    std::cout << std::endl;

    // Configuration from task spec
    const int d_head = 128;
    const int n_heads = 40;  // Qwen3.6
    const int warmup = 10;
    const int iters = 100;

    std::vector<int> token_counts = {1000, 4000, 14000, 60000, 128000};
    std::vector<Mode> modes = {
        Mode::UNPACK_ONLY,
        Mode::UNPACK_CODEBOOK,
        Mode::UNPACK_CB_WHT,
        Mode::FULL_DEQUANT_WRITE,
        Mode::FULL_DEQUANT_CSUM,
        Mode::PACK_QUANTIZE,
    };

    // Parse optional args
    int max_tokens = 128000;
    bool run_turbo3 = true;
    bool run_turbo4 = true;
    bool skip_pack = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--turbo4-only") {
            run_turbo3 = false;
        } else if (arg == "--turbo3-only") {
            run_turbo4 = false;
        } else if (arg == "--skip-pack") {
            skip_pack = true;
        }
    }

    std::vector<BenchResult> all_results;

    for (int n_tokens : token_counts) {
        if (n_tokens > max_tokens) continue;

        // Check memory: turbo4 block is 68 bytes, turbo3 is 50 bytes
        // total_vectors = n_tokens * n_heads * 2
        int total_vecs = n_tokens * n_heads * 2;
        size_t mem_needed_t4 = (size_t)total_vecs * (68 + 128 * 4 + 4);  // blocks + output + checksum
        size_t mem_needed_t3 = (size_t)total_vecs * (50 + 128 * 4 + 4);

        size_t dev_mem = dev.get_info<sycl::info::device::global_mem_size>();
        if (std::max(mem_needed_t4, mem_needed_t3) > dev_mem * 0.8) {
            std::cout << "SKIPPING n_tokens=" << n_tokens << " (would need "
                      << std::max(mem_needed_t4, mem_needed_t3) / (1024*1024) << " MB, have "
                      << dev_mem / (1024*1024) << " MB)" << std::endl;
            continue;
        }

        for (Mode mode : modes) {
            if (skip_pack && mode == Mode::PACK_QUANTIZE) continue;
            if (run_turbo4) {
                std::cout << "--- turbo4 | " << mode_name(mode) << " | " << n_tokens << " tokens ---" << std::endl;
                auto r = run_bench<block_turbo4_0>(q, mode, "turbo4", n_tokens, n_heads, d_head, warmup, iters);
                print_result(r);
                all_results.push_back(r);
            }

            if (run_turbo3) {
                std::cout << "--- turbo3 | " << mode_name(mode) << " | " << n_tokens << " tokens ---" << std::endl;
                auto r = run_bench<block_turbo3_0>(q, mode, "turbo3", n_tokens, n_heads, d_head, warmup, iters);
                print_result(r);
                all_results.push_back(r);
            }
        }
    }

    // Summary: Mode 4 vs Mode 5 comparison
    std::cout << "============================================" << std::endl;
    std::cout << "CRITICAL COMPARISON: Mode 4 (split write) vs Mode 5 (checksum)" << std::endl;
    std::cout << "============================================" << std::endl;

    for (const auto& r4 : all_results) {
        if (r4.mode != Mode::FULL_DEQUANT_WRITE) continue;
        for (const auto& r5 : all_results) {
            if (r5.mode != Mode::FULL_DEQUANT_CSUM) continue;
            if (r4.quant_type != r5.quant_type || r4.n_tokens != r5.n_tokens) continue;

            double overhead_us = r4.time_us - r5.time_us;
            double overhead_pct = 100.0 * overhead_us / r5.time_us;

            std::cout << r4.quant_type << " | " << r4.n_tokens << " tokens:" << std::endl;
            std::cout << "  Mode 4 (write):    " << std::fixed << std::setprecision(1) << r4.time_us << " us" << std::endl;
            std::cout << "  Mode 5 (checksum): " << r5.time_us << " us" << std::endl;
            std::cout << "  Write overhead:    " << overhead_us << " us (" << std::setprecision(1) << overhead_pct << "%)" << std::endl;
            std::cout << "  Est ms/token 14K:  write=" << std::setprecision(3) << r4.est_ms_14k
                      << " csum=" << r5.est_ms_14k << " delta=" << r4.est_ms_14k - r5.est_ms_14k << std::endl;
            std::cout << std::endl;
        }
    }

    return 0;
}
