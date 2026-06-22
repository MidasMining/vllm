#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <vector>
#include <string>
#include "../reference/tq_cpu_reference.h"

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

// ============================================================================
// TurboQuant Correctness Tests
// Validates: WHT round-trip, pack/unpack, full quant/dequant pipeline
// ============================================================================

static bool test_wht_roundtrip() {
    std::cout << "=== Test: WHT Round-Trip ===" << std::endl;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float v_orig[128], v[128];
    for (int i = 0; i < 128; i++) {
        v_orig[i] = dist(rng);
        v[i] = v_orig[i];
    }

    // Forward RHT
    tq_ref::forward_rht(v);

    // Inverse RHT
    tq_ref::inverse_rht(v);

    // Check round-trip error
    float max_err = 0.0f;
    for (int i = 0; i < 128; i++) {
        float err = std::fabs(v[i] - v_orig[i]);
        if (err > max_err) max_err = err;
    }

    std::cout << "  Max absolute error after WHT round-trip: " << std::scientific << max_err << std::endl;
    bool pass = max_err < 1e-5f;
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

static bool test_pack_unpack_turbo3() {
    std::cout << "\n=== Test: Turbo3 Pack/Unpack ===" << std::endl;

    uint8_t indices[128], unpacked[128];
    std::mt19937 rng(123);
    for (int i = 0; i < 128; i++) {
        indices[i] = rng() % 8;  // 3-bit range
    }

    block_turbo3_0 block;
    tq_ref::pack_turbo3(indices, 1.0f, block);
    tq_ref::unpack_turbo3(block, unpacked);

    bool pass = true;
    for (int i = 0; i < 128; i++) {
        if (indices[i] != unpacked[i]) {
            std::cout << "  MISMATCH at index " << i << ": expected " << (int)indices[i]
                      << " got " << (int)unpacked[i] << std::endl;
            pass = false;
        }
    }
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

static bool test_pack_unpack_turbo4() {
    std::cout << "\n=== Test: Turbo4 Pack/Unpack ===" << std::endl;

    uint8_t indices[128], unpacked[128];
    std::mt19937 rng(456);
    for (int i = 0; i < 128; i++) {
        indices[i] = rng() % 16;  // 4-bit range
    }

    block_turbo4_0 block;
    tq_ref::pack_turbo4(indices, 1.0f, block);
    tq_ref::unpack_turbo4(block, unpacked);

    bool pass = true;
    for (int i = 0; i < 128; i++) {
        if (indices[i] != unpacked[i]) {
            std::cout << "  MISMATCH at index " << i << ": expected " << (int)indices[i]
                      << " got " << (int)unpacked[i] << std::endl;
            pass = false;
        }
    }
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

static bool test_quantize_dequantize(const std::string& name, int num_vectors, int quant_type) {
    std::cout << "\n=== Test: " << name << " Quantize/Dequantize (" << num_vectors << " vectors) ===" << std::endl;

    std::mt19937 rng(789);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    double total_mse = 0.0;
    double total_nmse = 0.0;
    float max_abs_err = 0.0f;
    float max_nmse = 0.0f;

    for (int v = 0; v < num_vectors; v++) {
        float input[128], output[128];

        // Generate random FP16-range vector (typical KV cache magnitudes)
        float vec_scale = 0.1f + 2.0f * (rng() % 1000) / 1000.0f;
        for (int i = 0; i < 128; i++) {
            input[i] = dist(rng) * vec_scale;
        }

        if (quant_type == 3) {
            block_turbo3_0 block;
            tq_ref::quantize_vector_turbo3(input, block);
            tq_ref::dequantize_vector_turbo3(block, output);
        } else {
            block_turbo4_0 block;
            tq_ref::quantize_vector_turbo4(input, block);
            tq_ref::dequantize_vector_turbo4(block, output);
        }

        float mse = tq_ref::compute_mse(input, output, 128);
        float nmse = tq_ref::compute_nmse(input, output, 128);
        float mae = tq_ref::compute_max_abs_error(input, output, 128);

        total_mse += mse;
        total_nmse += nmse;
        if (mae > max_abs_err) max_abs_err = mae;
        if (nmse > max_nmse) max_nmse = nmse;
    }

    double avg_mse = total_mse / num_vectors;
    double avg_nmse = total_nmse / num_vectors;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Average MSE:       " << avg_mse << std::endl;
    std::cout << "  Average NMSE:      " << avg_nmse << std::endl;
    std::cout << "  Max abs error:     " << max_abs_err << std::endl;
    std::cout << "  Max NMSE:          " << max_nmse << std::endl;

    // Expected NMSE from paper: turbo4 ~ 0.009, turbo3 ~ 0.034
    float expected_nmse = (quant_type == 4) ? 0.009f : 0.034f;
    float tolerance = 0.01f;  // Allow some slack
    bool pass = (avg_nmse < expected_nmse + tolerance) && (avg_nmse > 0.001f);

    std::cout << "  Expected NMSE ~" << expected_nmse << " (tolerance +" << tolerance << ")" << std::endl;
    std::cout << "  " << (pass ? "PASS" : "CHECK") << std::endl;
    return pass;
}

#ifdef USE_SYCL
static bool test_sycl_dequant_turbo3(sycl::queue& q) {
    std::cout << "\n=== Test: SYCL Turbo3 Dequant vs CPU Reference ===" << std::endl;

    const int num_vectors = 1000;
    std::mt19937 rng(111);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Quantize on CPU
    std::vector<block_turbo3_0> blocks(num_vectors);
    std::vector<std::vector<float>> originals(num_vectors);

    for (int v = 0; v < num_vectors; v++) {
        originals[v].resize(128);
        float vec_scale = 0.1f + 2.0f * (rng() % 1000) / 1000.0f;
        for (int i = 0; i < 128; i++) {
            originals[v][i] = dist(rng) * vec_scale;
        }
        tq_ref::quantize_vector_turbo3(originals[v].data(), blocks[v]);
    }

    // Allocate SYCL buffers
    auto* d_blocks = sycl::malloc_device<block_turbo3_0>(num_vectors, q);
    auto* d_output = sycl::malloc_device<float>(num_vectors * 128, q);
    q.memcpy(d_blocks, blocks.data(), num_vectors * sizeof(block_turbo3_0)).wait();

    // Codebook and sign vectors as device constants
    auto* d_tc3 = sycl::malloc_device<float>(8, q);
    auto* d_ts1 = sycl::malloc_device<float>(128, q);
    auto* d_ts2 = sycl::malloc_device<float>(128, q);
    q.memcpy(d_tc3, TC3, 8 * sizeof(float));
    q.memcpy(d_ts1, TS1, 128 * sizeof(float));
    q.memcpy(d_ts2, TS2, 128 * sizeof(float));
    q.wait();

    // Launch SYCL dequant kernel: one work-group of 128 threads per vector
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(num_vectors * 128, 128),
            [=](sycl::nd_item<1> item) {
                int vec_id = item.get_group(0);
                int t = item.get_local_id(0);
                auto grp = item.get_group();

                const auto& blk = d_blocks[vec_id];

                // Step 1: Unpack
                uint8_t low2 = (blk.qs[t / 4] >> ((t % 4) * 2)) & 0x3;
                uint8_t high1 = (blk.signs[t / 8] >> (t % 8)) & 0x1;
                uint8_t idx = low2 | (high1 << 2);

                // Step 2: Codebook lookup
                float v = d_tc3[idx];

                // Step 3: Inverse RHT
                // Apply TS2
                v *= d_ts2[t];

                // Scale by TINV
                constexpr float tinv = 0.08838834764831845f;
                v *= tinv;

                // Store to local memory for butterfly
                auto local = sycl::ext::oneapi::group_local_memory_for_overwrite<float[128]>(grp);
                (*local)[t] = v;
                sycl::group_barrier(grp);

                // Butterfly WHT
                for (int h_val = 1; h_val < 128; h_val *= 2) {
                    if ((t % (2 * h_val)) < h_val) {
                        float a = (*local)[t];
                        float b = (*local)[t + h_val];
                        (*local)[t]         = a + b;
                        (*local)[t + h_val] = a - b;
                    }
                    sycl::group_barrier(grp);
                }

                // Apply TS1
                float result = (*local)[t] * d_ts1[t];

                // Step 4: Denormalize
                // FP16 → FP32 norm
                uint16_t norm_bits = blk.norm;
                uint32_t sign = (uint32_t)(norm_bits & 0x8000) << 16;
                uint32_t exp  = (norm_bits >> 10) & 0x1F;
                uint32_t mant = norm_bits & 0x3FF;
                float norm_val;
                if (exp == 0) {
                    uint32_t r = sign;
                    __builtin_memcpy(&norm_val, &r, 4);
                } else {
                    uint32_t r = sign | ((exp + 112) << 23) | (mant << 13);
                    __builtin_memcpy(&norm_val, &r, 4);
                }

                result *= norm_val;
                d_output[vec_id * 128 + t] = result;
            });
    }).wait();

    // Read back and compare
    std::vector<float> gpu_output(num_vectors * 128);
    q.memcpy(gpu_output.data(), d_output, num_vectors * 128 * sizeof(float)).wait();

    double total_nmse = 0.0;
    float max_diff = 0.0f;
    int mismatches = 0;

    for (int v = 0; v < num_vectors; v++) {
        float cpu_output[128];
        tq_ref::dequantize_vector_turbo3(blocks[v], cpu_output);

        for (int i = 0; i < 128; i++) {
            float diff = std::fabs(gpu_output[v * 128 + i] - cpu_output[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-4f) mismatches++;
        }

        float nmse = tq_ref::compute_nmse(originals[v].data(), &gpu_output[v * 128], 128);
        total_nmse += nmse;
    }

    double avg_nmse = total_nmse / num_vectors;
    std::cout << "  GPU vs CPU max diff: " << std::scientific << max_diff << std::endl;
    std::cout << "  GPU avg NMSE:        " << std::fixed << std::setprecision(6) << avg_nmse << std::endl;
    std::cout << "  Mismatches (>1e-4):  " << mismatches << std::endl;

    bool pass = (max_diff < 1e-3f) && (mismatches == 0);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;

    sycl::free(d_blocks, q);
    sycl::free(d_output, q);
    sycl::free(d_tc3, q);
    sycl::free(d_ts1, q);
    sycl::free(d_ts2, q);

    return pass;
}

static bool test_sycl_dequant_turbo4(sycl::queue& q) {
    std::cout << "\n=== Test: SYCL Turbo4 Dequant vs CPU Reference ===" << std::endl;

    const int num_vectors = 1000;
    std::mt19937 rng(222);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<block_turbo4_0> blocks(num_vectors);
    std::vector<std::vector<float>> originals(num_vectors);

    for (int v = 0; v < num_vectors; v++) {
        originals[v].resize(128);
        float vec_scale = 0.1f + 2.0f * (rng() % 1000) / 1000.0f;
        for (int i = 0; i < 128; i++) {
            originals[v][i] = dist(rng) * vec_scale;
        }
        tq_ref::quantize_vector_turbo4(originals[v].data(), blocks[v]);
    }

    auto* d_blocks = sycl::malloc_device<block_turbo4_0>(num_vectors, q);
    auto* d_output = sycl::malloc_device<float>(num_vectors * 128, q);
    q.memcpy(d_blocks, blocks.data(), num_vectors * sizeof(block_turbo4_0)).wait();

    auto* d_tc4 = sycl::malloc_device<float>(16, q);
    auto* d_ts1 = sycl::malloc_device<float>(128, q);
    auto* d_ts2 = sycl::malloc_device<float>(128, q);
    q.memcpy(d_tc4, TC4, 16 * sizeof(float));
    q.memcpy(d_ts1, TS1, 128 * sizeof(float));
    q.memcpy(d_ts2, TS2, 128 * sizeof(float));
    q.wait();

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

                // Step 2: Codebook lookup
                float v = d_tc4[idx];

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

                // Step 4: Denormalize
                uint16_t norm_bits = blk.norm;
                uint32_t sign = (uint32_t)(norm_bits & 0x8000) << 16;
                uint32_t exp  = (norm_bits >> 10) & 0x1F;
                uint32_t mant = norm_bits & 0x3FF;
                float norm_val;
                if (exp == 0) {
                    uint32_t r = sign;
                    __builtin_memcpy(&norm_val, &r, 4);
                } else {
                    uint32_t r = sign | ((exp + 112) << 23) | (mant << 13);
                    __builtin_memcpy(&norm_val, &r, 4);
                }

                result *= norm_val;
                d_output[vec_id * 128 + t] = result;
            });
    }).wait();

    std::vector<float> gpu_output(num_vectors * 128);
    q.memcpy(gpu_output.data(), d_output, num_vectors * 128 * sizeof(float)).wait();

    double total_nmse = 0.0;
    float max_diff = 0.0f;
    int mismatches = 0;

    for (int v = 0; v < num_vectors; v++) {
        float cpu_output[128];
        tq_ref::dequantize_vector_turbo4(blocks[v], cpu_output);

        for (int i = 0; i < 128; i++) {
            float diff = std::fabs(gpu_output[v * 128 + i] - cpu_output[i]);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-4f) mismatches++;
        }

        float nmse = tq_ref::compute_nmse(originals[v].data(), &gpu_output[v * 128], 128);
        total_nmse += nmse;
    }

    double avg_nmse = total_nmse / num_vectors;
    std::cout << "  GPU vs CPU max diff: " << std::scientific << max_diff << std::endl;
    std::cout << "  GPU avg NMSE:        " << std::fixed << std::setprecision(6) << avg_nmse << std::endl;
    std::cout << "  Mismatches (>1e-4):  " << mismatches << std::endl;

    bool pass = (max_diff < 1e-3f) && (mismatches == 0);
    std::cout << "  " << (pass ? "PASS" : "FAIL") << std::endl;

    sycl::free(d_blocks, q);
    sycl::free(d_output, q);
    sycl::free(d_tc4, q);
    sycl::free(d_ts1, q);
    sycl::free(d_ts2, q);

    return pass;
}
#endif

int main(int argc, char** argv) {
    std::cout << "============================================" << std::endl;
    std::cout << "TurboQuant Correctness Test Suite" << std::endl;
    std::cout << "============================================" << std::endl;

    int pass = 0, fail = 0;

    auto check = [&](bool result) { result ? pass++ : fail++; };

    // CPU-only tests
    check(test_wht_roundtrip());
    check(test_pack_unpack_turbo3());
    check(test_pack_unpack_turbo4());
    check(test_quantize_dequantize("Turbo3", 10000, 3));
    check(test_quantize_dequantize("Turbo4", 10000, 4));

#ifdef USE_SYCL
    std::cout << "\n--- SYCL GPU Tests ---" << std::endl;
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
        auto dev = q.get_device();
        std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Max WG size: " << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
        std::cout << "Local mem: " << dev.get_info<sycl::info::device::local_mem_size>() << " bytes" << std::endl;

        check(test_sycl_dequant_turbo3(q));
        check(test_sycl_dequant_turbo4(q));
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        fail += 2;
    }
#else
    std::cout << "\n(SYCL tests skipped — build with -DUSE_SYCL)" << std::endl;
#endif

    std::cout << "\n============================================" << std::endl;
    std::cout << "Results: " << pass << " passed, " << fail << " failed" << std::endl;
    std::cout << "============================================" << std::endl;

    return fail > 0 ? 1 : 0;
}
