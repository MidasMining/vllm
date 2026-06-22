#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include "../src/tq_types.h"
#include "../src/tq_codebook.h"

// ============================================================================
// Pure C++ CPU reference implementation of TurboQuant
// No SYCL dependencies — used as the correctness oracle
// ============================================================================

namespace tq_ref {

// ----------------------------------------------------------------------------
// Walsh-Hadamard Transform (forward): in-place, unnormalized butterfly
// 7 stages for n=128: log2(128) = 7
// ----------------------------------------------------------------------------
inline void wht_butterfly_inplace(float* v, int n) {
    for (int h = 1; h < n; h *= 2) {
        for (int i = 0; i < n; i += 2 * h) {
            for (int j = i; j < i + h; j++) {
                float a = v[j];
                float b = v[j + h];
                v[j]     = a + b;
                v[j + h] = a - b;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Forward Randomized Hadamard Transform
// v_rot = TS2 * TINV * H * (TS1 * v_hat)
// Returns the rotated vector in-place
// ----------------------------------------------------------------------------
inline void forward_rht(float* v, int n = 128) {
    // Step 1: Apply sign vector TS1
    for (int i = 0; i < n; i++) {
        v[i] *= TS1[i];
    }

    // Step 2: Butterfly WHT (unnormalized Hadamard)
    wht_butterfly_inplace(v, n);

    // Step 3: Apply TINV scaling and sign vector TS2
    for (int i = 0; i < n; i++) {
        v[i] *= TINV * TS2[i];
    }
}

// ----------------------------------------------------------------------------
// Inverse Randomized Hadamard Transform
// v_hat = TS1 * H * TINV * (TS2 * v_rot)
// The RHT is self-inverse (up to scaling already encoded in TINV)
// ----------------------------------------------------------------------------
inline void inverse_rht(float* v, int n = 128) {
    // Step 1: Apply sign vector TS2
    for (int i = 0; i < n; i++) {
        v[i] *= TS2[i];
    }

    // Step 2: Apply TINV scaling
    for (int i = 0; i < n; i++) {
        v[i] *= TINV;
    }

    // Step 3: Butterfly WHT
    wht_butterfly_inplace(v, n);

    // Step 4: Apply sign vector TS1
    for (int i = 0; i < n; i++) {
        v[i] *= TS1[i];
    }
}

// ----------------------------------------------------------------------------
// Quantize to nearest centroid index
// ----------------------------------------------------------------------------
inline uint8_t quantize_turbo3(float v) {
    // 8 centroids, 7 thresholds
    if (v < TM3[0]) return 0;
    if (v < TM3[1]) return 1;
    if (v < TM3[2]) return 2;
    if (v < TM3[3]) return 3;
    if (v < TM3[4]) return 4;
    if (v < TM3[5]) return 5;
    if (v < TM3[6]) return 6;
    return 7;
}

inline uint8_t quantize_turbo4(float v) {
    // 16 centroids, 15 thresholds — linear scan matching shader
    uint8_t idx = 0;
    for (int i = 0; i < 15; i++) {
        if (v >= TM4[i]) idx = i + 1;
    }
    return idx;
}

// ----------------------------------------------------------------------------
// Pack turbo3: 128 indices (3-bit each) → block_turbo3_0
// Lower 2 bits → qs[32], upper 1 bit → signs[16]
// ----------------------------------------------------------------------------
inline void pack_turbo3(const uint8_t* indices, float norm_val, block_turbo3_0& block) {
    block.norm = fp32_to_fp16(norm_val);
    std::memset(block.qs, 0, sizeof(block.qs));
    std::memset(block.signs, 0, sizeof(block.signs));

    for (int i = 0; i < 128; i++) {
        uint8_t idx = indices[i];
        // Lower 2 bits: 4 elements per byte
        uint8_t low2 = idx & 0x3;
        block.qs[i / 4] |= low2 << ((i % 4) * 2);
        // Upper 1 bit: 8 elements per byte
        uint8_t high1 = (idx >> 2) & 0x1;
        block.signs[i / 8] |= high1 << (i % 8);
    }
}

// ----------------------------------------------------------------------------
// Unpack turbo3: block_turbo3_0 → 128 indices
// ----------------------------------------------------------------------------
inline void unpack_turbo3(const block_turbo3_0& block, uint8_t* indices) {
    for (int i = 0; i < 128; i++) {
        uint8_t low2 = (block.qs[i / 4] >> ((i % 4) * 2)) & 0x3;
        uint8_t high1 = (block.signs[i / 8] >> (i % 8)) & 0x1;
        indices[i] = low2 | (high1 << 2);
    }
}

// ----------------------------------------------------------------------------
// Pack turbo4: 128 indices (4-bit each) → block_turbo4_0
// 2 elements per byte (nibble packed)
// ----------------------------------------------------------------------------
inline void pack_turbo4(const uint8_t* indices, float norm_val, block_turbo4_0& block) {
    block.norm = fp32_to_fp16(norm_val);
    block.rnorm = 0;  // reserved
    std::memset(block.qs, 0, sizeof(block.qs));

    for (int i = 0; i < 128; i += 2) {
        block.qs[i / 2] = (indices[i] & 0xF) | ((indices[i + 1] & 0xF) << 4);
    }
}

// ----------------------------------------------------------------------------
// Unpack turbo4: block_turbo4_0 → 128 indices
// ----------------------------------------------------------------------------
inline void unpack_turbo4(const block_turbo4_0& block, uint8_t* indices) {
    for (int i = 0; i < 128; i += 2) {
        indices[i]     = block.qs[i / 2] & 0xF;
        indices[i + 1] = (block.qs[i / 2] >> 4) & 0xF;
    }
}

// ----------------------------------------------------------------------------
// Full quantization pipeline (FP32 vector → packed block)
// Returns the reconstruction norm (corrected norm for denormalization)
// ----------------------------------------------------------------------------
inline float quantize_vector_turbo3(const float* input, block_turbo3_0& block) {
    float v[128];
    std::memcpy(v, input, 128 * sizeof(float));

    // Step 1: Compute L2 norm
    float norm_sq = 0.0f;
    for (int i = 0; i < 128; i++) norm_sq += v[i] * v[i];
    float gnrm = std::sqrt(norm_sq);

    // Step 2: Normalize
    float scale = (gnrm > 1e-10f) ? (1.0f / gnrm) : 0.0f;
    for (int i = 0; i < 128; i++) v[i] *= scale;

    // Step 3: Forward RHT
    forward_rht(v);

    // Step 4: Quantize each coordinate
    uint8_t indices[128];
    for (int i = 0; i < 128; i++) {
        indices[i] = quantize_turbo3(v[i]);
    }

    // Step 5: Compute reconstruction norm
    float rc_sq = 0.0f;
    for (int i = 0; i < 128; i++) {
        float c = TC3[indices[i]];
        rc_sq += c * c;
    }
    float rn = std::sqrt(rc_sq);
    float corrected_norm = (rn > 1e-10f) ? (gnrm / rn) : gnrm;

    // Step 6: Pack
    pack_turbo3(indices, corrected_norm, block);

    return corrected_norm;
}

inline float quantize_vector_turbo4(const float* input, block_turbo4_0& block) {
    float v[128];
    std::memcpy(v, input, 128 * sizeof(float));

    // Step 1: Compute L2 norm
    float norm_sq = 0.0f;
    for (int i = 0; i < 128; i++) norm_sq += v[i] * v[i];
    float gnrm = std::sqrt(norm_sq);

    // Step 2: Normalize
    float scale = (gnrm > 1e-10f) ? (1.0f / gnrm) : 0.0f;
    for (int i = 0; i < 128; i++) v[i] *= scale;

    // Step 3: Forward RHT
    forward_rht(v);

    // Step 4: Quantize
    uint8_t indices[128];
    for (int i = 0; i < 128; i++) {
        indices[i] = quantize_turbo4(v[i]);
    }

    // Step 5: Reconstruction norm
    float rc_sq = 0.0f;
    for (int i = 0; i < 128; i++) {
        float c = TC4[indices[i]];
        rc_sq += c * c;
    }
    float rn = std::sqrt(rc_sq);
    float corrected_norm = (rn > 1e-10f) ? (gnrm / rn) : gnrm;

    // Step 6: Pack
    pack_turbo4(indices, corrected_norm, block);

    return corrected_norm;
}

// ----------------------------------------------------------------------------
// Full dequantization pipeline (packed block → FP32 vector)
// ----------------------------------------------------------------------------
inline void dequantize_vector_turbo3(const block_turbo3_0& block, float* output) {
    // Step 1: Unpack indices
    uint8_t indices[128];
    unpack_turbo3(block, indices);

    // Step 2: Codebook lookup
    for (int i = 0; i < 128; i++) {
        output[i] = TC3[indices[i]];
    }

    // Step 3: Inverse RHT
    inverse_rht(output);

    // Step 4: Denormalize
    float norm_val = fp16_to_fp32(block.norm);
    for (int i = 0; i < 128; i++) {
        output[i] *= norm_val;
    }
}

inline void dequantize_vector_turbo4(const block_turbo4_0& block, float* output) {
    // Step 1: Unpack indices
    uint8_t indices[128];
    unpack_turbo4(block, indices);

    // Step 2: Codebook lookup
    for (int i = 0; i < 128; i++) {
        output[i] = TC4[indices[i]];
    }

    // Step 3: Inverse RHT
    inverse_rht(output);

    // Step 4: Denormalize
    float norm_val = fp16_to_fp32(block.norm);
    for (int i = 0; i < 128; i++) {
        output[i] *= norm_val;
    }
}

// ----------------------------------------------------------------------------
// Compute MSE between original and reconstructed vectors
// ----------------------------------------------------------------------------
inline float compute_mse(const float* original, const float* reconstructed, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = original[i] - reconstructed[i];
        sum += diff * diff;
    }
    return sum / n;
}

inline float compute_max_abs_error(const float* original, const float* reconstructed, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = std::fabs(original[i] - reconstructed[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Normalized MSE: MSE / variance_of_original
inline float compute_nmse(const float* original, const float* reconstructed, int n) {
    float mse = compute_mse(original, reconstructed, n);
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += original[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = original[i] - mean;
        var += d * d;
    }
    var /= n;
    return (var > 1e-20f) ? (mse / var) : 0.0f;
}

} // namespace tq_ref
