#pragma once
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include "../src/tq_types.h"
#include "../src/tq_codebook.h"
#include "tq_cpu_reference.h"

// ============================================================================
// CPU reference for attention computation — standard and rotated-space
// Used as correctness oracle for Phase 0.5
// ============================================================================

namespace tq_attn {

// Standard attention (Path A): full dequant K,V then dot products + weighted sum
// Returns output[n_heads][128]
inline void standard_attention_turbo4(
    const float* queries,                  // [n_heads * 128]
    const block_turbo4_0* k_blocks,        // [n_heads * n_tokens]
    const block_turbo4_0* v_blocks,        // [n_heads * n_tokens]
    float* output,                         // [n_heads * 128]
    int n_heads, int n_tokens)
{
    for (int h = 0; h < n_heads; h++) {
        // Dequant all K for this head and compute scores
        std::vector<float> scores(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            float k_vec[128];
            tq_ref::dequantize_vector_turbo4(k_blocks[h * n_tokens + t], k_vec);
            float dot = 0.0f;
            for (int d = 0; d < 128; d++)
                dot += queries[h * 128 + d] * k_vec[d];
            scores[t] = dot;
        }

        // Softmax
        float max_score = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        std::vector<float> weights(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            weights[t] = std::exp(scores[t] - max_score);
            sum_exp += weights[t];
        }
        for (int t = 0; t < n_tokens; t++)
            weights[t] /= sum_exp;

        // Weighted V sum
        float out[128] = {};
        for (int t = 0; t < n_tokens; t++) {
            float v_vec[128];
            tq_ref::dequantize_vector_turbo4(v_blocks[h * n_tokens + t], v_vec);
            for (int d = 0; d < 128; d++)
                out[d] += weights[t] * v_vec[d];
        }
        std::memcpy(output + h * 128, out, 128 * sizeof(float));
    }
}

// Rotated-space attention (Path D): skip WHT in inner loop
// 1. Rotate Q with forward RHT
// 2. K scores = dot(q_rot, codebook_lookup(K)) * norm — no WHT per token
// 3. Softmax
// 4. V accum = Σ weight * norm * codebook_lookup(V) — no WHT per token
// 5. Inverse RHT on accumulated output — one WHT per head
inline void rotated_attention_turbo4(
    const float* queries,                  // [n_heads * 128]
    const block_turbo4_0* k_blocks,        // [n_heads * n_tokens]
    const block_turbo4_0* v_blocks,        // [n_heads * n_tokens]
    float* output,                         // [n_heads * 128]
    int n_heads, int n_tokens)
{
    for (int h = 0; h < n_heads; h++) {
        // Step 1: Forward RHT on query
        float q_rot[128];
        std::memcpy(q_rot, queries + h * 128, 128 * sizeof(float));
        tq_ref::forward_rht(q_rot);

        // Step 2: K scores in rotated space (no WHT per token)
        std::vector<float> scores(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            const auto& blk = k_blocks[h * n_tokens + t];
            uint8_t indices[128];
            tq_ref::unpack_turbo4(blk, indices);

            float dot = 0.0f;
            for (int d = 0; d < 128; d++)
                dot += q_rot[d] * TC4[indices[d]];

            float norm_val = fp16_to_fp32(blk.norm);
            scores[t] = dot * norm_val;
        }

        // Step 3: Softmax
        float max_score = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        std::vector<float> weights(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            weights[t] = std::exp(scores[t] - max_score);
            sum_exp += weights[t];
        }
        for (int t = 0; t < n_tokens; t++)
            weights[t] /= sum_exp;

        // Step 4: V accumulate in rotated space (no WHT per token)
        float acc[128] = {};
        for (int t = 0; t < n_tokens; t++) {
            const auto& blk = v_blocks[h * n_tokens + t];
            uint8_t indices[128];
            tq_ref::unpack_turbo4(blk, indices);

            float norm_val = fp16_to_fp32(blk.norm);
            float w = weights[t] * norm_val;
            for (int d = 0; d < 128; d++)
                acc[d] += w * TC4[indices[d]];
        }

        // Step 5: Inverse RHT on accumulated output
        tq_ref::inverse_rht(acc);
        std::memcpy(output + h * 128, acc, 128 * sizeof(float));
    }
}

// Turbo3 variants
inline void standard_attention_turbo3(
    const float* queries, const block_turbo3_0* k_blocks,
    const block_turbo3_0* v_blocks, float* output,
    int n_heads, int n_tokens)
{
    for (int h = 0; h < n_heads; h++) {
        std::vector<float> scores(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            float k_vec[128];
            tq_ref::dequantize_vector_turbo3(k_blocks[h * n_tokens + t], k_vec);
            float dot = 0.0f;
            for (int d = 0; d < 128; d++)
                dot += queries[h * 128 + d] * k_vec[d];
            scores[t] = dot;
        }

        float max_score = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        std::vector<float> weights(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            weights[t] = std::exp(scores[t] - max_score);
            sum_exp += weights[t];
        }
        for (int t = 0; t < n_tokens; t++)
            weights[t] /= sum_exp;

        float out[128] = {};
        for (int t = 0; t < n_tokens; t++) {
            float v_vec[128];
            tq_ref::dequantize_vector_turbo3(v_blocks[h * n_tokens + t], v_vec);
            for (int d = 0; d < 128; d++)
                out[d] += weights[t] * v_vec[d];
        }
        std::memcpy(output + h * 128, out, 128 * sizeof(float));
    }
}

inline void rotated_attention_turbo3(
    const float* queries, const block_turbo3_0* k_blocks,
    const block_turbo3_0* v_blocks, float* output,
    int n_heads, int n_tokens)
{
    for (int h = 0; h < n_heads; h++) {
        float q_rot[128];
        std::memcpy(q_rot, queries + h * 128, 128 * sizeof(float));
        tq_ref::forward_rht(q_rot);

        std::vector<float> scores(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            const auto& blk = k_blocks[h * n_tokens + t];
            uint8_t indices[128];
            tq_ref::unpack_turbo3(blk, indices);
            float dot = 0.0f;
            for (int d = 0; d < 128; d++)
                dot += q_rot[d] * TC3[indices[d]];
            float norm_val = fp16_to_fp32(blk.norm);
            scores[t] = dot * norm_val;
        }

        float max_score = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        std::vector<float> weights(n_tokens);
        for (int t = 0; t < n_tokens; t++) {
            weights[t] = std::exp(scores[t] - max_score);
            sum_exp += weights[t];
        }
        for (int t = 0; t < n_tokens; t++)
            weights[t] /= sum_exp;

        float acc[128] = {};
        for (int t = 0; t < n_tokens; t++) {
            const auto& blk = v_blocks[h * n_tokens + t];
            uint8_t indices[128];
            tq_ref::unpack_turbo3(blk, indices);
            float norm_val = fp16_to_fp32(blk.norm);
            float w = weights[t] * norm_val;
            for (int d = 0; d < 128; d++)
                acc[d] += w * TC3[indices[d]];
        }

        tq_ref::inverse_rht(acc);
        std::memcpy(output + h * 128, acc, 128 * sizeof(float));
    }
}

} // namespace tq_attn
