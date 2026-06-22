#pragma once
#include <cstdint>
#include <cstddef>

// Block size for both turbo3 and turbo4 = head_dim = 128
static constexpr int QK_TURBO3 = 128;
static constexpr int QK_TURBO4 = 128;

// TurboQuant 3-bit block: 50 bytes per 128 values (3.125 bits/value)
// Layout: norm(fp16) + qs[32] (lower 2 bits, 4 per byte) + signs[16] (upper 1 bit, 8 per byte)
struct alignas(2) block_turbo3_0 {
    uint16_t norm;                        //  2 bytes: FP16 L2 norm
    uint8_t  qs[QK_TURBO3 / 4];          // 32 bytes: lower 2 bits of 3-bit index
    uint8_t  signs[QK_TURBO3 / 8];       // 16 bytes: upper 1 bit of 3-bit index
};
static_assert(sizeof(block_turbo3_0) == 2 + QK_TURBO3/4 + QK_TURBO3/8,
              "wrong turbo3_0 block size");
// = 50 bytes

// TurboQuant 4-bit block: 68 bytes per 128 values (4.25 bits/value)
// Layout: norm(fp16) + rnorm(fp16, reserved) + qs[64] (4-bit indices, 2 per byte)
struct alignas(2) block_turbo4_0 {
    uint16_t norm;                        //  2 bytes: FP16 L2 norm
    uint16_t rnorm;                       //  2 bytes: reserved (unused in 4-bit mode)
    uint8_t  qs[QK_TURBO4 / 2];          // 64 bytes: 4-bit nibble-packed indices
};
static_assert(sizeof(block_turbo4_0) == 68, "wrong turbo4_0 block size");

// Helper: FP16 <-> FP32 conversion (using hardware intrinsics where available)
inline float fp16_to_fp32(uint16_t h) {
    // IEEE 754 half-precision to single-precision
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            uint32_t result = sign;
            float f;
            __builtin_memcpy(&f, &result, 4);
            return f;
        }
        // Subnormal
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        // Inf/NaN
        uint32_t result = sign | 0x7F800000 | ((uint32_t)mant << 13);
        float f;
        __builtin_memcpy(&f, &result, 4);
        return f;
    }

    uint32_t result = sign | ((uint32_t)(exp + 112) << 23) | ((uint32_t)mant << 13);
    float f;
    __builtin_memcpy(&f, &result, 4);
    return f;
}

inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);

    uint16_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return sign;
        mant = (mant | 0x800000) >> (1 - exp);
        return sign | (uint16_t)(mant >> 13);
    } else if (exp >= 31) {
        return sign | 0x7C00; // Inf
    }
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}
