// =============================================================================
// soft_float_engine.h — Bit-Accurate Soft-Float Arithmetic Engine
// =============================================================================
// Hardware Mapping: This class models the unified arithmetic core (Stage 2).
// All floating-point math is performed using ONLY integer operations, exactly
// mimicking the hardware datapath's sign/exponent/mantissa pipelines.
//
// ⛔ NO native float/double is used for ANY arithmetic. Conversion helpers
//    (float_to_bits / bits_to_float) exist only for test stimulus generation.
// =============================================================================
#pragma once

#include "mprcu_types.h"

class SoftFloatEngine {
public:
    // ─────────────────────────────────────────────────────────────────────
    // FP32 IEEE-754 Constants (matching hardware parameters)
    // ─────────────────────────────────────────────────────────────────────
    static constexpr int FP32_EXP_BITS  = 8;
    static constexpr int FP32_MANT_BITS = 23;
    static constexpr int FP32_BIAS      = 127;
    static constexpr uint32_t FP32_EXP_MASK  = 0x7F800000u;
    static constexpr uint32_t FP32_MANT_MASK = 0x007FFFFFu;
    static constexpr uint32_t FP32_SIGN_MASK = 0x80000000u;
    static constexpr uint32_t FP32_QNAN      = 0x7FC00000u;
    static constexpr uint32_t FP32_POS_INF   = 0x7F800000u;
    static constexpr uint32_t FP32_NEG_INF   = 0xFF800000u;

    // ─────────────────────────────────────────────────────────────────────
    // FP16 IEEE-754 Constants
    // ─────────────────────────────────────────────────────────────────────
    static constexpr int FP16_EXP_BITS  = 5;
    static constexpr int FP16_MANT_BITS = 10;
    static constexpr int FP16_BIAS      = 15;
    static constexpr uint16_t FP16_EXP_MASK  = 0x7C00u;
    static constexpr uint16_t FP16_MANT_MASK = 0x03FFu;
    static constexpr uint16_t FP16_SIGN_MASK = 0x8000u;
    static constexpr uint16_t FP16_QNAN      = 0x7E00u;
    static constexpr uint16_t FP16_POS_INF   = 0x7C00u;

    // =====================================================================
    // Stage 1: Unpack — maps to hardware "Input Formatter"
    // =====================================================================
    static FPUnpacked unpack_fp32(uint32_t bits);
    static FPUnpacked unpack_fp16(uint16_t bits);

    // =====================================================================
    // Stage 5 (partial): Pack — maps to hardware "Output Formatter" for FP
    // =====================================================================
    static uint32_t pack_fp32(const FPUnpacked &fp, RoundMode rm);
    static uint16_t pack_fp16(const FPUnpacked &fp, RoundMode rm);

    // =====================================================================
    // Stage 2: Core Arithmetic — all via integer manipulation
    // =====================================================================

    /// FP32 Addition/Subtraction (sub = negate B's sign, then add)
    /// Hardware: exponent comparator → barrel shifter → integer adder → normalizer
    static uint32_t fp32_add(uint32_t a, uint32_t b, RoundMode rm);
    static uint32_t fp32_sub(uint32_t a, uint32_t b, RoundMode rm);

    /// FP32 Multiplication
    /// Hardware: sign XOR → exponent adder → mantissa multiplier (reuses
    ///           the 64× 4-bit integer multiplier array) → normalizer
    static uint32_t fp32_mul(uint32_t a, uint32_t b, RoundMode rm);

    /// FP16 counterparts
    static uint16_t fp16_add(uint16_t a, uint16_t b, RoundMode rm);
    static uint16_t fp16_sub(uint16_t a, uint16_t b, RoundMode rm);
    static uint16_t fp16_mul(uint16_t a, uint16_t b, RoundMode rm);

    /// FP32 Fused Multiply-Accumulate: result = acc + (a * b)
    /// The intermediate product uses extended precision (no rounding until final)
    static uint32_t fp32_fma(uint32_t a, uint32_t b, uint32_t acc, RoundMode rm);

    // =====================================================================
    // Integer SIMD operations — reusing the 64×4-bit multiplier concept
    // =====================================================================
    static uint32_t int_mul_simd_4b(uint32_t a, uint32_t b);   // 8× 4-bit
    static uint32_t int_mul_simd_8b(uint32_t a, uint32_t b);   // 4× 8-bit
    static uint32_t int_mul_simd_16b(uint32_t a, uint32_t b);  // 2× 16-bit
    static uint32_t int_mul_32b(uint32_t a, uint32_t b);       // 1× 32-bit

    static uint32_t int_add_simd_4b(uint32_t a, uint32_t b);
    static uint32_t int_add_simd_8b(uint32_t a, uint32_t b);
    static uint32_t int_add_simd_16b(uint32_t a, uint32_t b);
    static uint32_t int_add_32b(uint32_t a, uint32_t b);

    // =====================================================================
    // Complex number operations (FP16 real + FP16 imag)
    // =====================================================================
    static uint32_t complex_mul_fp16(uint32_t a, uint32_t b, RoundMode rm);

    // =====================================================================
    // Stage 4: Newton-Raphson for Division and Square Root
    // =====================================================================
    static uint32_t fp32_div_nr(uint32_t a, uint32_t b, RoundMode rm, int iterations);
    static uint32_t fp32_sqrt_nr(uint32_t a, RoundMode rm, int iterations);

    // =====================================================================
    // Precision conversion helpers
    // =====================================================================
    static uint32_t fp16_to_fp32(uint16_t h);
    static uint16_t fp32_to_fp16(uint32_t f, RoundMode rm);

private:
    // ─────────────────────────────────────────────────────────────────────
    // Internal: Mantissa multiplication via 4-bit decomposition
    // This mirrors how the hardware tiles the mantissa multiply across
    // the 64× 4-bit multiplier array with partial-product accumulation.
    // ─────────────────────────────────────────────────────────────────────
    static uint64_t mul_mantissa_via_4bit_array(uint32_t m_a, uint32_t m_b,
                                                 int a_bits, int b_bits);

    // Count leading zeros — maps to hardware CLZ unit
    static int clz32(uint32_t x);
    static int clz64(uint64_t x);

    // Rounding logic — direct model of the hardware rounding mux
    static uint64_t apply_rounding(uint64_t mantissa, int round_bit_pos,
                                   RoundMode rm, uint32_t sign);
};
