// =============================================================================
// pack_unit.cpp — Stage 5: Pack Unit Implementation
// =============================================================================

#include "pack_unit.h"
#include <limits>

// ═════════════════════════════════════════════════════════════════════════════
// SATURATION — Hardware: comparator checking overflow/underflow + mux
// ═════════════════════════════════════════════════════════════════════════════

int32_t PackUnit::saturate_signed(int64_t value, int bits) {
    // Compute signed range: [-(2^(bits-1)), 2^(bits-1) - 1]
    int64_t max_val = (1LL << (bits - 1)) - 1;
    int64_t min_val = -(1LL << (bits - 1));

    // Hardware: two comparators feeding a 3-input mux
    if (value > max_val) return static_cast<int32_t>(max_val);
    if (value < min_val) return static_cast<int32_t>(min_val);
    return static_cast<int32_t>(value);
}

uint32_t PackUnit::saturate_unsigned(uint64_t value, int bits) {
    uint64_t max_val = (1ULL << bits) - 1;
    if (value > max_val) return static_cast<uint32_t>(max_val);
    return static_cast<uint32_t>(value);
}

// ═════════════════════════════════════════════════════════════════════════════
// SCALING WITH ROUNDING — Hardware: barrel shifter + rounding adder
// ═════════════════════════════════════════════════════════════════════════════

int64_t PackUnit::scale_and_round(int64_t value, int shift_right, RoundMode rm) {
    if (shift_right <= 0) return value << (-shift_right);

    // Extract the discarded bits for rounding decision
    int64_t mask = (1LL << shift_right) - 1;
    int64_t discarded = value & mask;
    int64_t halfway = 1LL << (shift_right - 1);
    int64_t truncated = value >> shift_right;
    bool is_negative = (value < 0);

    bool increment = false;

    switch (rm) {
        case RoundMode::ROUND_NEAREST_EVEN:
            if (discarded > halfway) {
                increment = true;
            } else if (discarded == halfway) {
                // Ties to even: increment if truncated result is odd
                increment = (truncated & 1) != 0;
            }
            break;

        case RoundMode::ROUND_TOWARD_ZERO:
            // Never increment magnitude — but for negative numbers,
            // C++ arithmetic right shift rounds toward -inf, so adjust
            if (is_negative && discarded != 0) {
                increment = true;  // This "increments" toward zero for negatives
            }
            break;

        case RoundMode::ROUND_TOWARD_POS:
            if (!is_negative && discarded != 0) increment = true;
            break;

        case RoundMode::ROUND_TOWARD_NEG:
            // Arithmetic shift already rounds toward -inf, no adjustment for negatives
            if (is_negative && discarded != 0) {
                // Already rounded down — do nothing
            } else if (!is_negative) {
                // Truncate positive = correct floor behavior
            }
            break;
    }

    if (increment) truncated += 1;
    return truncated;
}

// ═════════════════════════════════════════════════════════════════════════════
// BIT-PACKING — Concatenate SIMD lanes to 32-bit output_z
// ═════════════════════════════════════════════════════════════════════════════

uint32_t PackUnit::pack_4bit_simd(const AccumulatorFile &acc, int scale_shift, RoundMode rm) {
    // 8 lanes: each 12-bit accumulator → scaled → saturated to 4 bits
    uint32_t result = 0;
    for (int i = 0; i < 8; ++i) {
        int64_t val = static_cast<int64_t>(static_cast<int16_t>(acc.read12(i)));
        val = scale_and_round(val, scale_shift, rm);
        uint32_t sat = saturate_unsigned(static_cast<uint64_t>(val & 0xF), 4);
        result |= (sat << (i * 4));
    }
    return result;
}

uint32_t PackUnit::pack_8bit_simd(const AccumulatorFile &acc, int scale_shift, RoundMode rm) {
    // 4 lanes: each 24-bit accumulator → scaled → saturated to 8 bits
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        // Sign-extend 24-bit value
        int32_t raw = static_cast<int32_t>(acc.read24(i));
        if (raw & 0x800000) raw |= 0xFF000000;  // sign extension
        int64_t val = static_cast<int64_t>(raw);
        val = scale_and_round(val, scale_shift, rm);
        int32_t sat = saturate_signed(val, 8);
        result |= ((static_cast<uint32_t>(sat) & 0xFF) << (i * 8));
    }
    return result;
}

uint32_t PackUnit::pack_16bit_simd(const AccumulatorFile &acc, int scale_shift, RoundMode rm) {
    // 2 lanes: each 48-bit accumulator → scaled → saturated to 16 bits
    uint32_t result = 0;
    for (int i = 0; i < 2; ++i) {
        int64_t val = static_cast<int64_t>(acc.read48(i));
        // Sign-extend 48-bit
        if (val & (1LL << 47)) val |= 0xFFFF000000000000LL;
        val = scale_and_round(val, scale_shift, rm);
        int32_t sat = saturate_signed(val, 16);
        result |= ((static_cast<uint32_t>(sat) & 0xFFFF) << (i * 16));
    }
    return result;
}

uint32_t PackUnit::pack_32bit(const AccumulatorFile &acc, int scale_shift, RoundMode rm) {
    // 1 lane: 96-bit accumulator → scaled → saturated to 32 bits
    uint64_t lo64;
    uint32_t hi32;
    acc.read96(lo64, hi32);

    // Construct a sign-extended 96-bit value, treat as signed
    // For simplicity, work with 64-bit precision (hardware would use full 96-bit)
    int64_t val;
    if (hi32 & 0x80000000u) {
        // Negative in 96-bit: approximate using upper bits
        val = static_cast<int64_t>(lo64);
        if (hi32 != 0xFFFFFFFFu) {
            // Very large magnitude: saturate
            return (hi32 & 0x80000000u) ? 0x80000000u : 0x7FFFFFFFu;
        }
    } else {
        val = static_cast<int64_t>(lo64);
        if (hi32 != 0) {
            return 0x7FFFFFFFu;  // Positive overflow
        }
    }

    val = scale_and_round(val, scale_shift, rm);
    return static_cast<uint32_t>(saturate_signed(val, 32));
}

uint32_t PackUnit::pack_fp16_dual(uint16_t hi, uint16_t lo) {
    return (static_cast<uint32_t>(hi) << 16) | lo;
}

// ═════════════════════════════════════════════════════════════════════════════
// GENERIC DISPATCHER — Mux selecting pack path based on CFG_REG precision
// ═════════════════════════════════════════════════════════════════════════════

uint32_t PackUnit::pack(const AccumulatorFile &acc, Precision prec, Mode mode,
                        int scale_shift, RoundMode rm) {
    // Float mode packing is handled separately in the top-level module
    // (since float results go through the SoftFloatEngine pack path).
    // This dispatcher handles integer and complex integer modes.

    switch (prec) {
        case Precision::P4:  return pack_4bit_simd(acc, scale_shift, rm);
        case Precision::P8:  return pack_8bit_simd(acc, scale_shift, rm);
        case Precision::P16: return pack_16bit_simd(acc, scale_shift, rm);
        case Precision::P32: return pack_32bit(acc, scale_shift, rm);
        default:             return 0;
    }
}
