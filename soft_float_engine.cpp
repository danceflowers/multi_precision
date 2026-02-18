// =============================================================================
// soft_float_engine.cpp — Bit-Accurate Soft-Float Arithmetic Engine
// =============================================================================
// Every function models a hardware pipeline stage or sub-block. Comments note
// the corresponding RTL signal / mux / pipeline register at each step.
//
// Arithmetic flow (FP32 Multiply example):
//   [Unpack] → sign XOR, exp adder, mantissa to 4-bit array
//            → [partial products] → [accumulate] → [normalize] → [round] → [pack]
// =============================================================================

#include "soft_float_engine.h"
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// CLZ — Count Leading Zeros (maps to hardware CLZ combinational block)
// ─────────────────────────────────────────────────────────────────────────────
int SoftFloatEngine::clz32(uint32_t x) {
    if (x == 0) return 32;
    int n = 0;
    if ((x & 0xFFFF0000u) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF000000u) == 0) { n +=  8; x <<=  8; }
    if ((x & 0xF0000000u) == 0) { n +=  4; x <<=  4; }
    if ((x & 0xC0000000u) == 0) { n +=  2; x <<=  2; }
    if ((x & 0x80000000u) == 0) { n +=  1; }
    return n;
}

int SoftFloatEngine::clz64(uint64_t x) {
    if (x == 0) return 64;
    uint32_t hi = static_cast<uint32_t>(x >> 32);
    if (hi != 0) return clz32(hi);
    return 32 + clz32(static_cast<uint32_t>(x));
}

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 1: UNPACK — Input Formatter
// ═════════════════════════════════════════════════════════════════════════════

FPUnpacked SoftFloatEngine::unpack_fp32(uint32_t bits) {
    FPUnpacked fp;

    // --- Wire assignments (combinational) ---
    fp.sign     = (bits >> 31) & 1;
    uint32_t raw_exp  = (bits >> FP32_MANT_BITS) & 0xFF;
    uint32_t raw_mant = bits & FP32_MANT_MASK;

    // --- Special-case detection logic ---
    if (raw_exp == 0xFF) {
        // Exponent all-ones: Inf or NaN
        fp.is_inf = (raw_mant == 0);
        fp.is_nan = (raw_mant != 0);
        fp.exponent = 128;  // saturated
        fp.mantissa = raw_mant;
    } else if (raw_exp == 0) {
        if (raw_mant == 0) {
            // ±Zero
            fp.is_zero = true;
            fp.exponent = -FP32_BIAS;  // effective exponent for zero
            fp.mantissa = 0;
        } else {
            // Subnormal: no hidden-1, exponent is 1 - bias
            fp.is_subnormal = true;
            fp.exponent = 1 - FP32_BIAS;  // = -126
            fp.mantissa = raw_mant;        // No hidden-1 insertion
        }
    } else {
        // Normal: insert hidden-1 at bit 23
        fp.exponent = static_cast<int32_t>(raw_exp) - FP32_BIAS;
        fp.mantissa = (1u << FP32_MANT_BITS) | raw_mant;  // 1.fraction form
    }
    return fp;
}

FPUnpacked SoftFloatEngine::unpack_fp16(uint16_t bits) {
    FPUnpacked fp;
    fp.sign     = (bits >> 15) & 1;
    uint32_t raw_exp  = (bits >> FP16_MANT_BITS) & 0x1F;
    uint32_t raw_mant = bits & FP16_MANT_MASK;

    if (raw_exp == 0x1F) {
        fp.is_inf = (raw_mant == 0);
        fp.is_nan = (raw_mant != 0);
        fp.exponent = 16;
        fp.mantissa = raw_mant;
    } else if (raw_exp == 0) {
        if (raw_mant == 0) {
            fp.is_zero = true;
            fp.exponent = -FP16_BIAS;
            fp.mantissa = 0;
        } else {
            fp.is_subnormal = true;
            fp.exponent = 1 - FP16_BIAS;  // = -14
            fp.mantissa = raw_mant;
        }
    } else {
        fp.exponent = static_cast<int32_t>(raw_exp) - FP16_BIAS;
        fp.mantissa = (1u << FP16_MANT_BITS) | raw_mant;
    }
    return fp;
}

// ═════════════════════════════════════════════════════════════════════════════
// ROUNDING LOGIC — Direct model of hardware rounding mux
// ═════════════════════════════════════════════════════════════════════════════
// Inputs:
//   mantissa       — extended mantissa with guard/round/sticky info
//   round_bit_pos  — bit position below which we truncate
//   rm             — rounding mode from CFG_REG
//   sign           — sign bit (needed for directed rounding modes)
// Output:
//   Rounded mantissa (shifted right by round_bit_pos)
// ═════════════════════════════════════════════════════════════════════════════

uint64_t SoftFloatEngine::apply_rounding(uint64_t mantissa, int round_bit_pos,
                                          RoundMode rm, uint32_t sign) {
    if (round_bit_pos <= 0) return mantissa;

    // Extract Guard, Round, Sticky bits — hardware parallel logic
    uint64_t guard_bit   = (mantissa >> (round_bit_pos - 1)) & 1;
    uint64_t round_bit   = (round_bit_pos >= 2) ? ((mantissa >> (round_bit_pos - 2)) & 1) : 0;
    uint64_t sticky_bits = (round_bit_pos >= 2) ? (mantissa & ((1ULL << (round_bit_pos - 2)) - 1)) : 0;
    // Sticky OR with round bit for combined sticky
    uint64_t sticky = (round_bit | sticky_bits) ? 1 : 0;

    uint64_t truncated = mantissa >> round_bit_pos;
    bool increment = false;

    switch (rm) {
        case RoundMode::ROUND_NEAREST_EVEN:
            // Round to nearest, ties to even: increment if guard=1 AND (sticky!=0 OR LSB=1)
            if (guard_bit && (sticky || (truncated & 1)))
                increment = true;
            break;

        case RoundMode::ROUND_TOWARD_ZERO:
            // Truncate — never increment
            break;

        case RoundMode::ROUND_TOWARD_POS:
            // Increment if positive AND any discarded bits are non-zero
            if (!sign && (guard_bit || sticky))
                increment = true;
            break;

        case RoundMode::ROUND_TOWARD_NEG:
            // Increment magnitude if negative AND any discarded bits are non-zero
            if (sign && (guard_bit || sticky))
                increment = true;
            break;
    }

    if (increment) truncated += 1;
    return truncated;
}

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 5 (partial): PACK — maps to Output Formatter for floating-point
// ═════════════════════════════════════════════════════════════════════════════

uint32_t SoftFloatEngine::pack_fp32(const FPUnpacked &fp, RoundMode rm) {
    // --- Special case mux (combinational) ---
    if (fp.is_nan) return FP32_QNAN;
    if (fp.is_inf) return (fp.sign ? FP32_NEG_INF : FP32_POS_INF);
    if (fp.is_zero && fp.mantissa == 0) {
        return fp.sign << 31;  // ±0
    }

    int32_t  exp  = fp.exponent;
    uint64_t mant = fp.mantissa;

    // --- Normalize: shift mantissa so that hidden-1 is at bit 23 ---
    // (models the hardware normalizer — CLZ + barrel shifter)
    if (mant != 0) {
        // Find position of the leading 1
        int leading = 63 - clz64(mant);
        int shift = leading - FP32_MANT_BITS;
        exp += shift;

        if (shift > 0) {
            // Need to shift right and round
            mant = apply_rounding(mant, shift, rm, fp.sign);
            // Check if rounding caused overflow (e.g., mantissa became 1.0...0 → 10.0...0)
            if (mant & (1ULL << (FP32_MANT_BITS + 1))) {
                mant >>= 1;
                exp += 1;
            }
        } else if (shift < 0) {
            mant <<= (-shift);
        }
    }

    // --- Overflow / Underflow check (comparator) ---
    int32_t biased_exp = exp + FP32_BIAS;

    if (biased_exp >= 0xFF) {
        // Overflow → Infinity
        return (fp.sign << 31) | FP32_EXP_MASK;
    }
    if (biased_exp <= 0) {
        // Subnormal or underflow
        int right_shift = 1 - biased_exp;
        if (right_shift >= 64) return (fp.sign << 31);  // total underflow to zero

        mant = apply_rounding(mant, right_shift, rm, fp.sign);
        if (mant & (1ULL << FP32_MANT_BITS)) {
            // Rounding promoted back to normal range
            return (fp.sign << 31) | (1u << FP32_MANT_BITS) | (static_cast<uint32_t>(mant) & FP32_MANT_MASK);
        }
        return (fp.sign << 31) | (static_cast<uint32_t>(mant) & FP32_MANT_MASK);
    }

    // --- Normal pack: concatenate sign | exp | mantissa ---
    uint32_t result = (fp.sign << 31)
                    | (static_cast<uint32_t>(biased_exp) << FP32_MANT_BITS)
                    | (static_cast<uint32_t>(mant) & FP32_MANT_MASK);
    return result;
}

uint16_t SoftFloatEngine::pack_fp16(const FPUnpacked &fp, RoundMode rm) {
    if (fp.is_nan) return FP16_QNAN;
    if (fp.is_inf) return static_cast<uint16_t>((fp.sign << 15) | FP16_POS_INF);
    if (fp.is_zero && fp.mantissa == 0) {
        return static_cast<uint16_t>(fp.sign << 15);
    }

    int32_t  exp  = fp.exponent;
    uint64_t mant = fp.mantissa;

    if (mant != 0) {
        int leading = 63 - clz64(mant);
        int shift = leading - FP16_MANT_BITS;
        exp += shift;
        if (shift > 0) {
            mant = apply_rounding(mant, shift, rm, fp.sign);
            if (mant & (1ULL << (FP16_MANT_BITS + 1))) {
                mant >>= 1;
                exp += 1;
            }
        } else if (shift < 0) {
            mant <<= (-shift);
        }
    }

    int32_t biased_exp = exp + FP16_BIAS;
    if (biased_exp >= 0x1F) {
        return static_cast<uint16_t>((fp.sign << 15) | FP16_POS_INF);
    }
    if (biased_exp <= 0) {
        int right_shift = 1 - biased_exp;
        if (right_shift >= 64) return static_cast<uint16_t>(fp.sign << 15);
        mant = apply_rounding(mant, right_shift, rm, fp.sign);
        return static_cast<uint16_t>((fp.sign << 15) | (static_cast<uint16_t>(mant) & FP16_MANT_MASK));
    }

    return static_cast<uint16_t>((fp.sign << 15)
                                | (static_cast<uint16_t>(biased_exp) << FP16_MANT_BITS)
                                | (static_cast<uint16_t>(mant) & FP16_MANT_MASK));
}

// ═════════════════════════════════════════════════════════════════════════════
// MANTISSA MULTIPLICATION VIA 4-BIT ARRAY — Hardware Resource Sharing Model
// ═════════════════════════════════════════════════════════════════════════════
// The hardware has 64× 4-bit unsigned multipliers that are stitched together
// to form wider multiplications. This function decomposes the mantissa
// multiply into 4-bit × 4-bit partial products and accumulates them,
// exactly as the hardware would.
//
// For a 24×24 FP32 mantissa multiply:
//   24 bits = 6 × 4-bit nibbles per operand → 36 partial products
//   Each partial product: 8 bits, shifted by (i+j)*4 bit positions
//   Total: accumulated into a 48-bit result
// ═════════════════════════════════════════════════════════════════════════════

uint64_t SoftFloatEngine::mul_mantissa_via_4bit_array(uint32_t m_a, uint32_t m_b,
                                                       int a_bits, int b_bits) {
    int nibbles_a = (a_bits + 3) / 4;
    int nibbles_b = (b_bits + 3) / 4;

    uint64_t accumulator = 0;

    // --- Partial product generation and accumulation ---
    // Hardware: each 4-bit multiplier produces an 8-bit result,
    // which is then shifted and added to the accumulator tree.
    for (int i = 0; i < nibbles_a; ++i) {
        uint32_t nibble_a = (m_a >> (i * 4)) & 0xF;
        for (int j = 0; j < nibbles_b; ++j) {
            uint32_t nibble_b = (m_b >> (j * 4)) & 0xF;

            // 4-bit × 4-bit = 8-bit partial product (single multiplier cell)
            uint64_t partial = static_cast<uint64_t>(nibble_a) * static_cast<uint64_t>(nibble_b);

            // Shift by combined nibble position (Wallace tree position)
            int shift = (i + j) * 4;
            accumulator += partial << shift;
        }
    }
    return accumulator;
}

// ═════════════════════════════════════════════════════════════════════════════
// FP32 ADDITION — Bit-Accurate Pipeline Model
// ═════════════════════════════════════════════════════════════════════════════
// Pipeline: Unpack → Exponent Compare → Barrel Shift → Integer Add →
//           Normalize (CLZ + Shift) → Round → Pack
// ═════════════════════════════════════════════════════════════════════════════

uint32_t SoftFloatEngine::fp32_add(uint32_t a, uint32_t b, RoundMode rm) {
    // ── Stage 1: Unpack ──
    FPUnpacked fa = unpack_fp32(a);
    FPUnpacked fb = unpack_fp32(b);

    // ── Special case logic (combinational mux) ──
    if (fa.is_nan || fb.is_nan) return FP32_QNAN;
    if (fa.is_inf && fb.is_inf) {
        // +Inf + -Inf = NaN
        if (fa.sign != fb.sign) return FP32_QNAN;
        return a;  // same sign inf
    }
    if (fa.is_inf) return a;
    if (fb.is_inf) return b;
    if (fa.is_zero && fb.is_zero) {
        // -0 + -0 = -0, otherwise +0
        uint32_t result_sign = (rm == RoundMode::ROUND_TOWARD_NEG)
                             ? (fa.sign | fb.sign)
                             : (fa.sign & fb.sign);
        return result_sign << 31;
    }
    if (fa.is_zero) return b;
    if (fb.is_zero) return a;

    // ── Stage 2: Prepare mantissas with 3 guard bits (G, R, S positions) ──
    // We work with 27-bit mantissas: 24 bits + 3 guard bits
    const int GUARD_BITS = 3;
    uint64_t mant_a = static_cast<uint64_t>(fa.mantissa) << GUARD_BITS;
    uint64_t mant_b = static_cast<uint64_t>(fb.mantissa) << GUARD_BITS;
    int32_t  exp_a  = fa.exponent;
    int32_t  exp_b  = fb.exponent;

    // ── Exponent comparison (hardware: subtractor) ──
    int32_t exp_diff = exp_a - exp_b;
    int32_t result_exp;

    // ── Alignment barrel shift ──
    // The smaller operand is right-shifted; shifted-out bits become sticky
    if (exp_diff > 0) {
        // B is smaller, shift B right
        result_exp = exp_a;
        if (exp_diff < 64) {
            uint64_t sticky = (mant_b & ((1ULL << exp_diff) - 1)) ? 1 : 0;
            mant_b = (mant_b >> exp_diff) | sticky;
        } else {
            mant_b = (mant_b != 0) ? 1 : 0;  // collapsed to sticky
        }
    } else if (exp_diff < 0) {
        result_exp = exp_b;
        int32_t shift = -exp_diff;
        if (shift < 64) {
            uint64_t sticky = (mant_a & ((1ULL << shift) - 1)) ? 1 : 0;
            mant_a = (mant_a >> shift) | sticky;
        } else {
            mant_a = (mant_a != 0) ? 1 : 0;
        }
    } else {
        result_exp = exp_a;
    }

    // ── Integer Addition / Subtraction ──
    uint64_t result_mant;
    uint32_t result_sign;

    if (fa.sign == fb.sign) {
        // Same sign: unsigned add
        result_mant = mant_a + mant_b;
        result_sign = fa.sign;
    } else {
        // Different signs: effective subtraction
        if (mant_a >= mant_b) {
            result_mant = mant_a - mant_b;
            result_sign = fa.sign;
        } else {
            result_mant = mant_b - mant_a;
            result_sign = fb.sign;
        }
    }

    // ── Check for zero result ──
    if (result_mant == 0) {
        uint32_t zero_sign = (rm == RoundMode::ROUND_TOWARD_NEG) ? 1 : 0;
        return zero_sign << 31;
    }

    // ── Normalization (CLZ → barrel shift) ──
    // Find the leading 1 position
    int leading = 63 - clz64(result_mant);
    int target  = FP32_MANT_BITS + GUARD_BITS;  // bit 26

    if (leading > target) {
        // Overflow: shift right, collect sticky
        int rshift = leading - target;
        uint64_t sticky = (result_mant & ((1ULL << rshift) - 1)) ? 1 : 0;
        result_mant = (result_mant >> rshift) | sticky;
        result_exp += rshift;
    } else if (leading < target) {
        // Underflow: shift left (cancellation in subtraction)
        int lshift = target - leading;
        result_mant <<= lshift;
        result_exp -= lshift;
    }

    // ── Rounding ──
    result_mant = apply_rounding(result_mant, GUARD_BITS, rm, result_sign);

    // Check for carry-out from rounding
    if (result_mant & (1ULL << (FP32_MANT_BITS + 1))) {
        result_mant >>= 1;
        result_exp += 1;
    }

    // ── Pack result ──
    FPUnpacked result;
    result.sign     = result_sign;
    result.exponent = result_exp;
    result.mantissa = result_mant;
    return pack_fp32(result, rm);
}

uint32_t SoftFloatEngine::fp32_sub(uint32_t a, uint32_t b, RoundMode rm) {
    // Subtraction = addition with B's sign flipped (hardware: XOR gate on sign bit)
    return fp32_add(a, b ^ FP32_SIGN_MASK, rm);
}

// ═════════════════════════════════════════════════════════════════════════════
// FP32 MULTIPLICATION — Bit-Accurate Pipeline Model
// ═════════════════════════════════════════════════════════════════════════════
// Pipeline: Unpack → Sign XOR → Exponent Add → Mantissa Mul (4-bit array)
//           → Normalize → Round → Pack
// ═════════════════════════════════════════════════════════════════════════════

uint32_t SoftFloatEngine::fp32_mul(uint32_t a, uint32_t b, RoundMode rm) {
    FPUnpacked fa = unpack_fp32(a);
    FPUnpacked fb = unpack_fp32(b);

    // ── Sign: XOR gate ──
    uint32_t result_sign = fa.sign ^ fb.sign;

    // ── Special cases (priority mux) ──
    if (fa.is_nan || fb.is_nan) return FP32_QNAN;
    if (fa.is_inf) {
        if (fb.is_zero) return FP32_QNAN;  // Inf × 0 = NaN
        FPUnpacked r; r.is_inf = true; r.sign = result_sign;
        return pack_fp32(r, rm);
    }
    if (fb.is_inf) {
        if (fa.is_zero) return FP32_QNAN;  // 0 × Inf = NaN
        FPUnpacked r; r.is_inf = true; r.sign = result_sign;
        return pack_fp32(r, rm);
    }
    if (fa.is_zero || fb.is_zero) {
        return result_sign << 31;
    }

    // ── Exponent: adder ──
    int32_t result_exp = fa.exponent + fb.exponent;

    // ── Mantissa multiplication via 4-bit array ──
    // fa.mantissa is 24 bits (with hidden-1), fb.mantissa likewise
    // Product is up to 48 bits
    uint64_t product = mul_mantissa_via_4bit_array(
        static_cast<uint32_t>(fa.mantissa),
        static_cast<uint32_t>(fb.mantissa),
        FP32_MANT_BITS + 1,  // 24 bits including hidden-1
        FP32_MANT_BITS + 1
    );

    // ── Normalization ──
    // Product of two 1.xxxx numbers is either 1x.xxxx or 01.xxxx in [1, 4)
    // The hidden-1 positions: bit 23 × bit 23 → product bit 46 is the 2× carry
    // We need to normalize to bit 23 for mantissa
    int leading = 63 - clz64(product);
    int target  = FP32_MANT_BITS;  // bit 23

    // Shift amount to place leading-1 at bit 23 + some guard bits
    // We'll keep extra bits for rounding
    const int EXTRA_BITS = 3;  // guard, round, sticky
    int shift = leading - (target + EXTRA_BITS);

    if (shift > 0) {
        uint64_t sticky = (product & ((1ULL << shift) - 1)) ? 1 : 0;
        product = (product >> shift) | sticky;
        result_exp += (leading - 2 * FP32_MANT_BITS);
    } else if (shift < 0) {
        product <<= (-shift);
        result_exp += (leading - 2 * FP32_MANT_BITS);
    } else {
        result_exp += (leading - 2 * FP32_MANT_BITS);
    }

    // ── Rounding ──
    product = apply_rounding(product, EXTRA_BITS, rm, result_sign);

    // Carry-out check
    if (product & (1ULL << (FP32_MANT_BITS + 1))) {
        product >>= 1;
        result_exp += 1;
    }

    // ── Pack ──
    FPUnpacked result;
    result.sign     = result_sign;
    result.exponent = result_exp;
    result.mantissa = product;
    return pack_fp32(result, rm);
}

// ═════════════════════════════════════════════════════════════════════════════
// FP32 FMA — Fused Multiply-Accumulate
// ═════════════════════════════════════════════════════════════════════════════
// result = acc + (a × b), with single rounding at the end.
// The intermediate product keeps full precision (48 bits from mantissa mul).
// ═════════════════════════════════════════════════════════════════════════════

uint32_t SoftFloatEngine::fp32_fma(uint32_t a, uint32_t b, uint32_t acc, RoundMode rm) {
    FPUnpacked fa = unpack_fp32(a);
    FPUnpacked fb = unpack_fp32(b);
    FPUnpacked fc = unpack_fp32(acc);

    uint32_t product_sign = fa.sign ^ fb.sign;

    // ── Special cases ──
    if (fa.is_nan || fb.is_nan || fc.is_nan) return FP32_QNAN;
    if ((fa.is_inf && fb.is_zero) || (fb.is_inf && fa.is_zero)) return FP32_QNAN;
    if (fa.is_inf || fb.is_inf) {
        if (fc.is_inf && (product_sign != fc.sign)) return FP32_QNAN;
        FPUnpacked r; r.is_inf = true; r.sign = product_sign;
        return pack_fp32(r, rm);
    }
    if (fc.is_inf) {
        return acc;
    }

    // ── Compute product mantissa (full precision) ──
    if (fa.is_zero || fb.is_zero) {
        // product is zero, result = acc
        if (fc.is_zero) {
            uint32_t s = (rm == RoundMode::ROUND_TOWARD_NEG) ? (product_sign | fc.sign) : (product_sign & fc.sign);
            return s << 31;
        }
        return acc;
    }

    int32_t product_exp = fa.exponent + fb.exponent;
    uint64_t product_mant = mul_mantissa_via_4bit_array(
        static_cast<uint32_t>(fa.mantissa),
        static_cast<uint32_t>(fb.mantissa),
        FP32_MANT_BITS + 1,
        FP32_MANT_BITS + 1
    );

    // The product mantissa is at position (2 * FP32_MANT_BITS) for the hidden-1.
    // To correctly track the exponent, the implicit point is at bit (2 * MANT_BITS).
    // product value = product_mant * 2^(product_exp - 2*FP32_MANT_BITS)

    // Normalize: place leading-1 at a known working position (bit 47)
    int prod_leading = 63 - clz64(product_mant);
    // Adjust exponent to maintain value invariant:
    // value = mant * 2^(exp - implicit_point). Shifting mant by N means we
    // adjust exp so that the value is preserved: new_exp = exp + N (right shift)
    // We want leading-1 at bit 47 (our working position).
    // implicit_point tracks with the mantissa shift.
    int32_t product_implicit_pt = 2 * FP32_MANT_BITS;  // = 46 initially
    int prod_target = 47;
    if (prod_leading != prod_target) {
        int shift = prod_leading - prod_target;  // positive = shift right
        if (shift > 0) {
            product_mant >>= shift;
        } else {
            product_mant <<= (-shift);
        }
        product_implicit_pt -= shift;  // shift left → implicit_pt increases
    }

    if (fc.is_zero) {
        // No accumulate, just round and pack the product
        // Bring back to FP32 format: leading-1 at bit 23 + GUARD
        const int GUARD = 3;
        int target2 = FP32_MANT_BITS + GUARD;
        int leading2 = 63 - clz64(product_mant);  // should be 47
        int norm_shift = leading2 - target2;       // shift right to reach target
        // Compute the final exponent: value = mant * 2^(exp - implicit_pt)
        // After shifting mant right by norm_shift: new_mant = mant >> norm_shift
        // new value = (mant >> norm_shift) * 2^(exp - (implicit_pt - norm_shift))
        // We need pack_fp32 to interpret mantissa with hidden-1 at bit 23.
        // pack_fp32 interprets: value = mant * 2^(exponent)  with hidden-1 at bit 23
        // So: exponent = exp - implicit_pt + norm_shift + 0  ... let me be more careful.
        // Real value = product_mant * 2^(product_exp - product_implicit_pt)
        // After >> norm_shift: new_mant * 2^(product_exp - product_implicit_pt + norm_shift)
        // pack_fp32 expects: mant with hidden-1 at bit 23, and exponent where
        //   value = mant/2^23 * 2^exponent = mant * 2^(exponent - 23)
        // So: product_exp - product_implicit_pt + norm_shift = final_exp - 23
        //     final_exp = product_exp - product_implicit_pt + norm_shift + 23
        int32_t final_exp = product_exp - product_implicit_pt + norm_shift + GUARD + FP32_MANT_BITS;
        // But we also need target2 = 23 + GUARD, and we shift by norm_shift = leading2 - target2
        // Actually, after shift, leading-1 is at bit target2 = 26.
        // For rounding, we want hidden-1 at bit 26 (23 + 3 guard bits).
        // After rounding away 3 bits, hidden-1 is at bit 23. That's what pack_fp32 expects.

        if (norm_shift > 0) {
            uint64_t sticky = (product_mant & ((1ULL << norm_shift) - 1)) ? 1 : 0;
            product_mant = (product_mant >> norm_shift) | sticky;
        } else if (norm_shift < 0) {
            product_mant <<= (-norm_shift);
        }

        product_mant = apply_rounding(product_mant, GUARD, rm, product_sign);
        if (product_mant & (1ULL << (FP32_MANT_BITS + 1))) { product_mant >>= 1; final_exp++; }
        FPUnpacked result; result.sign = product_sign; result.exponent = final_exp; result.mantissa = product_mant;
        return pack_fp32(result, rm);
    }

    // ── Add product to accumulator ──
    uint64_t acc_mant = fc.mantissa;
    int32_t acc_exp = fc.exponent;
    // acc value = acc_mant * 2^(acc_exp - 23) (hidden-1 at bit 23)
    // product value = product_mant * 2^(product_exp - product_implicit_pt)
    // We need to align them. Convert both to a common exponent base.
    // Use "effective exponent" = exp - implicit_pt
    int32_t product_eff_exp = product_exp - product_implicit_pt;
    int32_t acc_eff_exp = acc_exp - FP32_MANT_BITS;

    // Align: shift the one with larger eff_exp... no, shift the smaller one right.
    int32_t exp_diff = product_eff_exp - acc_eff_exp;
    int32_t result_eff_exp;
    if (exp_diff > 0) {
        result_eff_exp = product_eff_exp;
        if (exp_diff < 64) {
            uint64_t sticky = (acc_mant & ((1ULL << exp_diff) - 1)) ? 1 : 0;
            acc_mant = (acc_mant >> exp_diff) | sticky;
        } else {
            acc_mant = (acc_mant != 0) ? 1 : 0;
        }
    } else if (exp_diff < 0) {
        result_eff_exp = acc_eff_exp;
        int32_t s = -exp_diff;
        if (s < 64) {
            uint64_t sticky = (product_mant & ((1ULL << s) - 1)) ? 1 : 0;
            product_mant = (product_mant >> s) | sticky;
        } else {
            product_mant = (product_mant != 0) ? 1 : 0;
        }
    } else {
        result_eff_exp = product_eff_exp;
    }

    // Effective add/sub
    uint64_t result_mant;
    uint32_t result_sign;
    if (product_sign == fc.sign) {
        result_mant = product_mant + acc_mant;
        result_sign = product_sign;
    } else {
        if (product_mant >= acc_mant) {
            result_mant = product_mant - acc_mant;
            result_sign = product_sign;
        } else {
            result_mant = acc_mant - product_mant;
            result_sign = fc.sign;
        }
    }

    if (result_mant == 0) {
        uint32_t s = (rm == RoundMode::ROUND_TOWARD_NEG) ? 1 : 0;
        return s << 31;
    }

    // Normalize, round, pack
    // result value = result_mant * 2^result_eff_exp
    // We want to express as: mant_23 * 2^(final_exp - 23) for pack_fp32
    const int GUARD = 3;
    int leading = 63 - clz64(result_mant);
    int target = FP32_MANT_BITS + GUARD;  // 26
    int norm_shift = leading - target;     // positive = shift right
    int32_t final_exp = result_eff_exp + norm_shift + GUARD + FP32_MANT_BITS;

    if (norm_shift > 0) {
        uint64_t sticky = (result_mant & ((1ULL << norm_shift) - 1)) ? 1 : 0;
        result_mant = (result_mant >> norm_shift) | sticky;
    } else if (norm_shift < 0) {
        result_mant <<= (-norm_shift);
    }

    result_mant = apply_rounding(result_mant, GUARD, rm, result_sign);
    if (result_mant & (1ULL << (FP32_MANT_BITS + 1))) { result_mant >>= 1; final_exp++; }

    FPUnpacked result;
    result.sign = result_sign;
    result.exponent = final_exp;
    result.mantissa = result_mant;
    return pack_fp32(result, rm);
}

// ═════════════════════════════════════════════════════════════════════════════
// FP16 OPERATIONS
// ═════════════════════════════════════════════════════════════════════════════

uint16_t SoftFloatEngine::fp16_add(uint16_t a, uint16_t b, RoundMode rm) {
    FPUnpacked fa = unpack_fp16(a);
    FPUnpacked fb = unpack_fp16(b);

    if (fa.is_nan || fb.is_nan) return FP16_QNAN;
    if (fa.is_inf && fb.is_inf) {
        if (fa.sign != fb.sign) return FP16_QNAN;
        return a;
    }
    if (fa.is_inf) return a;
    if (fb.is_inf) return b;
    if (fa.is_zero && fb.is_zero) {
        uint32_t rs = (rm == RoundMode::ROUND_TOWARD_NEG) ? (fa.sign | fb.sign) : (fa.sign & fb.sign);
        return static_cast<uint16_t>(rs << 15);
    }
    if (fa.is_zero) return b;
    if (fb.is_zero) return a;

    const int GUARD_BITS = 3;
    uint64_t mant_a = static_cast<uint64_t>(fa.mantissa) << GUARD_BITS;
    uint64_t mant_b = static_cast<uint64_t>(fb.mantissa) << GUARD_BITS;
    int32_t exp_a = fa.exponent, exp_b = fb.exponent;
    int32_t exp_diff = exp_a - exp_b;
    int32_t result_exp;

    if (exp_diff > 0) {
        result_exp = exp_a;
        if (exp_diff < 64) {
            uint64_t sticky = (mant_b & ((1ULL << exp_diff) - 1)) ? 1 : 0;
            mant_b = (mant_b >> exp_diff) | sticky;
        } else { mant_b = mant_b ? 1 : 0; }
    } else if (exp_diff < 0) {
        result_exp = exp_b;
        int32_t s = -exp_diff;
        if (s < 64) {
            uint64_t sticky = (mant_a & ((1ULL << s) - 1)) ? 1 : 0;
            mant_a = (mant_a >> s) | sticky;
        } else { mant_a = mant_a ? 1 : 0; }
    } else { result_exp = exp_a; }

    uint64_t result_mant; uint32_t result_sign;
    if (fa.sign == fb.sign) { result_mant = mant_a + mant_b; result_sign = fa.sign; }
    else {
        if (mant_a >= mant_b) { result_mant = mant_a - mant_b; result_sign = fa.sign; }
        else { result_mant = mant_b - mant_a; result_sign = fb.sign; }
    }

    if (result_mant == 0) return static_cast<uint16_t>(((rm == RoundMode::ROUND_TOWARD_NEG) ? 1u : 0u) << 15);

    int leading = 63 - clz64(result_mant);
    int target = FP16_MANT_BITS + GUARD_BITS;
    if (leading > target) {
        int rs = leading - target;
        uint64_t sticky = (result_mant & ((1ULL << rs) - 1)) ? 1 : 0;
        result_mant = (result_mant >> rs) | sticky;
        result_exp += rs;
    } else if (leading < target) {
        result_mant <<= (target - leading);
        result_exp -= (target - leading);
    }

    result_mant = apply_rounding(result_mant, GUARD_BITS, rm, result_sign);
    if (result_mant & (1ULL << (FP16_MANT_BITS + 1))) { result_mant >>= 1; result_exp++; }

    FPUnpacked r; r.sign = result_sign; r.exponent = result_exp; r.mantissa = result_mant;
    return pack_fp16(r, rm);
}

uint16_t SoftFloatEngine::fp16_sub(uint16_t a, uint16_t b, RoundMode rm) {
    return fp16_add(a, b ^ FP16_SIGN_MASK, rm);
}

uint16_t SoftFloatEngine::fp16_mul(uint16_t a, uint16_t b, RoundMode rm) {
    FPUnpacked fa = unpack_fp16(a);
    FPUnpacked fb = unpack_fp16(b);
    uint32_t result_sign = fa.sign ^ fb.sign;

    if (fa.is_nan || fb.is_nan) return FP16_QNAN;
    if (fa.is_inf) {
        if (fb.is_zero) return FP16_QNAN;
        FPUnpacked r; r.is_inf = true; r.sign = result_sign; return pack_fp16(r, rm);
    }
    if (fb.is_inf) {
        if (fa.is_zero) return FP16_QNAN;
        FPUnpacked r; r.is_inf = true; r.sign = result_sign; return pack_fp16(r, rm);
    }
    if (fa.is_zero || fb.is_zero) return static_cast<uint16_t>(result_sign << 15);

    int32_t result_exp = fa.exponent + fb.exponent;
    uint64_t product = mul_mantissa_via_4bit_array(
        static_cast<uint32_t>(fa.mantissa), static_cast<uint32_t>(fb.mantissa),
        FP16_MANT_BITS + 1, FP16_MANT_BITS + 1);

    int leading = 63 - clz64(product);
    const int EXTRA = 3;
    int target = FP16_MANT_BITS + EXTRA;
    int shift = leading - target;

    if (shift > 0) {
        uint64_t sticky = (product & ((1ULL << shift) - 1)) ? 1 : 0;
        product = (product >> shift) | sticky;
    } else if (shift < 0) {
        product <<= (-shift);
    }
    result_exp += (leading - 2 * FP16_MANT_BITS);

    product = apply_rounding(product, EXTRA, rm, result_sign);
    if (product & (1ULL << (FP16_MANT_BITS + 1))) { product >>= 1; result_exp++; }

    FPUnpacked r; r.sign = result_sign; r.exponent = result_exp; r.mantissa = product;
    return pack_fp16(r, rm);
}

// ═════════════════════════════════════════════════════════════════════════════
// COMPLEX MULTIPLICATION (FP16 pair: [31:16]=Imag, [15:0]=Real)
// ═════════════════════════════════════════════════════════════════════════════
// Implements: (Ar + Ai·i)(Br + Bi·i) = (Ar·Br - Ai·Bi) + (Ar·Bi + Ai·Br)·i
// Uses 4 FP16 multiplications and 2 FP16 additions, reusing the same array.
// ═════════════════════════════════════════════════════════════════════════════

uint32_t SoftFloatEngine::complex_mul_fp16(uint32_t a, uint32_t b, RoundMode rm) {
    uint16_t ar = static_cast<uint16_t>(a & 0xFFFF);        // Real part A
    uint16_t ai = static_cast<uint16_t>((a >> 16) & 0xFFFF); // Imag part A
    uint16_t br = static_cast<uint16_t>(b & 0xFFFF);        // Real part B
    uint16_t bi = static_cast<uint16_t>((b >> 16) & 0xFFFF); // Imag part B

    // 4 multiplications (resource sharing: sequential reuse of multiplier array)
    uint16_t ar_br = fp16_mul(ar, br, rm);  // Ar × Br
    uint16_t ai_bi = fp16_mul(ai, bi, rm);  // Ai × Bi
    uint16_t ar_bi = fp16_mul(ar, bi, rm);  // Ar × Bi
    uint16_t ai_br = fp16_mul(ai, br, rm);  // Ai × Br

    // Real = Ar·Br - Ai·Bi
    uint16_t real_part = fp16_sub(ar_br, ai_bi, rm);
    // Imag = Ar·Bi + Ai·Br
    uint16_t imag_part = fp16_add(ar_bi, ai_br, rm);

    // Pack: [31:16] = Imag, [15:0] = Real
    return (static_cast<uint32_t>(imag_part) << 16) | real_part;
}

// ═════════════════════════════════════════════════════════════════════════════
// INTEGER SIMD OPERATIONS — Reusing the 64×4-bit Multiplier Concept
// ═════════════════════════════════════════════════════════════════════════════

uint32_t SoftFloatEngine::int_mul_simd_4b(uint32_t a, uint32_t b) {
    // 8 lanes of 4-bit × 4-bit = 4-bit (lower 4 bits of product, saturating)
    uint32_t result = 0;
    for (int i = 0; i < 8; ++i) {
        uint32_t na = (a >> (i * 4)) & 0xF;
        uint32_t nb = (b >> (i * 4)) & 0xF;
        uint32_t prod = na * nb;
        prod = (prod > 0xF) ? 0xF : prod;  // Saturate to 4-bit
        result |= (prod << (i * 4));
    }
    return result;
}

uint32_t SoftFloatEngine::int_mul_simd_8b(uint32_t a, uint32_t b) {
    // 4 lanes of 8-bit × 8-bit = 8-bit (lower 8 bits)
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t na = (a >> (i * 8)) & 0xFF;
        uint32_t nb = (b >> (i * 8)) & 0xFF;
        uint32_t prod = na * nb;
        prod &= 0xFF;  // Truncate to 8-bit
        result |= (prod << (i * 8));
    }
    return result;
}

uint32_t SoftFloatEngine::int_mul_simd_16b(uint32_t a, uint32_t b) {
    // 2 lanes of 16-bit × 16-bit = 16-bit (lower 16 bits)
    uint32_t result = 0;
    for (int i = 0; i < 2; ++i) {
        uint32_t na = (a >> (i * 16)) & 0xFFFF;
        uint32_t nb = (b >> (i * 16)) & 0xFFFF;
        uint32_t prod = na * nb;
        prod &= 0xFFFF;
        result |= (prod << (i * 16));
    }
    return result;
}

uint32_t SoftFloatEngine::int_mul_32b(uint32_t a, uint32_t b) {
    // Full 32-bit multiply (lower 32 bits of result)
    // Internally decomposed through the 4-bit array for hardware fidelity
    uint64_t full = mul_mantissa_via_4bit_array(a, b, 32, 32);
    return static_cast<uint32_t>(full);  // lower 32 bits
}

uint32_t SoftFloatEngine::int_add_simd_4b(uint32_t a, uint32_t b) {
    uint32_t result = 0;
    for (int i = 0; i < 8; ++i) {
        uint32_t na = (a >> (i * 4)) & 0xF;
        uint32_t nb = (b >> (i * 4)) & 0xF;
        uint32_t sum = (na + nb) & 0xF;  // Wrap within 4-bit lane
        result |= (sum << (i * 4));
    }
    return result;
}

uint32_t SoftFloatEngine::int_add_simd_8b(uint32_t a, uint32_t b) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t na = (a >> (i * 8)) & 0xFF;
        uint32_t nb = (b >> (i * 8)) & 0xFF;
        result |= (((na + nb) & 0xFF) << (i * 8));
    }
    return result;
}

uint32_t SoftFloatEngine::int_add_simd_16b(uint32_t a, uint32_t b) {
    uint32_t result = 0;
    for (int i = 0; i < 2; ++i) {
        uint32_t na = (a >> (i * 16)) & 0xFFFF;
        uint32_t nb = (b >> (i * 16)) & 0xFFFF;
        result |= (((na + nb) & 0xFFFF) << (i * 16));
    }
    return result;
}

uint32_t SoftFloatEngine::int_add_32b(uint32_t a, uint32_t b) {
    return a + b;
}

// ═════════════════════════════════════════════════════════════════════════════
// NEWTON-RAPHSON DIVISION — Stage 4 (Non-Linear Unit)
// ═════════════════════════════════════════════════════════════════════════════
// Computes a / b using: X_{n+1} = X_n × (2 - b × X_n)
// Initial guess via LUT, then `iterations` refinement steps.
// All intermediate arithmetic uses fp32_mul / fp32_sub.
// ═════════════════════════════════════════════════════════════════════════════

uint32_t SoftFloatEngine::fp32_div_nr(uint32_t a, uint32_t b, RoundMode rm, int iterations) {
    FPUnpacked fa = unpack_fp32(a);
    FPUnpacked fb = unpack_fp32(b);

    uint32_t result_sign_bit = ((a ^ b) & FP32_SIGN_MASK);

    // Special cases
    if (fa.is_nan || fb.is_nan) return FP32_QNAN;
    if (fa.is_inf && fb.is_inf) return FP32_QNAN;
    if (fa.is_inf) return result_sign_bit | FP32_POS_INF;
    if (fb.is_zero) {
        if (fa.is_zero) return FP32_QNAN;  // 0/0
        return result_sign_bit | FP32_POS_INF;  // x/0
    }
    if (fa.is_zero) return result_sign_bit;  // 0/x = 0
    if (fb.is_inf) return result_sign_bit;   // x/inf = 0

    // ── LUT: Initial guess for 1/b ──
    // Normalize b to [1, 2) range by extracting exponent
    // Initial guess: X0 ≈ 1/b using a piecewise-linear LUT
    // For simplicity, we use: X0 = 2^(1 - exp_b) * (48/17 - 32/17 * frac_b)
    // Approximated as: take b, force exponent to 126 (≈1/b in magnitude)
    uint32_t b_abs = b & ~FP32_SIGN_MASK;
    uint32_t b_exp = (b_abs >> 23) & 0xFF;

    // Create initial reciprocal estimate: flip exponent
    uint32_t est_exp = (2 * FP32_BIAS - 1) - b_exp;
    if (est_exp >= 0xFF) est_exp = 0xFE;  // Clamp to avoid overflow
    uint32_t x = (est_exp << 23) | (0x007FFFFFu - (b_abs & FP32_MANT_MASK));
    // Remove sign for iteration, apply at end
    x &= ~FP32_SIGN_MASK;
    uint32_t b_positive = b_abs;

    // ── Iterative refinement: X_{n+1} = X_n × (2 - b × X_n) ──
    // "2.0" in IEEE-754: 0x40000000
    const uint32_t TWO_FP32 = 0x40000000u;

    for (int i = 0; i < iterations; ++i) {
        uint32_t bx   = fp32_mul(b_positive, x, rm);    // b × X_n
        uint32_t diff = fp32_sub(TWO_FP32, bx, rm);     // 2 - b × X_n
        x = fp32_mul(x, diff, rm);                       // X_n × (2 - b × X_n)
    }

    // ── Final: a × (1/b) ──
    uint32_t a_positive = a & ~FP32_SIGN_MASK;
    uint32_t result = fp32_mul(a_positive, x, rm);

    // Apply combined sign
    result = (result & ~FP32_SIGN_MASK) | result_sign_bit;
    return result;
}

// ═════════════════════════════════════════════════════════════════════════════
// NEWTON-RAPHSON SQUARE ROOT — Stage 4
// ═════════════════════════════════════════════════════════════════════════════
// Computes sqrt(a) using inverse sqrt: Y_{n+1} = Y_n × (3 - a × Y_n²) / 2
// Then result = a × Y_final.
// ═════════════════════════════════════════════════════════════════════════════

uint32_t SoftFloatEngine::fp32_sqrt_nr(uint32_t a, RoundMode rm, int iterations) {
    FPUnpacked fa = unpack_fp32(a);

    // Special cases
    if (fa.is_nan) return FP32_QNAN;
    if (fa.sign && !fa.is_zero) return FP32_QNAN;  // sqrt(negative) = NaN
    if (fa.is_inf) return a;                         // sqrt(+Inf) = +Inf
    if (fa.is_zero) return a;                        // sqrt(±0) = ±0

    // ── LUT: Initial guess for 1/sqrt(a) ──
    uint32_t a_abs = a & ~FP32_SIGN_MASK;
    uint32_t a_exp = (a_abs >> 23) & 0xFF;

    // Famous "fast inverse square root" style initial estimate
    // y0 = float_with_exp((3*BIAS - 1 - exp_a) / 2) is a rough 1/sqrt(a)
    uint32_t est_exp = ((3u * FP32_BIAS - 1 - a_exp) >> 1);
    uint32_t y = (est_exp << 23) | (a_abs & FP32_MANT_MASK);

    // Constants
    const uint32_t THREE_FP32 = 0x40400000u;  // 3.0
    const uint32_t HALF_FP32  = 0x3F000000u;  // 0.5

    // ── Iteration: Y_{n+1} = Y_n × (3 - a × Y_n²) × 0.5 ──
    for (int i = 0; i < iterations; ++i) {
        uint32_t y2  = fp32_mul(y, y, rm);          // Y_n²
        uint32_t ay2 = fp32_mul(a_abs, y2, rm);     // a × Y_n²
        uint32_t sub = fp32_sub(THREE_FP32, ay2, rm); // 3 - a × Y_n²
        uint32_t prod = fp32_mul(y, sub, rm);        // Y_n × (3 - a × Y_n²)
        y = fp32_mul(prod, HALF_FP32, rm);           // × 0.5
    }

    // ── Result: sqrt(a) = a × (1/sqrt(a)) ──
    return fp32_mul(a_abs, y, rm);
}

// ═════════════════════════════════════════════════════════════════════════════
// PRECISION CONVERSION HELPERS
// ═════════════════════════════════════════════════════════════════════════════

uint32_t SoftFloatEngine::fp16_to_fp32(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) >> 15) & 1;
    uint32_t exp  = (static_cast<uint32_t>(h) >> 10) & 0x1F;
    uint32_t mant = static_cast<uint32_t>(h) & 0x3FF;

    if (exp == 0x1F) {
        // Inf / NaN → FP32 Inf / NaN (preserve NaN payload)
        return (sign << 31) | 0x7F800000u | (mant << 13);
    }
    if (exp == 0) {
        if (mant == 0) return sign << 31;  // ±0
        // Subnormal FP16 → Normal FP32: find leading 1 and normalize
        int shift = 0;
        while ((mant & (1u << 10)) == 0) { mant <<= 1; shift++; }
        mant &= 0x3FF;  // Remove hidden-1 that we just shifted into position
        uint32_t fp32_exp = FP32_BIAS - FP16_BIAS - shift + 1;
        return (sign << 31) | (fp32_exp << 23) | (mant << 13);
    }
    // Normal: re-bias exponent, shift mantissa
    uint32_t fp32_exp = exp + (FP32_BIAS - FP16_BIAS);
    return (sign << 31) | (fp32_exp << 23) | (mant << 13);
}

uint16_t SoftFloatEngine::fp32_to_fp16(uint32_t f, RoundMode rm) {
    FPUnpacked fp = unpack_fp32(f);
    // unpack_fp32 places hidden-1 at bit 23. pack_fp16 expects hidden-1 at bit 10.
    // Value = mantissa * 2^(exponent - 23) for FP32 convention
    // For FP16, pack_fp16 interprets: value = mantissa * 2^(exponent - 10)
    // So we must NOT modify the mantissa position before pack; instead we
    // let pack_fp16's normalizer handle it, but we must keep the exponent
    // as the TRUE mathematical exponent. The normalizer in pack_fp16 adjusts
    // the mantissa to have hidden-1 at bit 10, and it adds the shift to exponent.
    // This is incorrect for cross-format: the shift should be zero-sum with
    // the mantissa position difference.
    //
    // Solution: pre-shift mantissa to FP16 position (bit 10) with rounding,
    // keeping the exponent unchanged.
    if (!fp.is_nan && !fp.is_inf && !fp.is_zero && fp.mantissa != 0) {
        // Mantissa has hidden-1 at bit 23. Shift right by 13 to bit 10.
        // We must apply rounding on the 13 discarded bits.
        int shift = FP32_MANT_BITS - FP16_MANT_BITS;  // 13
        uint64_t sticky = (fp.mantissa & ((1ULL << (shift - 1)) - 1)) ? 1 : 0;  // bits 0..11
        uint64_t guard = (fp.mantissa >> (shift - 1)) & 1;  // bit 12
        fp.mantissa >>= shift;

        // Apply rounding manually
        bool increment = false;
        switch (rm) {
            case RoundMode::ROUND_NEAREST_EVEN:
                if (guard && (sticky || (fp.mantissa & 1))) increment = true;
                break;
            case RoundMode::ROUND_TOWARD_ZERO: break;
            case RoundMode::ROUND_TOWARD_POS:
                if (!fp.sign && (guard || sticky)) increment = true;
                break;
            case RoundMode::ROUND_TOWARD_NEG:
                if (fp.sign && (guard || sticky)) increment = true;
                break;
        }
        if (increment) fp.mantissa++;

        // If rounding caused carry past bit 11
        if (fp.mantissa & (1ULL << (FP16_MANT_BITS + 1))) {
            fp.mantissa >>= 1;
            fp.exponent++;
        }
    }
    return pack_fp16(fp, rm);
}
