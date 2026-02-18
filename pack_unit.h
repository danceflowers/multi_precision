// =============================================================================
// pack_unit.h — Stage 5: Pack Unit (Output Formatter)
// =============================================================================
// Hardware Mapping: This module sits between the Accumulator File (96-bit) and
// the 32-bit output_z bus. It performs:
//   1. Scaling (shift to align decimal point)
//   2. Saturation/Clamping (overflow detection + mux)
//   3. Truncation/Rounding (lower-bit discard with rounding)
//   4. Bit-Packing (concatenate SIMD lanes back to 32-bit word)
// =============================================================================
#pragma once

#include "mprcu_types.h"

class PackUnit {
public:
    // ─────────────────────────────────────────────────────────────────────
    // Integer saturation: clamp a wider value into N-bit signed/unsigned
    // Models the hardware saturation comparator + mux
    // ─────────────────────────────────────────────────────────────────────

    /// Saturate a signed value to fit in `bits` (signed range)
    static int32_t saturate_signed(int64_t value, int bits);

    /// Saturate an unsigned value to fit in `bits` (unsigned range)
    static uint32_t saturate_unsigned(uint64_t value, int bits);

    // ─────────────────────────────────────────────────────────────────────
    // Scaling / Truncation with rounding
    // ─────────────────────────────────────────────────────────────────────

    /// Right-shift with rounding mode (models barrel shifter + round logic)
    static int64_t scale_and_round(int64_t value, int shift_right, RoundMode rm);

    // ─────────────────────────────────────────────────────────────────────
    // Bit-Packing: Accumulator → 32-bit output_z
    // ─────────────────────────────────────────────────────────────────────

    /// Pack 8× 4-bit results from accumulator 12-bit views into 32-bit
    static uint32_t pack_4bit_simd(const AccumulatorFile &acc, int scale_shift, RoundMode rm);

    /// Pack 4× 8-bit results from accumulator 24-bit views into 32-bit
    static uint32_t pack_8bit_simd(const AccumulatorFile &acc, int scale_shift, RoundMode rm);

    /// Pack 2× 16-bit results from accumulator 48-bit views into 32-bit
    static uint32_t pack_16bit_simd(const AccumulatorFile &acc, int scale_shift, RoundMode rm);

    /// Pack 1× 32-bit result from accumulator 96-bit view into 32-bit
    static uint32_t pack_32bit(const AccumulatorFile &acc, int scale_shift, RoundMode rm);

    /// Pack 2× FP16 results from accumulator into 32-bit ([31:16] | [15:0])
    static uint32_t pack_fp16_dual(uint16_t hi, uint16_t lo);

    /// Generic dispatcher based on precision
    static uint32_t pack(const AccumulatorFile &acc, Precision prec, Mode mode,
                         int scale_shift, RoundMode rm);
};
