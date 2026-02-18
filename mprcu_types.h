// =============================================================================
// mprcu_types.h — Multi-Precision Reconfigurable Compute Unit: Type Definitions
// =============================================================================
// Hardware Mapping: These types correspond to the CFG_REG and STATUS_REG fields
// visible at the system interface boundary. Every enum value maps 1:1 to a
// hardware configuration bit-field.
// =============================================================================
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <array>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// CFG_REG Field Enumerations
// ─────────────────────────────────────────────────────────────────────────────

/// Operating mode selector — maps to CFG_REG[3:0]
enum class Mode : uint8_t {
    INTEGER  = 0x0,
    FLOAT    = 0x1,
    COMPLEX  = 0x2,
};

/// Precision selector — maps to CFG_REG[7:4]
enum class Precision : uint8_t {
    P4  = 4,   // 4-bit  elements → 8 lanes
    P8  = 8,   // 8-bit  elements → 4 lanes
    P16 = 16,  // 16-bit elements → 2 lanes
    P32 = 32,  // 32-bit elements → 1 lane
};

/// Operation selector — maps to CFG_REG[11:8]
enum class Operation : uint8_t {
    ADD  = 0,
    MUL  = 1,
    MAC  = 2,
    DIV  = 3,
    SQRT = 4,
};

/// IEEE-754 rounding mode — maps to CFG_REG[14:12]
enum class RoundMode : uint8_t {
    ROUND_NEAREST_EVEN = 0,  // RNE — default for IEEE-754
    ROUND_TOWARD_ZERO  = 1,  // Truncate
    ROUND_TOWARD_POS   = 2,  // Ceil
    ROUND_TOWARD_NEG   = 3,  // Floor
};

// ─────────────────────────────────────────────────────────────────────────────
// STATUS_REG and FSM State
// ─────────────────────────────────────────────────────────────────────────────

/// Non-linear unit FSM state — maps to hardware state register
enum class NLState : uint8_t {
    IDLE = 0,
    CALC = 1,
    DONE = 2,
};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration Register (CFG_REG) aggregate
// ─────────────────────────────────────────────────────────────────────────────
struct CfgReg {
    Mode       mode       = Mode::FLOAT;
    Precision  precision  = Precision::P32;
    Operation  operation  = Operation::MUL;
    RoundMode  round_mode = RoundMode::ROUND_NEAREST_EVEN;
    uint8_t    iter_count = 4;  // Newton-Raphson iterations for Div/Sqrt (1–8)
};

/// Status Register (STATUS_REG) aggregate — directly maps to flop outputs
struct StatusReg {
    bool     busy       = false;
    bool     overflow   = false;
    bool     underflow  = false;
    bool     nan_flag   = false;
    bool     inf_flag   = false;
    uint32_t cycle_cnt  = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// 96-bit Accumulator File — Hardware Mapping
// ─────────────────────────────────────────────────────────────────────────────
// The accumulator is a unified 96-bit register with aliased sub-views.
// In hardware this is a single register file with mux-based read ports.
// ─────────────────────────────────────────────────────────────────────────────
struct AccumulatorFile {
    uint32_t data[3] = {0, 0, 0};  // data[0]=LSB, data[2]=MSB → 96 bits total

    /// Clear entire 96-bit storage
    void clear() { data[0] = data[1] = data[2] = 0; }

    // ── Full 96-bit access (used for FP32 / Complex MAC) ───────────────
    void write96(uint64_t lo64, uint32_t hi32) {
        data[0] = static_cast<uint32_t>(lo64);
        data[1] = static_cast<uint32_t>(lo64 >> 32);
        data[2] = hi32;
    }

    void read96(uint64_t &lo64, uint32_t &hi32) const {
        lo64 = (static_cast<uint64_t>(data[1]) << 32) | data[0];
        hi32 = data[2];
    }

    // ── 2×48-bit views (for FP16 operations) ──────────────────────────
    void write48(int idx, uint64_t val) {
        assert(idx >= 0 && idx < 2);
        if (idx == 0) {
            data[0] = static_cast<uint32_t>(val);
            data[1] = (data[1] & 0xFFFF0000u) | static_cast<uint32_t>((val >> 32) & 0xFFFF);
        } else {
            data[1] = (data[1] & 0x0000FFFFu) | static_cast<uint32_t>((val & 0xFFFF) << 16);
            data[2] = static_cast<uint32_t>(val >> 16);
        }
    }

    uint64_t read48(int idx) const {
        assert(idx >= 0 && idx < 2);
        if (idx == 0) {
            return (static_cast<uint64_t>(data[1] & 0xFFFF) << 32) | data[0];
        } else {
            return (static_cast<uint64_t>(data[2]) << 16) | (data[1] >> 16);
        }
    }

    // ── 4×24-bit views (for 8-bit operations) ─────────────────────────
    void write24(int idx, uint32_t val) {
        assert(idx >= 0 && idx < 4);
        val &= 0x00FFFFFFu;
        int bit_off = idx * 24;
        // Clear and set across the 3×32-bit words
        for (int b = 0; b < 24; ++b) {
            int gbit = bit_off + b;
            int word = gbit / 32;
            int wbit = gbit % 32;
            if (val & (1u << b))
                data[word] |=  (1u << wbit);
            else
                data[word] &= ~(1u << wbit);
        }
    }

    uint32_t read24(int idx) const {
        assert(idx >= 0 && idx < 4);
        uint32_t result = 0;
        int bit_off = idx * 24;
        for (int b = 0; b < 24; ++b) {
            int gbit = bit_off + b;
            int word = gbit / 32;
            int wbit = gbit % 32;
            if (data[word] & (1u << wbit))
                result |= (1u << b);
        }
        return result;
    }

    // ── 8×12-bit views (for 4-bit operations) ─────────────────────────
    void write12(int idx, uint16_t val) {
        assert(idx >= 0 && idx < 8);
        val &= 0x0FFFu;
        int bit_off = idx * 12;
        for (int b = 0; b < 12; ++b) {
            int gbit = bit_off + b;
            int word = gbit / 32;
            int wbit = gbit % 32;
            if (val & (1u << b))
                data[word] |=  (1u << wbit);
            else
                data[word] &= ~(1u << wbit);
        }
    }

    uint16_t read12(int idx) const {
        assert(idx >= 0 && idx < 8);
        uint16_t result = 0;
        int bit_off = idx * 12;
        for (int b = 0; b < 12; ++b) {
            int gbit = bit_off + b;
            int word = gbit / 32;
            int wbit = gbit % 32;
            if (data[word] & (1u << wbit))
                result |= (1u << b);
        }
        return result;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Soft-Float unpacked representation
// ─────────────────────────────────────────────────────────────────────────────
// This struct mirrors the pipeline latch between Stage 1 (Unpack) and Stage 2
// (Arithmetic Core). In hardware, each field corresponds to a named bus.
// ─────────────────────────────────────────────────────────────────────────────
struct FPUnpacked {
    uint32_t sign     = 0;   // 1 bit
    int32_t  exponent = 0;   // Biased or unbiased, depends on context
    uint64_t mantissa = 0;   // With or without hidden-1, depends on stage
    bool     is_zero  = false;
    bool     is_inf   = false;
    bool     is_nan   = false;
    bool     is_subnormal = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Helper: union-free bit reinterpretation (strict aliasing safe)
// ─────────────────────────────────────────────────────────────────────────────
inline uint32_t float_to_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline float bits_to_float(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}


