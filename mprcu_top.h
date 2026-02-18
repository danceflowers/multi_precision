// =============================================================================
// mprcu_top.h — Top-Level MP-RCU Module
// =============================================================================
// Hardware Mapping: This class is the structural top-level that instantiates
// and connects all sub-modules (Unpack, Arithmetic Core, Accumulator, NL Unit,
// Pack Unit) and manages the cycle-level simulation via step().
//
// Architecture:
//   input_a/b → [Unpack] → [Arithmetic Core] → [Accumulator] → [NL Unit] → [Pack] → output_z
//
// Timing:
//   - Atomic ops (Add/Mul/MAC): complete in 1 step() call
//   - Iterative ops (Div/Sqrt): FSM-based, multiple step() calls
// =============================================================================
#pragma once

#include "mprcu_types.h"
#include "soft_float_engine.h"
#include "pack_unit.h"
#include <string>

class MPRCU_Top {
public:
    // ─────────────────────────────────────────────────────────────────────
    // System Interface (directly maps to top-level ports)
    // ─────────────────────────────────────────────────────────────────────
    uint32_t input_a  = 0;
    uint32_t input_b  = 0;
    uint32_t output_z = 0;

    // ─────────────────────────────────────────────────────────────────────
    // Configuration and Status Registers (memory-mapped in hardware)
    // ─────────────────────────────────────────────────────────────────────
    CfgReg    cfg;
    StatusReg status;

    // ─────────────────────────────────────────────────────────────────────
    // Internal pipeline storage (maps to pipeline flip-flops)
    // ─────────────────────────────────────────────────────────────────────
    AccumulatorFile accumulator;

    // ─────────────────────────────────────────────────────────────────────
    // Non-Linear Unit FSM state (maps to state register)
    // ─────────────────────────────────────────────────────────────────────
    NLState  nl_state    = NLState::IDLE;
    uint32_t nl_iter_cnt = 0;
    uint32_t nl_operand_a = 0;  // Latched operands for iterative ops
    uint32_t nl_operand_b = 0;
    uint32_t nl_result   = 0;

    // ─────────────────────────────────────────────────────────────────────
    // Constructor / Reset
    // ─────────────────────────────────────────────────────────────────────
    MPRCU_Top();
    void reset();

    // ─────────────────────────────────────────────────────────────────────
    // Main simulation interface: advance one clock cycle
    // ─────────────────────────────────────────────────────────────────────
    void step();

    // ─────────────────────────────────────────────────────────────────────
    // Convenience: configure and execute in one call (test helper)
    // ─────────────────────────────────────────────────────────────────────
    uint32_t execute(uint32_t a, uint32_t b,
                     Mode mode, Precision prec, Operation op,
                     RoundMode rm = RoundMode::ROUND_NEAREST_EVEN,
                     int iterations = 4);

    // ─────────────────────────────────────────────────────────────────────
    // Debug / Trace
    // ─────────────────────────────────────────────────────────────────────
    std::string dump_status() const;

private:
    // ── Pipeline stage functions (each maps to a hardware stage) ──
    void stage1_unpack();    // Input formatter
    void stage2_compute();   // Unified arithmetic core
    void stage3_accumulate();// Accumulator file write
    void stage5_pack();      // Output formatter

    // ── Non-linear unit FSM (iterative ops) ──
    void stage4_nonlinear_start();
    bool stage4_nonlinear_step();  // Returns true when DONE

    // ── Internal pipeline registers (inter-stage latches) ──
    // Stage 1 → Stage 2: unpacked operands
    struct {
        // Integer SIMD lanes
        uint32_t int_a = 0, int_b = 0;

        // Float: up to 2× FP16 or 1× FP32
        uint32_t fp32_a = 0, fp32_b = 0;
        uint16_t fp16_a[2] = {}, fp16_b[2] = {};

        // Complex: separated real/imag
        uint16_t complex_ar = 0, complex_ai = 0;
        uint16_t complex_br = 0, complex_bi = 0;
    } s1_out;

    // Stage 2 → Stage 3: arithmetic results (pre-accumulation)
    struct {
        uint64_t result_lo = 0;   // Lower 64 bits of result
        uint32_t result_hi = 0;   // Upper 32 bits (for 96-bit accumulation)
        uint32_t fp32_result = 0; // FP32 result word
        uint16_t fp16_result[2] = {};
        uint32_t complex_result = 0;
    } s2_out;

    // Stage 5 output latch
    uint32_t s5_packed = 0;

    int scale_shift = 0;  // Output scaling factor
};
