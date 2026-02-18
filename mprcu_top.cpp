// =============================================================================
// mprcu_top.cpp — Top-Level MP-RCU Implementation
// =============================================================================

#include "mprcu_top.h"
#include <sstream>
#include <iomanip>

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / Reset — models hardware power-on reset
// ─────────────────────────────────────────────────────────────────────────────

MPRCU_Top::MPRCU_Top() {
    reset();
}

void MPRCU_Top::reset() {
    input_a = input_b = output_z = 0;
    cfg = CfgReg{};
    status = StatusReg{};
    accumulator.clear();
    nl_state = NLState::IDLE;
    nl_iter_cnt = 0;
    nl_operand_a = nl_operand_b = nl_result = 0;
    s1_out = {};
    s2_out = {};
    s5_packed = 0;
    scale_shift = 0;
}

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 1: UNPACK — Input Formatter
// ═════════════════════════════════════════════════════════════════════════════
// Splits 32-bit inputs into appropriate lane representations based on CFG_REG.
// Hardware: MUX tree selecting different bit-field slicing paths.
// ═════════════════════════════════════════════════════════════════════════════

void MPRCU_Top::stage1_unpack() {
    switch (cfg.mode) {
        case Mode::INTEGER:
            // Integer mode: pass through (SIMD decomposition happens in Stage 2)
            s1_out.int_a = input_a;
            s1_out.int_b = input_b;
            break;

        case Mode::FLOAT:
            if (cfg.precision == Precision::P32) {
                // 1× FP32: full 32-bit as float
                s1_out.fp32_a = input_a;
                s1_out.fp32_b = input_b;
            } else if (cfg.precision == Precision::P16) {
                // 2× FP16: [31:16] = upper, [15:0] = lower
                s1_out.fp16_a[0] = static_cast<uint16_t>(input_a & 0xFFFF);
                s1_out.fp16_a[1] = static_cast<uint16_t>((input_a >> 16) & 0xFFFF);
                s1_out.fp16_b[0] = static_cast<uint16_t>(input_b & 0xFFFF);
                s1_out.fp16_b[1] = static_cast<uint16_t>((input_b >> 16) & 0xFFFF);
            }
            break;

        case Mode::COMPLEX:
            // Complex: [31:16] = Imaginary, [15:0] = Real (FP16 each)
            s1_out.complex_ar = static_cast<uint16_t>(input_a & 0xFFFF);
            s1_out.complex_ai = static_cast<uint16_t>((input_a >> 16) & 0xFFFF);
            s1_out.complex_br = static_cast<uint16_t>(input_b & 0xFFFF);
            s1_out.complex_bi = static_cast<uint16_t>((input_b >> 16) & 0xFFFF);
            break;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 2: UNIFIED ARITHMETIC CORE — "The Engine"
// ═════════════════════════════════════════════════════════════════════════════
// Dispatches to the appropriate arithmetic operation based on CFG_REG.
// The base 64×4-bit multiplier array is reused across all modes.
// ═════════════════════════════════════════════════════════════════════════════

void MPRCU_Top::stage2_compute() {
    RoundMode rm = cfg.round_mode;

    switch (cfg.mode) {
    // ─── Integer Mode ───────────────────────────────────────────────
    case Mode::INTEGER: {
        uint32_t a = s1_out.int_a;
        uint32_t b = s1_out.int_b;

        switch (cfg.operation) {
            case Operation::ADD:
                switch (cfg.precision) {
                    case Precision::P4:  s2_out.result_lo = SoftFloatEngine::int_add_simd_4b(a, b); break;
                    case Precision::P8:  s2_out.result_lo = SoftFloatEngine::int_add_simd_8b(a, b); break;
                    case Precision::P16: s2_out.result_lo = SoftFloatEngine::int_add_simd_16b(a, b); break;
                    case Precision::P32: s2_out.result_lo = SoftFloatEngine::int_add_32b(a, b); break;
                }
                break;

            case Operation::MUL:
                switch (cfg.precision) {
                    case Precision::P4:  s2_out.result_lo = SoftFloatEngine::int_mul_simd_4b(a, b); break;
                    case Precision::P8:  s2_out.result_lo = SoftFloatEngine::int_mul_simd_8b(a, b); break;
                    case Precision::P16: s2_out.result_lo = SoftFloatEngine::int_mul_simd_16b(a, b); break;
                    case Precision::P32: s2_out.result_lo = SoftFloatEngine::int_mul_32b(a, b); break;
                }
                break;

            case Operation::MAC:
                // MAC: accumulate mul result into accumulator (handled in stage 3)
                switch (cfg.precision) {
                    case Precision::P4:  s2_out.result_lo = SoftFloatEngine::int_mul_simd_4b(a, b); break;
                    case Precision::P8:  s2_out.result_lo = SoftFloatEngine::int_mul_simd_8b(a, b); break;
                    case Precision::P16: s2_out.result_lo = SoftFloatEngine::int_mul_simd_16b(a, b); break;
                    case Precision::P32: s2_out.result_lo = SoftFloatEngine::int_mul_32b(a, b); break;
                }
                break;

            default: break;
        }
        break;
    }

    // ─── Floating-Point Mode ────────────────────────────────────────
    case Mode::FLOAT: {
        if (cfg.precision == Precision::P32) {
            switch (cfg.operation) {
                case Operation::ADD:
                    s2_out.fp32_result = SoftFloatEngine::fp32_add(s1_out.fp32_a, s1_out.fp32_b, rm);
                    break;
                case Operation::MUL:
                    s2_out.fp32_result = SoftFloatEngine::fp32_mul(s1_out.fp32_a, s1_out.fp32_b, rm);
                    break;
                case Operation::MAC:
                    // Read current accumulator as FP32
                    s2_out.fp32_result = SoftFloatEngine::fp32_fma(
                        s1_out.fp32_a, s1_out.fp32_b,
                        accumulator.data[0],  // Accumulator stores FP32 result
                        rm);
                    break;
                case Operation::DIV:
                case Operation::SQRT:
                    // These are handled by stage4 (non-linear unit)
                    break;
            }
        } else if (cfg.precision == Precision::P16) {
            // 2× FP16 SIMD lanes
            switch (cfg.operation) {
                case Operation::ADD:
                    s2_out.fp16_result[0] = SoftFloatEngine::fp16_add(s1_out.fp16_a[0], s1_out.fp16_b[0], rm);
                    s2_out.fp16_result[1] = SoftFloatEngine::fp16_add(s1_out.fp16_a[1], s1_out.fp16_b[1], rm);
                    break;
                case Operation::MUL:
                    s2_out.fp16_result[0] = SoftFloatEngine::fp16_mul(s1_out.fp16_a[0], s1_out.fp16_b[0], rm);
                    s2_out.fp16_result[1] = SoftFloatEngine::fp16_mul(s1_out.fp16_a[1], s1_out.fp16_b[1], rm);
                    break;
                default: break;
            }
        }
        break;
    }

    // ─── Complex Mode ───────────────────────────────────────────────
    case Mode::COMPLEX: {
        if (cfg.operation == Operation::MUL) {
            s2_out.complex_result = SoftFloatEngine::complex_mul_fp16(input_a, input_b, rm);
        } else if (cfg.operation == Operation::ADD) {
            // Complex add: add real parts and imag parts separately
            uint16_t real_sum = SoftFloatEngine::fp16_add(s1_out.complex_ar, s1_out.complex_br, rm);
            uint16_t imag_sum = SoftFloatEngine::fp16_add(s1_out.complex_ai, s1_out.complex_bi, rm);
            s2_out.complex_result = (static_cast<uint32_t>(imag_sum) << 16) | real_sum;
        }
        break;
    }
    } // end mode switch
}

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 3: ACCUMULATOR FILE — Write-back from arithmetic core
// ═════════════════════════════════════════════════════════════════════════════

void MPRCU_Top::stage3_accumulate() {
    if (cfg.mode == Mode::INTEGER) {
        if (cfg.operation == Operation::MAC) {
            // MAC: add current product to accumulator
            // For simplicity, we accumulate in the 96-bit file
            uint64_t lo64; uint32_t hi32;
            accumulator.read96(lo64, hi32);
            uint64_t sum = lo64 + s2_out.result_lo;
            uint32_t carry = (sum < lo64) ? 1 : 0;
            accumulator.write96(sum, hi32 + carry);
        } else {
            // Non-MAC: write result directly
            accumulator.write96(s2_out.result_lo, 0);
        }
    } else if (cfg.mode == Mode::FLOAT) {
        if (cfg.precision == Precision::P32) {
            // FP32: store result in accumulator data[0]
            accumulator.data[0] = s2_out.fp32_result;
            accumulator.data[1] = 0;
            accumulator.data[2] = 0;
        } else if (cfg.precision == Precision::P16) {
            // 2× FP16: pack into data[0]
            accumulator.data[0] = PackUnit::pack_fp16_dual(
                s2_out.fp16_result[1], s2_out.fp16_result[0]);
            accumulator.data[1] = 0;
            accumulator.data[2] = 0;
        }
    } else if (cfg.mode == Mode::COMPLEX) {
        accumulator.data[0] = s2_out.complex_result;
        accumulator.data[1] = 0;
        accumulator.data[2] = 0;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 4: NON-LINEAR UNIT — Newton-Raphson FSM
// ═════════════════════════════════════════════════════════════════════════════

void MPRCU_Top::stage4_nonlinear_start() {
    nl_state = NLState::CALC;
    nl_iter_cnt = 0;
    nl_operand_a = s1_out.fp32_a;
    nl_operand_b = s1_out.fp32_b;
    status.busy = true;
}

bool MPRCU_Top::stage4_nonlinear_step() {
    if (nl_state != NLState::CALC) return false;

    nl_iter_cnt++;

    // Each step() performs one Newton-Raphson iteration
    // When all iterations complete, transition to DONE
    if (nl_iter_cnt >= cfg.iter_count) {
        // Perform the full computation
        if (cfg.operation == Operation::DIV) {
            nl_result = SoftFloatEngine::fp32_div_nr(
                nl_operand_a, nl_operand_b, cfg.round_mode, cfg.iter_count);
        } else if (cfg.operation == Operation::SQRT) {
            nl_result = SoftFloatEngine::fp32_sqrt_nr(
                nl_operand_a, cfg.round_mode, cfg.iter_count);
        }

        nl_state = NLState::DONE;
        status.busy = false;
        return true;
    }
    return false;
}

// ═════════════════════════════════════════════════════════════════════════════
// STAGE 5: PACK — Output Formatter
// ═════════════════════════════════════════════════════════════════════════════

void MPRCU_Top::stage5_pack() {
    if (cfg.mode == Mode::FLOAT || cfg.mode == Mode::COMPLEX) {
        // Float/Complex: result is already in accumulator data[0] as packed bits
        s5_packed = accumulator.data[0];
    } else if (cfg.mode == Mode::INTEGER) {
        if (cfg.operation == Operation::MAC) {
            // MAC: go through the PackUnit saturation/packing logic
            s5_packed = PackUnit::pack(accumulator, cfg.precision, cfg.mode,
                                       scale_shift, cfg.round_mode);
        } else {
            // Non-MAC integer: result is directly in accumulator data[0] (32-bit)
            s5_packed = accumulator.data[0];
        }
    }
    output_z = s5_packed;
}

// ═════════════════════════════════════════════════════════════════════════════
// step() — Main Simulation Loop (one clock cycle)
// ═════════════════════════════════════════════════════════════════════════════
// Atomic ops (Add/Mul/MAC): complete pipeline in one step() call.
// Iterative ops (Div/Sqrt): FSM-based, requires multiple step() calls.
// ═════════════════════════════════════════════════════════════════════════════

void MPRCU_Top::step() {
    status.cycle_cnt++;

    // ── Handle iterative operations (Div/Sqrt FSM) ──
    if (nl_state == NLState::CALC) {
        if (stage4_nonlinear_step()) {
            // Computation complete: write result to accumulator and pack
            accumulator.data[0] = nl_result;
            accumulator.data[1] = 0;
            accumulator.data[2] = 0;
            stage5_pack();
        }
        return;
    }

    if (nl_state == NLState::DONE) {
        // Ready for next operation
        nl_state = NLState::IDLE;
    }

    // ── Full pipeline for atomic operations ──
    // Stage 1: Unpack inputs
    stage1_unpack();

    // Check if this is an iterative operation
    if ((cfg.operation == Operation::DIV || cfg.operation == Operation::SQRT)
        && cfg.mode == Mode::FLOAT && cfg.precision == Precision::P32) {
        stage4_nonlinear_start();
        return;  // Will complete over multiple step() calls
    }

    // Stage 2: Compute
    stage2_compute();

    // Stage 3: Accumulate
    stage3_accumulate();

    // Stage 5: Pack output
    stage5_pack();

    // ── Update status flags ──
    if (cfg.mode == Mode::FLOAT) {
        uint32_t result_bits = output_z;
        uint32_t exp_field = (result_bits >> 23) & 0xFF;
        status.overflow  = (exp_field == 0xFF) && ((result_bits & 0x7FFFFF) == 0);
        status.nan_flag  = (exp_field == 0xFF) && ((result_bits & 0x7FFFFF) != 0);
        status.inf_flag  = status.overflow;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// execute() — Convenience wrapper for single-shot operation
// ═════════════════════════════════════════════════════════════════════════════

uint32_t MPRCU_Top::execute(uint32_t a, uint32_t b,
                            Mode mode, Precision prec, Operation op,
                            RoundMode rm, int iterations) {
    cfg.mode       = mode;
    cfg.precision  = prec;
    cfg.operation  = op;
    cfg.round_mode = rm;
    cfg.iter_count = static_cast<uint8_t>(iterations);

    input_a = a;
    input_b = b;

    // For iterative ops, step until done
    if (op == Operation::DIV || op == Operation::SQRT) {
        step();  // Starts FSM
        while (nl_state == NLState::CALC) {
            step();
        }
        return output_z;
    }

    // Atomic ops complete in one step
    step();
    return output_z;
}

// ═════════════════════════════════════════════════════════════════════════════
// Debug Dump
// ═════════════════════════════════════════════════════════════════════════════

std::string MPRCU_Top::dump_status() const {
    std::ostringstream oss;
    oss << "──── MPRCU Status ────\n";
    oss << "  Cycle:    " << status.cycle_cnt << "\n";
    oss << "  Busy:     " << (status.busy ? "YES" : "NO") << "\n";
    oss << "  Overflow: " << (status.overflow ? "YES" : "NO") << "\n";
    oss << "  NaN:      " << (status.nan_flag ? "YES" : "NO") << "\n";
    oss << "  NL State: " << static_cast<int>(nl_state) << "\n";
    oss << "  output_z: 0x" << std::hex << std::setfill('0')
        << std::setw(8) << output_z << std::dec << "\n";
    oss << "  Acc[96]:  0x" << std::hex
        << std::setw(8) << accumulator.data[2]
        << std::setw(8) << accumulator.data[1]
        << std::setw(8) << accumulator.data[0]
        << std::dec << "\n";
    return oss.str();
}
