// =============================================================================
// main.cpp — MP-RCU C-Model Test Bench
// =============================================================================
// Tests all major operations: FP32/FP16 Add/Mul/FMA, Complex, Integer SIMD,
// Newton-Raphson Div/Sqrt, and PackUnit saturation logic.
//
// NOTE: float_to_bits() / bits_to_float() are used ONLY for test stimulus
// generation and result display. No native float is used inside the engine.
// =============================================================================

#include "mprcu_top.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Test helper: compare soft-float result with expected bits
// ─────────────────────────────────────────────────────────────────────────────
static int tests_passed = 0;
static int tests_failed = 0;

static void check_fp32(const char* name, uint32_t got, uint32_t expected, int ulp_tolerance = 0) {
    int32_t diff = static_cast<int32_t>(got) - static_cast<int32_t>(expected);
    if (diff < 0) diff = -diff;

    if (diff <= ulp_tolerance) {
        printf("  [PASS] %-30s got=0x%08X (%.7e)  expected=0x%08X (%.7e)\n",
               name, got, bits_to_float(got), expected, bits_to_float(expected));
        tests_passed++;
    } else {
        printf("  [FAIL] %-30s got=0x%08X (%.7e)  expected=0x%08X (%.7e)  diff=%d ULP\n",
               name, got, bits_to_float(got), expected, bits_to_float(expected), diff);
        tests_failed++;
    }
}

static void check_u32(const char* name, uint32_t got, uint32_t expected) {
    if (got == expected) {
        printf("  [PASS] %-30s got=0x%08X  expected=0x%08X\n", name, got, expected);
        tests_passed++;
    } else {
        printf("  [FAIL] %-30s got=0x%08X  expected=0x%08X\n", name, got, expected);
        tests_failed++;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// TEST SUITE
// ═════════════════════════════════════════════════════════════════════════════

void test_fp32_basic() {
    printf("\n=== FP32 Basic Operations ===\n");
    MPRCU_Top dut;

    // ── Addition ──
    {
        uint32_t a = float_to_bits(1.5f);
        uint32_t b = float_to_bits(2.25f);
        uint32_t result = dut.execute(a, b, Mode::FLOAT, Precision::P32, Operation::ADD);
        check_fp32("1.5 + 2.25 = 3.75", result, float_to_bits(3.75f));
    }
    {
        uint32_t a = float_to_bits(1.0f);
        uint32_t b = float_to_bits(-1.0f);
        uint32_t result = dut.execute(a, b, Mode::FLOAT, Precision::P32, Operation::ADD);
        check_fp32("1.0 + (-1.0) = 0.0", result, float_to_bits(0.0f));
    }
    {
        // Large + small (alignment test)
        uint32_t a = float_to_bits(1.0e10f);
        uint32_t b = float_to_bits(1.0f);
        uint32_t result = dut.execute(a, b, Mode::FLOAT, Precision::P32, Operation::ADD);
        check_fp32("1e10 + 1.0", result, float_to_bits(1.0e10f + 1.0f), 1);
    }

    // ── Multiplication ──
    {
        uint32_t a = float_to_bits(3.0f);
        uint32_t b = float_to_bits(4.0f);
        uint32_t result = dut.execute(a, b, Mode::FLOAT, Precision::P32, Operation::MUL);
        check_fp32("3.0 * 4.0 = 12.0", result, float_to_bits(12.0f));
    }
    {
        uint32_t a = float_to_bits(-2.5f);
        uint32_t b = float_to_bits(0.4f);
        uint32_t result = dut.execute(a, b, Mode::FLOAT, Precision::P32, Operation::MUL);
        check_fp32("-2.5 * 0.4 = -1.0", result, float_to_bits(-1.0f), 1);
    }
    {
        uint32_t a = float_to_bits(1.23456789f);
        uint32_t b = float_to_bits(9.87654321f);
        uint32_t result = dut.execute(a, b, Mode::FLOAT, Precision::P32, Operation::MUL);
        uint32_t expected = float_to_bits(1.23456789f * 9.87654321f);
        check_fp32("1.234... * 9.876...", result, expected, 1);
    }
}

void test_fp32_special_cases() {
    printf("\n=== FP32 Special Cases ===\n");
    MPRCU_Top dut;

    // NaN propagation
    {
        uint32_t result = dut.execute(SoftFloatEngine::FP32_QNAN, float_to_bits(1.0f),
                                      Mode::FLOAT, Precision::P32, Operation::ADD);
        check_fp32("NaN + 1.0 = NaN", result, SoftFloatEngine::FP32_QNAN);
    }
    // Inf + (-Inf) = NaN
    {
        uint32_t result = dut.execute(SoftFloatEngine::FP32_POS_INF, SoftFloatEngine::FP32_NEG_INF,
                                      Mode::FLOAT, Precision::P32, Operation::ADD);
        check_fp32("+Inf + (-Inf) = NaN", result, SoftFloatEngine::FP32_QNAN);
    }
    // 0 * Inf = NaN
    {
        uint32_t result = dut.execute(float_to_bits(0.0f), SoftFloatEngine::FP32_POS_INF,
                                      Mode::FLOAT, Precision::P32, Operation::MUL);
        check_fp32("0 * Inf = NaN", result, SoftFloatEngine::FP32_QNAN);
    }
    // Inf * 2 = Inf
    {
        uint32_t result = dut.execute(SoftFloatEngine::FP32_POS_INF, float_to_bits(2.0f),
                                      Mode::FLOAT, Precision::P32, Operation::MUL);
        check_fp32("Inf * 2 = Inf", result, SoftFloatEngine::FP32_POS_INF);
    }
}

void test_fp32_subnormals() {
    printf("\n=== FP32 Subnormal Handling ===\n");
    MPRCU_Top dut;

    // Smallest subnormal
    {
        uint32_t a = 0x00000001u;  // Smallest positive subnormal
        uint32_t b = float_to_bits(2.0f);
        uint32_t result = dut.execute(a, b, Mode::FLOAT, Precision::P32, Operation::MUL);
        // Should be 0x00000002 (double the smallest subnormal)
        check_fp32("min_subnormal * 2", result, 0x00000002u, 1);
    }
    // Subnormal + subnormal
    {
        uint32_t a = 0x00000001u;
        uint32_t b = 0x00000001u;
        uint32_t result = dut.execute(a, b, Mode::FLOAT, Precision::P32, Operation::ADD);
        check_fp32("min_sub + min_sub", result, 0x00000002u);
    }
}

void test_fp32_mac() {
    printf("\n=== FP32 Fused Multiply-Accumulate ===\n");
    MPRCU_Top dut;

    // Simple MAC: 0 + (2.0 × 3.0) = 6.0
    {
        dut.reset();
        dut.accumulator.data[0] = float_to_bits(0.0f);
        uint32_t result = dut.execute(float_to_bits(2.0f), float_to_bits(3.0f),
                                      Mode::FLOAT, Precision::P32, Operation::MAC);
        check_fp32("0 + 2*3 = 6", result, float_to_bits(6.0f), 1);
    }
    // MAC with accumulator: 10.0 + (2.0 × 3.0) = 16.0
    {
        dut.reset();
        dut.accumulator.data[0] = float_to_bits(10.0f);
        uint32_t result = dut.execute(float_to_bits(2.0f), float_to_bits(3.0f),
                                      Mode::FLOAT, Precision::P32, Operation::MAC);
        check_fp32("10 + 2*3 = 16", result, float_to_bits(16.0f), 1);
    }
}

void test_fp32_div_sqrt() {
    printf("\n=== FP32 Newton-Raphson Division & Square Root ===\n");
    MPRCU_Top dut;

    // Division: 10.0 / 3.0
    {
        uint32_t result = dut.execute(float_to_bits(10.0f), float_to_bits(3.0f),
                                      Mode::FLOAT, Precision::P32, Operation::DIV,
                                      RoundMode::ROUND_NEAREST_EVEN, 6);
        uint32_t expected = float_to_bits(10.0f / 3.0f);
        check_fp32("10 / 3", result, expected, 4);  // NR has limited precision
    }
    // Division: 1.0 / 2.0
    {
        uint32_t result = dut.execute(float_to_bits(1.0f), float_to_bits(2.0f),
                                      Mode::FLOAT, Precision::P32, Operation::DIV,
                                      RoundMode::ROUND_NEAREST_EVEN, 6);
        check_fp32("1 / 2 = 0.5", result, float_to_bits(0.5f), 4);
    }
    // Sqrt: sqrt(4.0) = 2.0
    {
        uint32_t result = dut.execute(float_to_bits(4.0f), 0,
                                      Mode::FLOAT, Precision::P32, Operation::SQRT,
                                      RoundMode::ROUND_NEAREST_EVEN, 8);
        check_fp32("sqrt(4) = 2", result, float_to_bits(2.0f), 4);
    }
    // Sqrt: sqrt(2.0)
    {
        uint32_t result = dut.execute(float_to_bits(2.0f), 0,
                                      Mode::FLOAT, Precision::P32, Operation::SQRT,
                                      RoundMode::ROUND_NEAREST_EVEN, 8);
        uint32_t expected = float_to_bits(sqrtf(2.0f));
        check_fp32("sqrt(2)", result, expected, 8);
    }
}

void test_fp16_operations() {
    printf("\n=== FP16 Operations (2× SIMD) ===\n");
    MPRCU_Top dut;

    // Pack 2× FP16: [1.0, 2.0] + [0.5, 0.25]
    // FP16: 1.0 = 0x3C00, 2.0 = 0x4000, 0.5 = 0x3800, 0.25 = 0x3400
    {
        uint32_t a = (0x4000u << 16) | 0x3C00u;  // [2.0, 1.0]
        uint32_t b = (0x3400u << 16) | 0x3800u;  // [0.25, 0.5]
        uint32_t result = dut.execute(a, b, Mode::FLOAT, Precision::P16, Operation::ADD);
        // Expected: [2.25, 1.5] = [0x4080, 0x3E00]
        uint32_t expected = (0x4080u << 16) | 0x3E00u;
        check_u32("FP16 [2.0,1.0]+[0.25,0.5]", result, expected);
    }

    // FP16 Mul: [1.5, 2.0] × [2.0, 3.0]
    {
        uint32_t a = (0x3E00u << 16) | 0x4000u;  // [1.5, 2.0]
        uint32_t b = (0x4000u << 16) | 0x4200u;  // [2.0, 3.0]
        uint32_t result = dut.execute(a, b, Mode::FLOAT, Precision::P16, Operation::MUL);
        // Expected: [3.0, 6.0] = [0x4200, 0x4600]
        uint32_t expected = (0x4200u << 16) | 0x4600u;
        check_u32("FP16 [1.5,2.0]*[2.0,3.0]", result, expected);
    }
}

void test_complex_mul() {
    printf("\n=== Complex Multiplication (FP16 Real+Imag) ===\n");
    MPRCU_Top dut;

    // (1 + 2i) × (3 + 4i) = (1×3 - 2×4) + (1×4 + 2×3)i = -5 + 10i
    // FP16: 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200, 4.0=0x4400
    //       -5.0=0xC500, 10.0=0x4900
    {
        uint32_t a = (0x4000u << 16) | 0x3C00u;  // [imag=2.0, real=1.0]
        uint32_t b = (0x4400u << 16) | 0x4200u;  // [imag=4.0, real=3.0]
        uint32_t result = dut.execute(a, b, Mode::COMPLEX, Precision::P16, Operation::MUL);
        uint32_t expected = (0x4900u << 16) | 0xC500u;  // [imag=10.0, real=-5.0]
        check_u32("(1+2i)*(3+4i)=(-5+10i)", result, expected);
    }
}

void test_integer_simd() {
    printf("\n=== Integer SIMD Operations ===\n");
    MPRCU_Top dut;

    // 8× 4-bit add: each lane adds independently with wrap
    {
        uint32_t a = 0x12345678u;
        uint32_t b = 0x11111111u;
        uint32_t result = dut.execute(a, b, Mode::INTEGER, Precision::P4, Operation::ADD);
        // Lane-by-lane 4-bit add with wrap:
        // 8+1=9, 7+1=8, 6+1=7, 5+1=6, 4+1=5, 3+1=4, 2+1=3, 1+1=2
        check_u32("4-bit SIMD add", result, 0x23456789u);
    }

    // 4× 8-bit add
    {
        uint32_t a = 0x01020304u;
        uint32_t b = 0x0A0B0C0Du;
        uint32_t result = dut.execute(a, b, Mode::INTEGER, Precision::P8, Operation::ADD);
        // 0x01+0x0A=0x0B, 0x02+0x0B=0x0D, 0x03+0x0C=0x0F, 0x04+0x0D=0x11
        check_u32("8-bit SIMD add", result, 0x0B0D0F11u);
    }

    // 32-bit multiply
    {
        uint32_t a = 7;
        uint32_t b = 6;
        uint32_t result = dut.execute(a, b, Mode::INTEGER, Precision::P32, Operation::MUL);
        check_u32("32-bit 7*6=42", result, 42u);
    }
}

void test_pack_unit() {
    printf("\n=== Pack Unit Saturation ===\n");

    // Signed saturation
    {
        int32_t r = PackUnit::saturate_signed(200, 8);
        printf("  [%s] sat_signed(200, 8)=%d  expected=127\n",
               (r == 127) ? "PASS" : "FAIL", r);
        if (r == 127) tests_passed++; else tests_failed++;
    }
    {
        int32_t r = PackUnit::saturate_signed(-200, 8);
        printf("  [%s] sat_signed(-200, 8)=%d  expected=-128\n",
               (r == -128) ? "PASS" : "FAIL", r);
        if (r == -128) tests_passed++; else tests_failed++;
    }
    // Unsigned saturation
    {
        uint32_t r = PackUnit::saturate_unsigned(300, 8);
        printf("  [%s] sat_unsigned(300, 8)=%u  expected=255\n",
               (r == 255) ? "PASS" : "FAIL", r);
        if (r == 255) tests_passed++; else tests_failed++;
    }
}

void test_accumulator_views() {
    printf("\n=== Accumulator Aliased Views ===\n");
    AccumulatorFile acc;
    acc.clear();

    // Write/Read 96-bit
    acc.write96(0xDEADBEEFCAFEBABEULL, 0x12345678u);
    uint64_t lo; uint32_t hi;
    acc.read96(lo, hi);
    {
        bool pass = (lo == 0xDEADBEEFCAFEBABEULL && hi == 0x12345678u);
        printf("  [%s] 96-bit read/write\n", pass ? "PASS" : "FAIL");
        if (pass) tests_passed++; else tests_failed++;
    }

    // Write/Read 12-bit (8 lanes)
    acc.clear();
    for (int i = 0; i < 8; ++i) acc.write12(i, static_cast<uint16_t>(i * 100));
    {
        bool pass = true;
        for (int i = 0; i < 8; ++i) {
            if (acc.read12(i) != static_cast<uint16_t>((i * 100) & 0xFFF)) {
                pass = false;
                break;
            }
        }
        printf("  [%s] 8×12-bit read/write\n", pass ? "PASS" : "FAIL");
        if (pass) tests_passed++; else tests_failed++;
    }
}

void test_precision_conversion() {
    printf("\n=== Precision Conversion FP16 <-> FP32 ===\n");

    // FP16 1.0 (0x3C00) → FP32 1.0 (0x3F800000)
    {
        uint32_t r = SoftFloatEngine::fp16_to_fp32(0x3C00);
        check_fp32("FP16→FP32 1.0", r, 0x3F800000u);
    }
    // FP32 1.0 → FP16 1.0
    {
        uint16_t r = SoftFloatEngine::fp32_to_fp16(0x3F800000u, RoundMode::ROUND_NEAREST_EVEN);
        bool pass = (r == 0x3C00);
        printf("  [%s] FP32→FP16 1.0:  got=0x%04X  expected=0x3C00\n", pass ? "PASS" : "FAIL", r);
        if (pass) tests_passed++; else tests_failed++;
    }
    // FP16 Inf
    {
        uint32_t r = SoftFloatEngine::fp16_to_fp32(0x7C00);
        check_fp32("FP16→FP32 +Inf", r, 0x7F800000u);
    }
}

void test_rounding_modes() {
    printf("\n=== Rounding Modes ===\n");

    // Test: 1.0 + epsilon near rounding boundary
    // We'll test multiplication that produces exact mid-point values
    // 1.00000011920928955... × 1.0 = should be exact in RNE

    // Test: FP32 add with round toward zero
    {
        MPRCU_Top dut;
        uint32_t a = float_to_bits(1.0f);
        uint32_t b = 0x33800000u;  // Very small positive: ~5.96e-8
        uint32_t rne = dut.execute(a, b, Mode::FLOAT, Precision::P32, Operation::ADD,
                                   RoundMode::ROUND_NEAREST_EVEN);
        uint32_t rtz = dut.execute(a, b, Mode::FLOAT, Precision::P32, Operation::ADD,
                                   RoundMode::ROUND_TOWARD_ZERO);
        printf("  [INFO] 1.0 + tiny:  RNE=0x%08X  RTZ=0x%08X\n", rne, rtz);
        // RTZ should be <= RNE for positive results
        bool pass = (rtz <= rne);
        printf("  [%s] RTZ <= RNE for positive add\n", pass ? "PASS" : "FAIL");
        if (pass) tests_passed++; else tests_failed++;
    }
}

void test_cycle_counting() {
    printf("\n=== Cycle Counting (Div/Sqrt FSM) ===\n");
    MPRCU_Top dut;
    dut.reset();

    dut.cfg.mode = Mode::FLOAT;
    dut.cfg.precision = Precision::P32;
    dut.cfg.operation = Operation::DIV;
    dut.cfg.iter_count = 4;
    dut.cfg.round_mode = RoundMode::ROUND_NEAREST_EVEN;
    dut.input_a = float_to_bits(10.0f);
    dut.input_b = float_to_bits(2.0f);

    uint32_t start_cycle = dut.status.cycle_cnt;
    dut.step();  // Start FSM
    while (dut.nl_state == NLState::CALC) {
        dut.step();
    }
    uint32_t total_cycles = dut.status.cycle_cnt - start_cycle;

    printf("  Division took %u cycles (iter_count=%d)\n", total_cycles, dut.cfg.iter_count);
    // Should be iter_count + 1 (start + N iterations)
    check_fp32("10/2=5", dut.output_z, float_to_bits(5.0f), 4);
    bool pass = (total_cycles == static_cast<uint32_t>(dut.cfg.iter_count + 1));
    printf("  [%s] Cycle count matches iter_count+1\n", pass ? "PASS" : "FAIL");
    if (pass) tests_passed++; else tests_failed++;
}

// ═════════════════════════════════════════════════════════════════════════════
// MAIN
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   MP-RCU C-Model — Golden Model Test Suite          ║\n");
    printf("║   Bit-Accurate Soft-Float | No Native Float Math    ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    test_fp32_basic();
    test_fp32_special_cases();
    test_fp32_subnormals();
    test_fp32_mac();
    test_fp32_div_sqrt();
    test_fp16_operations();
    test_complex_mul();
    test_integer_simd();
    test_pack_unit();
    test_accumulator_views();
    test_precision_conversion();
    test_rounding_modes();
    test_cycle_counting();

    printf("\n══════════════════════════════════════════════════════\n");
    printf("  Results: %d PASSED, %d FAILED out of %d tests\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("══════════════════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
