/*
 * Code2Edge Target Hardware Benchmark Harness
 *
 * Target:    Arduino UNO Q
 * MCU:       STMicroelectronics STM32U585 (ARM Cortex-M33 @ 160 MHz)
 * Core:      arduino:zephyr (1.0.0)
 * FQBN:      arduino:zephyr:unoq
 *
 * Purpose:
 * Standalone, zero-heap benchmark harness using Cortex-M33 DWT cycle counting (DWT_CYCCNT).
 * Measures per-iteration CPU cycles, min/max/average/stddev, and calculates execution time in microseconds.
 */

#include <Arduino.h>
#include <math.h>

// ==============================================================================
// 1. Hardware Architecture & DWT Cycle Counter Configuration
// ==============================================================================

// Target CPU Clock Assumption: STM32U585 Cortex-M33 running at nominal 160 MHz
#define TARGET_CPU_HZ 160000000UL
#define MAX_BENCHMARK_ITERATIONS 100
#define SERIAL_BAUD_RATE 115200

// Direct ARMv8-M / Cortex-M33 DWT & CoreDebug Memory-Mapped Registers
#define DWT_DEMCR_REG       (*((volatile uint32_t*)0xE000EDFC)) // Debug Exception and Monitor Control Register
#define DWT_CTRL_REG        (*((volatile uint32_t*)0xE0001000)) // DWT Control Register
#define DWT_CYCCNT_REG      (*((volatile uint32_t*)0xE0001004)) // DWT Cycle Count Register
#define DWT_TRCENA_BIT      (1UL << 24)                         // DEMCR TRCENA enable bit
#define DWT_CYCCNTENA_BIT   (1UL << 0)                          // DWT_CTRL CYCCNTENA enable bit

static bool s_dwt_initialized = false;

/**
 * Initialize ARM Cortex-M33 DWT cycle counter.
 * Returns true if cycle counter successfully enables and increments.
 */
static bool dwt_init(void) {
    // 1. Enable Global Trace in CoreDebug DEMCR
    DWT_DEMCR_REG |= DWT_TRCENA_BIT;

    // 2. Reset Cycle Counter
    DWT_CYCCNT_REG = 0;

    // 3. Enable Cycle Counter in DWT Control Register
    DWT_CTRL_REG |= DWT_CYCCNTENA_BIT;

    // 4. Verify that the counter is actively incrementing
    volatile uint32_t c1 = DWT_CYCCNT_REG;
    for (volatile int i = 0; i < 100; i++) {
        __asm__ volatile ("nop");
    }
    volatile uint32_t c2 = DWT_CYCCNT_REG;

    s_dwt_initialized = (c2 > c1);
    return s_dwt_initialized;
}

static inline uint32_t dwt_get_cycles(void) {
    return DWT_CYCCNT_REG;
}

// ==============================================================================
// 2. Benchmark Data Structures & Reporting
// ==============================================================================

struct BenchmarkResult {
    const char* name;
    size_t iterations;
    uint32_t min_cycles;
    uint32_t max_cycles;
    double avg_cycles;
    double stddev_cycles;
    uint32_t cpu_hz;
    double min_us;
    double max_us;
    double avg_us;
    uint32_t checksum;
};

// Static storage to guarantee ZERO heap allocation during measurement
static uint32_t s_cycles_record[MAX_BENCHMARK_ITERATIONS];

// Forward declaration of the deterministic validation kernel
static void deterministic_kernel(void);
static volatile uint32_t g_workload_checksum = 0;

/**
 * Reusable benchmark runner.
 *
 * @param name Name of the workload/kernel
 * @param fn Function pointer to benchmark
 * @param iterations Number of measured iterations (max MAX_BENCHMARK_ITERATIONS)
 * @param warmup Number of unmeasured warm-up iterations
 */
static void benchmark_target(const char* name, void (*fn)(), size_t iterations = 50, size_t warmup = 5) {
    if (!s_dwt_initialized) {
        Serial.println(F("ERROR: DWT cycle counter is not initialized or not supported on this platform."));
        return;
    }

    if (iterations > MAX_BENCHMARK_ITERATIONS) {
        iterations = MAX_BENCHMARK_ITERATIONS;
    }

    // Explicit target clock assumption for STM32U585 on Arduino UNO Q (160 MHz)
    const uint32_t cpu_hz = TARGET_CPU_HZ;

    // 1. Warm-up iterations (fills instruction cache / pipeline state)
    for (size_t w = 0; w < warmup; ++w) {
        fn();
    }

    // 2. Measured iterations
    uint32_t min_cycles = 0xFFFFFFFF;
    uint32_t max_cycles = 0;
    uint64_t total_cycles = 0;

    for (size_t i = 0; i < iterations; ++i) {
        uint32_t start = dwt_get_cycles();
        fn();
        uint32_t end = dwt_get_cycles();

        uint32_t diff = end - start;
        s_cycles_record[i] = diff;
        total_cycles += diff;

        if (diff < min_cycles) min_cycles = diff;
        if (diff > max_cycles) max_cycles = diff;
    }

    // 3. Statistical computations
    double avg_cycles = (double)total_cycles / (double)iterations;

    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < iterations; ++i) {
        double d = (double)s_cycles_record[i] - avg_cycles;
        sum_sq_diff += (d * d);
    }
    double stddev_cycles = (iterations > 1) ? sqrt(sum_sq_diff / (double)(iterations - 1)) : 0.0;

    double cycles_per_us = (double)cpu_hz / 1000000.0;
    double min_us = (double)min_cycles / cycles_per_us;
    double max_us = (double)max_cycles / cycles_per_us;
    double avg_us = avg_cycles / cycles_per_us;

    BenchmarkResult res = {
        name,
        iterations,
        min_cycles,
        max_cycles,
        avg_cycles,
        stddev_cycles,
        cpu_hz,
        min_us,
        max_us,
        avg_us,
        g_workload_checksum
    };

    // 4. Human-Readable Serial Output
    Serial.println(F("CODE2EDGE_BENCHMARK_START"));
    Serial.println(F("================================================================"));
    Serial.print(F(" Benchmark Workload: "));
    Serial.println(res.name);
    Serial.println(F(" Target MCU:         STMicroelectronics STM32U585 (ARM Cortex-M33)"));
    Serial.println(F(" Board FQBN:         arduino:zephyr:unoq"));
    Serial.print(F(" CPU Clock:          "));
    Serial.print(res.cpu_hz / 1000000.0, 1);
    Serial.println(F(" MHz"));
    Serial.print(F(" Iterations:         "));
    Serial.print(res.iterations);
    Serial.print(F(" (Warmup: "));
    Serial.print(warmup);
    Serial.println(F(")"));
    Serial.println(F("----------------------------------------------------------------"));
    Serial.print(F(" Min Latency:        "));
    Serial.print(res.min_cycles);
    Serial.print(F(" cycles ("));
    Serial.print(res.min_us, 4);
    Serial.println(F(" us)"));
    Serial.print(F(" Max Latency:        "));
    Serial.print(res.max_cycles);
    Serial.print(F(" cycles ("));
    Serial.print(res.max_us, 4);
    Serial.println(F(" us)"));
    Serial.print(F(" Avg Latency:        "));
    Serial.print(res.avg_cycles, 2);
    Serial.print(F(" cycles ("));
    Serial.print(res.avg_us, 4);
    Serial.println(F(" us)"));
    Serial.print(F(" StdDev Latency:     "));
    Serial.print(res.stddev_cycles, 2);
    Serial.println(F(" cycles"));
    Serial.print(F(" Kernel Checksum:    0x"));
    Serial.println(res.checksum, HEX);
    Serial.println(F("================================================================"));

    // 5. Machine-Readable Single-Line JSON Output
    Serial.print(F("BENCHMARK_JSON={\"name\":\""));
    Serial.print(res.name);
    Serial.print(F("\",\"iterations\":"));
    Serial.print(res.iterations);
    Serial.print(F(",\"min_cycles\":"));
    Serial.print(res.min_cycles);
    Serial.print(F(",\"max_cycles\":"));
    Serial.print(res.max_cycles);
    Serial.print(F(",\"avg_cycles\":"));
    Serial.print(res.avg_cycles, 2);
    Serial.print(F(",\"stddev_cycles\":"));
    Serial.print(res.stddev_cycles, 2);
    Serial.print(F(",\"cpu_hz\":"));
    Serial.print(res.cpu_hz);
    Serial.print(F(",\"min_us\":"));
    Serial.print(res.min_us, 4);
    Serial.print(F(",\"max_us\":"));
    Serial.print(res.max_us, 4);
    Serial.print(F(",\"avg_us\":"));
    Serial.print(res.avg_us, 4);
    Serial.print(F(",\"checksum\":"));
    Serial.print(res.checksum);
    Serial.println(F("}"));

    Serial.println(F("CODE2EDGE_BENCHMARK_END"));
}

// ==============================================================================
// 3. Deterministic Validation Workload
// ==============================================================================

/**
 * A small deterministic CPU arithmetic kernel.
 * Mixes and hashes a 64-word array using 32-bit linear/nonlinear operations.
 * The result is written to volatile g_workload_checksum to prevent dead-code elimination.
 */
static void deterministic_kernel(void) {
    static uint32_t s_buffer[64];
    uint32_t acc = 0x6A09E667UL; // Initial seed constant

    for (size_t i = 0; i < 64; ++i) {
        s_buffer[i] = (uint32_t)(i * 2654435761UL + acc);
        acc ^= (s_buffer[i] << 5) | (s_buffer[i] >> 27);
        acc += 0x9E3779B9UL;
    }

    for (size_t i = 0; i < 64; ++i) {
        acc = (acc * 31UL) ^ s_buffer[i];
    }

    g_workload_checksum = acc;
}

// ==============================================================================
// 4. Arduino Entry Points
// ==============================================================================

void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    delay(800);

    Serial.println(F("================================================================"));
    Serial.println(F(" Code2Edge Target Benchmark Harness"));
    Serial.println(F(" Target: STM32U585 MCU on Arduino UNO Q"));
    Serial.println(F("================================================================"));

    bool dwt_ok = dwt_init();
    if (!dwt_ok) {
        Serial.println(F("FATAL: Failed to initialize DWT cycle counter!"));
        while (1) {
            delay(1000);
        }
    }
    Serial.println(F("DWT cycle counter initialized successfully."));
    Serial.println(F("Running baseline validation benchmark..."));
    Serial.println();

    // Execute standard 50-iteration benchmark
    benchmark_target("deterministic_kernel", deterministic_kernel, 50, 5);
}

void loop() {
    // Interactive trigger: send 'B' or 'R' over serial to re-run benchmark
    if (Serial.available() > 0) {
        char ch = (char)Serial.read();
        if (ch == 'B' || ch == 'b' || ch == 'R' || ch == 'r') {
            Serial.println();
            Serial.println(F("Triggering re-run of benchmark harness..."));
            benchmark_target("deterministic_kernel", deterministic_kernel, 50, 5);
        }
    }
    delay(100);
}
