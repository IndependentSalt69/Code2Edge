/*
 * Code2Edge Target Hardware Benchmark Harness & DS-CNN Inference Runner
 *
 * Target:    Arduino UNO Q
 * MCU:       STMicroelectronics STM32U585 (ARM Cortex-M33 @ 160 MHz)
 * Core:      arduino:zephyr (1.0.0)
 * FQBN:      arduino:zephyr:unoq
 *
 * Purpose:
 * Standalone, zero-heap benchmark and inference harness using Cortex-M33 DWT cycle counting (DWT_CYCCNT).
 * Measures per-iteration CPU cycles, min/max/average/stddev, and calculates execution time in microseconds.
 *
 * Workloads:
 * 1. deterministic_kernel: Arithmetic timing validation
 * 2. mel_spectrogram: Keyword Spotting Preprocessing (STFT -> Mel -> Log -> Normalize)
 * 3. dscnn_inference: Full INT8 Quantized DS-CNN Model Inference (119k params)
 *
 * Interactive Serial Commands (115200 baud):
 * - 'I' / 'i' : Run full end-to-end DS-CNN inference (Preproc -> Quantize -> NN) & print detailed report
 * - 'B' / 'b' : Run all benchmarks (deterministic + mel_spectrogram + dscnn_inference)
 * - 'M' / 'm' : Run mel_spectrogram benchmark only
 * - 'N' / 'n' : Run dscnn_inference benchmark only
 * - 'D' / 'd' : Parity dump (streams checksum, tensor shape, boundary values)
 * - 'K' / 'k' : Run deterministic validation benchmark only
 */

#include <Arduino.h>
#include <math.h>

// Feature extraction, model runner, and test fixture includes
#include "feature_extraction.h"
#include "model_runner.h"
#include "model_data.h"
#include "fixtures/audio_fixture_yes.h"

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

// Output buffer for preprocessing features: (64 mels, 101 frames) = 6464 floats (25.25 KB)
alignas(16) static float s_mel_features_out[PREPROC_OUTPUT_SIZE];

// Model inference output result structure
static struct ModelInferenceResult s_model_result;

static volatile uint32_t g_workload_checksum = 0;

/**
 * Reusable benchmark runner.
 */
static void benchmark_target(const char* name, void (*fn)(), size_t iterations = 50, size_t warmup = 5) {
    if (!s_dwt_initialized) {
        Serial.println(F("ERROR: DWT cycle counter is not initialized or not supported on this platform."));
        return;
    }

    if (iterations > MAX_BENCHMARK_ITERATIONS) {
        iterations = MAX_BENCHMARK_ITERATIONS;
    }

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
    Serial.print(F(" Workload Checksum:  0x"));
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
// 3. Workloads
// ==============================================================================

/**
 * 1. Deterministic Validation Kernel (Infrastructure verification)
 */
static void deterministic_kernel(void) {
    static uint32_t s_buffer[64];
    uint32_t acc = 0x6A09E667UL;

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

/**
 * 2. Keyword Spotting Preprocessing Kernel (mel_spectrogram)
 * Executes full STFT -> Mel -> Log -> Normalize pipeline on g_audio_fixture_yes.
 */
static void mel_spectrogram_workload(void) {
    feature_extraction_run(g_audio_fixture_yes, s_mel_features_out);
    g_workload_checksum = feature_extraction_compute_checksum(s_mel_features_out, PREPROC_OUTPUT_SIZE);
}

/**
 * 3. Keyword Spotting Neural Network Inference Kernel (dscnn_inference)
 * Executes quantized INT8 forward pass on s_model_input_int8.
 * Measures DWT_CYCCNT timing around neural network inference ONLY.
 */
static void dscnn_inference_workload(void) {
    model_runner_run(s_mel_features_out, &s_model_result);

    // Compute simple checksum over output INT8 logits
    uint32_t chk = 0x243F6A88UL;
    for (int i = 0; i < MODEL_NUM_CLASSES; ++i) {
        chk = (chk * 33UL) ^ (uint8_t)s_model_result.output_int8[i];
    }
    g_workload_checksum = chk;
}

/**
 * Print detailed DS-CNN Model Inference Report.
 */
static void print_inference_report(void) {
    // 1. Run Preprocessing on compiled audio fixture
    feature_extraction_run(g_audio_fixture_yes, s_mel_features_out);

    // 2. Execute Model Inference directly on float32 preprocessing features
    // (Quantizes input into arena buffer and executes neural network with DWT_CYCCNT timing)
    model_runner_run(s_mel_features_out, &s_model_result);

    // Compute input INT8 min/max for reporting
    int8_t min_q = 127;
    int8_t max_q = -128;
    const float inv_scale = 1.0f / MODEL_INPUT_SCALE;
    const int32_t zp = MODEL_INPUT_ZERO_POINT;
    for (size_t i = 0; i < PREPROC_OUTPUT_SIZE; ++i) {
        int32_t q = (int32_t)roundf(s_mel_features_out[i] * inv_scale) + zp;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        if ((int8_t)q < min_q) min_q = (int8_t)q;
        if ((int8_t)q > max_q) max_q = (int8_t)q;
    }

    Serial.println(F("CODE2EDGE_INFERENCE_START"));
    Serial.println(F("================================================================"));
    Serial.println(F(" Code2Edge DS-CNN On-Device Inference Verification"));
    Serial.println(F(" Target MCU:         STMicroelectronics STM32U585 (ARM Cortex-M33)"));
    Serial.println(F(" Board FQBN:         arduino:zephyr:unoq"));
    Serial.print(F(" CPU Clock:          "));
    Serial.print(TARGET_CPU_HZ / 1000000.0, 1);
    Serial.println(F(" MHz"));
    Serial.print(F(" Audio Fixture:      "));
    Serial.println(F(AUDIO_FIXTURE_LABEL));
    Serial.print(F(" Static Arena SRAM:  "));
    Serial.print(model_runner_get_arena_size());
    Serial.print(F(" bytes ("));
    Serial.print((float)model_runner_get_arena_size() / 1024.0f, 2);
    Serial.println(F(" KB)"));
    Serial.println(F("----------------------------------------------------------------"));
    Serial.println(F(" INPUT TENSOR SPECIFICATION:"));
    Serial.println(F("   Shape:            [1, 1, 64, 101] (6464 elements)"));
    Serial.println(F("   Dtype:            INT8"));
    Serial.print(F("   Scale:            "));
    Serial.println(MODEL_INPUT_SCALE, 8);
    Serial.print(F("   Zero Point:       "));
    Serial.println(MODEL_INPUT_ZERO_POINT);
    Serial.print(F("   Min / Max Value:  ["));
    Serial.print(min_q);
    Serial.print(F(", "));
    Serial.print(max_q);
    Serial.println(F("]"));
    Serial.println(F("----------------------------------------------------------------"));
    Serial.println(F(" OUTPUT TENSOR SPECIFICATION:"));
    Serial.println(F("   Shape:            [1, 12] (12 classes)"));
    Serial.println(F("   Dtype:            INT8"));
    Serial.print(F("   Scale:            "));
    Serial.println(MODEL_OUTPUT_SCALE, 8);
    Serial.print(F("   Zero Point:       "));
    Serial.println(MODEL_OUTPUT_ZERO_POINT);
    Serial.println(F("----------------------------------------------------------------"));
    
    // Output INT8 values
    Serial.print(F(" OUTPUT INT8 LOGITS: ["));
    for (int i = 0; i < MODEL_NUM_CLASSES; ++i) {
        Serial.print(s_model_result.output_int8[i]);
        if (i < MODEL_NUM_CLASSES - 1) Serial.print(F(", "));
    }
    Serial.println(F("]"));

    // Output Dequantized values
    Serial.print(F(" DEQUANTIZED VALUES: ["));
    for (int i = 0; i < MODEL_NUM_CLASSES; ++i) {
        Serial.print(s_model_result.output_dequantized[i], 4);
        if (i < MODEL_NUM_CLASSES - 1) Serial.print(F(", "));
    }
    Serial.println(F("]"));

    Serial.println(F("----------------------------------------------------------------"));
    Serial.print(F(" PREDICTED ARGMAX:   "));
    Serial.print(s_model_result.predicted_index);
    Serial.print(F(" (\""));
    Serial.print(s_model_result.predicted_label);
    Serial.println(F("\")"));

    Serial.print(F(" INFERENCE CYCLES:   "));
    Serial.print(s_model_result.inference_cycles);
    Serial.print(F(" cycles ("));
    Serial.print(s_model_result.inference_us, 2);
    Serial.println(F(" us)"));
    Serial.println(F("================================================================"));

    // Machine readable JSON line
    Serial.print(F("INFERENCE_JSON={\"fixture\":\""));
    Serial.print(F(AUDIO_FIXTURE_LABEL));
    Serial.print(F("\",\"input_min\":"));
    Serial.print(min_q);
    Serial.print(F(",\"input_max\":"));
    Serial.print(max_q);
    Serial.print(F(",\"predicted_index\":"));
    Serial.print(s_model_result.predicted_index);
    Serial.print(F(",\"predicted_label\":\""));
    Serial.print(s_model_result.predicted_label);
    Serial.print(F("\",\"inference_cycles\":"));
    Serial.print(s_model_result.inference_cycles);
    Serial.print(F(",\"inference_us\":"));
    Serial.print(s_model_result.inference_us, 2);
    Serial.print(F(",\"arena_bytes\":"));
    Serial.print(model_runner_get_arena_size());
    Serial.print(F(",\"logits\":["));
    for (int i = 0; i < MODEL_NUM_CLASSES; ++i) {
        Serial.print(s_model_result.output_int8[i]);
        if (i < MODEL_NUM_CLASSES - 1) Serial.print(F(","));
    }
    Serial.print(F("],\"dequantized\":["));
    for (int i = 0; i < MODEL_NUM_CLASSES; ++i) {
        Serial.print(s_model_result.output_dequantized[i], 4);
        if (i < MODEL_NUM_CLASSES - 1) Serial.print(F(","));
    }
    Serial.println(F("]}"));

    Serial.println(F("CODE2EDGE_INFERENCE_END"));
}

/**
 * Host/Device Parity Verification Hook:
 * Dumps preprocessed features over UART for differential comparison against PyTorch golden tensors.
 */
static void dump_parity_features(void) {
    feature_extraction_run(g_audio_fixture_yes, s_mel_features_out);
    g_workload_checksum = feature_extraction_compute_checksum(s_mel_features_out, PREPROC_OUTPUT_SIZE);

    Serial.println(F("CODE2EDGE_PARITY_DUMP_START"));
    Serial.print(F("FIXTURE_LABEL="));
    Serial.println(F(AUDIO_FIXTURE_LABEL));
    Serial.print(F("FIXTURE_CHECKSUM=0x"));
    Serial.println(g_workload_checksum, HEX);
    Serial.println(F("TENSOR_SHAPE=[1,1,64,101]"));
    Serial.println(F("TENSOR_SIZE=6464"));
    
    // Output sample points (first 5 and last 5 elements)
    Serial.print(F("HEAD_VALUES=["));
    for (int i = 0; i < 5; ++i) {
        Serial.print(s_mel_features_out[i], 6);
        if (i < 4) Serial.print(F(","));
    }
    Serial.println(F("]"));

    Serial.print(F("TAIL_VALUES=["));
    for (int i = PREPROC_OUTPUT_SIZE - 5; i < PREPROC_OUTPUT_SIZE; ++i) {
        Serial.print(s_mel_features_out[i], 6);
        if (i < PREPROC_OUTPUT_SIZE - 1) Serial.print(F(","));
    }
    Serial.println(F("]"));

    // Full tensor dump in IEEE-754 hex format for lossless numerical parity verification
    Serial.println(F("CODE2EDGE_PARITY_TENSOR_HEX_START"));
    const uint32_t *u32_view = (const uint32_t*)s_mel_features_out;
    for (size_t i = 0; i < PREPROC_OUTPUT_SIZE; ++i) {
        uint32_t v = u32_view[i];
        for (int b = 28; b >= 0; b -= 4) {
            uint8_t nibble = (uint8_t)((v >> b) & 0x0F);
            Serial.print(nibble < 10 ? (char)('0' + nibble) : (char)('A' + nibble - 10));
        }
        if ((i + 1) % 16 == 0 || i == PREPROC_OUTPUT_SIZE - 1) {
            Serial.println();
        } else {
            Serial.print(' ');
        }
    }
    Serial.println(F("CODE2EDGE_PARITY_TENSOR_HEX_END"));

    Serial.println(F("CODE2EDGE_PARITY_DUMP_END"));
}

// ==============================================================================
// 4. Arduino Entry Points
// ==============================================================================

void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    delay(500);

    bool dwt_ok = dwt_init();
    if (!dwt_ok) {
        Serial.println(F("FATAL: Failed to initialize DWT cycle counter!"));
        while (1) {
            delay(1000);
        }
    }
    
    // Initialize feature extraction tables and model runner
    feature_extraction_init();
    bool model_ok = model_runner_init();
    if (!model_ok) {
        Serial.println(F("FATAL: Failed to initialize DS-CNN model runner!"));
    }

    // Prepare input features once for benchmarking
    feature_extraction_run(g_audio_fixture_yes, s_mel_features_out);
}

void loop() {
    if (Serial.available() > 0) {
        int c = Serial.read();
        if (c == 'I' || c == 'i') {
            print_inference_report();
            Serial.flush();
        } else if (c == 'B' || c == 'b' || c == 'R' || c == 'r') {
            Serial.println(F("\n--- Triggering All Benchmarks ---"));
            benchmark_target("deterministic_kernel", deterministic_kernel, 50, 5);
            benchmark_target("mel_spectrogram", mel_spectrogram_workload, 10, 2);
            benchmark_target("dscnn_inference", dscnn_inference_workload, 10, 2);
            Serial.flush();
        } else if (c == 'M' || c == 'm' || c == 'P' || c == 'p') {
            Serial.println(F("\n--- Triggering mel_spectrogram Benchmark ---"));
            benchmark_target("mel_spectrogram", mel_spectrogram_workload, 10, 2);
            Serial.flush();
        } else if (c == 'N' || c == 'n') {
            Serial.println(F("\n--- Triggering dscnn_inference Benchmark ---"));
            benchmark_target("dscnn_inference", dscnn_inference_workload, 10, 2);
            Serial.flush();
        } else if (c == 'D' || c == 'd') {
            dump_parity_features();
            Serial.flush();
        } else if (c == 'K' || c == 'k') {
            Serial.println(F("\n--- Triggering deterministic_kernel Benchmark ---"));
            benchmark_target("deterministic_kernel", deterministic_kernel, 50, 5);
            Serial.flush();
        }
    }
    delay(20);
}
