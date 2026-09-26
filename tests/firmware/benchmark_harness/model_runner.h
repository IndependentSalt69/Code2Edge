/*
 * Code2Edge: STM32U585 DS-CNN Model Runner Header
 *
 * Target:    Arduino UNO Q (STM32U585 ARM Cortex-M33 @ 160 MHz)
 * Core:      arduino:zephyr (1.0.0)
 * Runtime:   Zero-heap INT8 Quantized DS-CNN Engine with DWT_CYCCNT timing
 */

#ifndef CODE2EDGE_MODEL_RUNNER_H
#define CODE2EDGE_MODEL_RUNNER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define MODEL_INPUT_CHANNELS    1
#define MODEL_INPUT_FREQ_BINS   64
#define MODEL_INPUT_TIME_STEPS  101
#define MODEL_INPUT_SIZE        (MODEL_INPUT_CHANNELS * MODEL_INPUT_FREQ_BINS * MODEL_INPUT_TIME_STEPS) // 6464

#define MODEL_NUM_CLASSES       12
#define MODEL_INPUT_SCALE       0.018517991527915f
#define MODEL_INPUT_ZERO_POINT  (-51)

#define MODEL_OUTPUT_SCALE      0.049993276596069336f
#define MODEL_OUTPUT_ZERO_POINT 3

#ifdef __cplusplus
extern "C" {
#endif

extern const char* const kModelLabels[MODEL_NUM_CLASSES];

struct ModelInferenceResult {
    int8_t output_int8[MODEL_NUM_CLASSES];
    float output_dequantized[MODEL_NUM_CLASSES];
    int predicted_index;
    const char *predicted_label;
    uint32_t inference_cycles;
    float inference_us;
};

/**
 * Initialize model runner, verify FlatBuffer header and map Flash weight pointers.
 * Zero dynamic memory allocation.
 */
bool model_runner_init(void);

/**
 * Quantize float32 [1, 1, 64, 101] input features to int8.
 * Formula: q = clamp(round(x / scale) + zero_point, -128, 127)
 */
void model_runner_quantize_input(const float *features_f32, int8_t *features_int8);

/**
 * Execute quantized DS-CNN forward pass on int8 input.
 * Measures inference cycles using DWT_CYCCNT.
 */
bool model_runner_invoke(const int8_t *input_int8, struct ModelInferenceResult *result_out);

/**
 * End-to-end inference directly on float32 preprocessing features.
 */
bool model_runner_run(const float *features_f32, struct ModelInferenceResult *result_out);

/**
 * Returns tensor arena static memory footprint in bytes.
 */
size_t model_runner_get_arena_size(void);

#ifdef __cplusplus
}
#endif

#endif // CODE2EDGE_MODEL_RUNNER_H
