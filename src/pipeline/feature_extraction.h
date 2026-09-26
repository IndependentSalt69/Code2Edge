/*
 * Code2Edge Keyword Spotting Preprocessing Pipeline
 * Generated from reference/tiny-kws/src/common.py
 *
 * Target: ARM Cortex-M33 (STM32U585)
 * Conforms to: contracts/target/target-profile.json
 */

#ifndef CODE2EDGE_FEATURE_EXTRACTION_H
#define CODE2EDGE_FEATURE_EXTRACTION_H

#include <stdint.h>
#include <stddef.h>

#define AUDIO_SAMPLE_RATE     16000
#define AUDIO_CLIP_SAMPLES    16000
#define PREPROC_N_FFT         400
#define PREPROC_HOP_LENGTH    160
#define PREPROC_N_MELS        64
#define PREPROC_N_FRAMES      101
#define PREPROC_N_FFT_BINS    201
#define PREPROC_OUTPUT_SIZE   (PREPROC_N_MELS * PREPROC_N_FRAMES) // 6464 int8_t values

#define NORM_MEAN             (-6.9023604393005371f)
#define NORM_STD              (4.8172130584716797f)
#define LOG_EPS               (1e-06f)
#define FEATURE_INPUT_SCALE      (0.018517991527915f)
#define FEATURE_INPUT_ZERO_POINT (-51)
#define FEATURE_INPUT_MIN        (-128)
#define FEATURE_INPUT_MAX        (127)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize preprocessing tables (no-op since tables are resident in Flash .rodata).
 */
void feature_extraction_init(void);

/**
 * Execute STFT -> Mel -> Log -> Normalize on a single frame of 400 windowed samples.
 */
void feature_extraction_compute_frame(const float *frame_windowed, float *power_spec_out, float *mel_energies_out);

/**
 * Execute full preprocessing directly on normalized float32 audio samples in [-1.0, 1.0].
 */
void feature_extraction_run_f32(const float *audio_pcm_f32_16k, float *mel_features_out);

/**
 * Stage-by-stage execution on float32 audio for differential parity verification.
 * Any unused output buffer pointer can be NULL.
 */
void feature_extraction_stages_f32(
    const float *audio_pcm_f32_16k,
    float *s0_input_out,
    float *s1_power_out,
    float *s1_mel_out,
    float *s2_log_out,
    float *s3_norm_out
);

/**
 * Execute STFT -> Mel -> Log -> Normalize on 16 kHz 16-bit mono PCM audio.
 *
 * @param audio_pcm_16k Pointer to 16,000 int16_t PCM audio samples (1.0 sec @ 16 kHz)
 * @param mel_features_out Output buffer for 6,464 int8_t quantized normalized log-mel features.
 *                         Row-major layout: (64 mels, 101 frames) -> [mel_idx * 101 + frame_idx]
 */
void feature_extraction_run(const int16_t *audio_pcm_16k, int8_t *mel_features_out);

/**
 * Compute checksum (FNV-1a hash) over the generated float32 output feature array.
 */
uint32_t feature_extraction_compute_checksum(const float *features, size_t count);

#ifdef __cplusplus
}
#endif

#endif // CODE2EDGE_FEATURE_EXTRACTION_H
