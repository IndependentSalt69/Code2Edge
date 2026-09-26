#!/usr/bin/env python3
"""
Generate C implementation and test fixtures for Code2Edge preprocessing (STFT -> Mel -> Log -> Normalize).
Adheres strictly to reference/tiny-kws/src/common.py and contracts/target/input-tensor.json.
"""

import math
import json
import wave
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = REPO_ROOT / "src" / "pipeline"
FIXTURES_DIR = REPO_ROOT / "tests" / "firmware" / "benchmark_harness" / "fixtures"

# ---------------------------------------------------------------------------
# Constants from frozen reference
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 64
F_MIN = 20.0
F_MAX = 7600.0
N_FRAMES = 101

N_FFT_BINS = N_FFT // 2 + 1  # 201


def load_reference_config():
    normalization_path = REPO_ROOT / "reference" / "normalization.json"
    model_validation_path = (
        REPO_ROOT / "evidence" / "model" / "model_artifact_validation.json"
    )

    normalization = json.loads(normalization_path.read_text())
    model_validation = json.loads(model_validation_path.read_text())
    input_tensor = model_validation["input_tensor"]

    if input_tensor["shape"] != [1, 1, 64, 101]:
        raise ValueError(f"Unexpected model input shape: {input_tensor['shape']}")
    if input_tensor["dtype"] != "INT8":
        raise ValueError(f"Unexpected model input dtype: {input_tensor['dtype']}")
    if len(input_tensor["scales"]) != 1 or len(input_tensor["zero_points"]) != 1:
        raise ValueError("Expected one tensorwise model input scale and zero point")

    return {
        "norm_mean": float(normalization["mean"]),
        "norm_std": float(normalization["std"]),
        "log_eps": float(normalization["log_eps"]),
        "input_scale": float(input_tensor["scales"][0]),
        "input_zero_point": int(input_tensor["zero_points"][0]),
    }


def generate_tables():
    # 1. Periodic Hann Window
    n = np.arange(N_FFT)
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)).astype(np.float32)

    # 2. Twiddle tables for 400 points
    cos_table = np.cos(2 * np.pi * n / N_FFT).astype(np.float32)
    sin_table = np.sin(2 * np.pi * n / N_FFT).astype(np.float32)

    # 3. Mel Filterbank: require the tracked frozen reference artifact.
    mel_fb_path = REPO_ROOT / "reference" / "artifacts" / "mel_filterbank.npy"
    if not mel_fb_path.exists():
        raise FileNotFoundError(
            f"Required frozen mel filterbank artifact missing: {mel_fb_path}"
        )

    mel_fb = np.load(str(mel_fb_path), allow_pickle=False).astype(np.float32)
    expected_mel_shape = (N_FFT_BINS, N_MELS)
    if mel_fb.shape != expected_mel_shape:
        raise ValueError(
            f"Frozen mel filterbank shape {mel_fb.shape} != {expected_mel_shape}"
        )

    mel_bands = []
    flat_weights = []

    for m in range(N_MELS):
        col = mel_fb[:, m]
        non_zeros = np.where(col > 0)[0]
        if len(non_zeros) == 0:
            mel_bands.append({"start_bin": 0, "num_bins": 0, "weight_offset": len(flat_weights)})
        else:
            start_k = int(non_zeros[0])
            num_bins = int(non_zeros[-1] - start_k + 1)
            offset = len(flat_weights)
            weights = [float(w) for w in col[start_k:start_k + num_bins]]
            flat_weights.extend(weights)
            mel_bands.append({
                "start_bin": start_k,
                "num_bins": num_bins,
                "weight_offset": offset
            })

    return hann, cos_table, sin_table, mel_bands, flat_weights


def write_header_and_c():
    config = load_reference_config()
    norm_mean = config["norm_mean"]
    norm_std = config["norm_std"]
    log_eps = config["log_eps"]
    input_scale = config["input_scale"]
    input_zero_point = config["input_zero_point"]

    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    hann, cos_table, sin_table, mel_bands, flat_weights = generate_tables()

    # 1. Header file
    header_content = f"""/*
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

#define AUDIO_SAMPLE_RATE     {SAMPLE_RATE}
#define AUDIO_CLIP_SAMPLES    {CLIP_SAMPLES}
#define PREPROC_N_FFT         {N_FFT}
#define PREPROC_HOP_LENGTH    {HOP_LENGTH}
#define PREPROC_N_MELS        {N_MELS}
#define PREPROC_N_FRAMES      {N_FRAMES}
#define PREPROC_N_FFT_BINS    {N_FFT_BINS}
#define PREPROC_OUTPUT_SIZE   (PREPROC_N_MELS * PREPROC_N_FRAMES) // {N_MELS * N_FRAMES} int8_t values

#define NORM_MEAN             ({norm_mean:.16f}f)
#define NORM_STD              ({norm_std:.16f}f)
#define LOG_EPS               ({log_eps}f)
#define FEATURE_INPUT_SCALE      ({input_scale:.15f}f)
#define FEATURE_INPUT_ZERO_POINT ({input_zero_point})
#define FEATURE_INPUT_MIN        (-128)
#define FEATURE_INPUT_MAX        (127)

#ifdef __cplusplus
extern "C" {{
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
}}
#endif

#endif // CODE2EDGE_FEATURE_EXTRACTION_H
"""

    with open(PIPELINE_DIR / "feature_extraction.h", "w", encoding="utf-8") as f:
        f.write(header_content)

    # 2. C Source File
    c_content = [
        '#include "feature_extraction.h"',
        '#include <math.h>',
        '#include <string.h>',
        '',
        '// ============================================================================',
        '// Pre-computed Tables in Flash (.rodata)',
        '// ============================================================================',
        '',
        f'// Periodic Hann Window (length {N_FFT})',
        f'static const float g_hann_window[{N_FFT}] = {{'
    ]

    for i in range(0, N_FFT, 8):
        chunk = ", ".join(f"{hann[j]:.8f}f" for j in range(i, min(i + 8, N_FFT)))
        c_content.append(f"    {chunk},")
    c_content.append("};\n")

    c_content.append(f'// 400-point Cosine Lookup Table')
    c_content.append(f'static const float g_cos_400[{N_FFT}] = {{')
    for i in range(0, N_FFT, 8):
        chunk = ", ".join(f"{cos_table[j]:.8f}f" for j in range(i, min(i + 8, N_FFT)))
        c_content.append(f"    {chunk},")
    c_content.append("};\n")

    c_content.append(f'// 400-point Sine Lookup Table')
    c_content.append(f'static const float g_sin_400[{N_FFT}] = {{')
    for i in range(0, N_FFT, 8):
        chunk = ", ".join(f"{sin_table[j]:.8f}f" for j in range(i, min(i + 8, N_FFT)))
        c_content.append(f"    {chunk},")
    c_content.append("};\n")

    c_content.append('typedef struct {')
    c_content.append('    uint16_t start_bin;')
    c_content.append('    uint16_t num_bins;')
    c_content.append('    uint16_t weight_offset;')
    c_content.append('} MelBandInfo;\n')

    c_content.append(f'static const MelBandInfo g_mel_bands[{N_MELS}] = {{')
    for b in mel_bands:
        c_content.append(f'    {{ {b["start_bin"]}, {b["num_bins"]}, {b["weight_offset"]} }},')
    c_content.append('};\n')

    c_content.append(f'static const float g_mel_weights[{len(flat_weights)}] = {{')
    for i in range(0, len(flat_weights), 8):
        chunk = ", ".join(f"{flat_weights[j]:.8f}f" for j in range(i, min(i + 8, len(flat_weights))))
        c_content.append(f"    {chunk},")
    c_content.append('};\n')

    c_content.append('''
// ============================================================================
// Core Preprocessing Functions
// ============================================================================

void feature_extraction_init(void) {
    // All tables statically allocated in .rodata
}

static inline int16_t get_padded_sample_i16(const int16_t *audio, int idx) {
    if (idx < 0) {
        idx = -idx;
    } else if (idx >= AUDIO_CLIP_SAMPLES) {
        idx = 2 * (AUDIO_CLIP_SAMPLES - 1) - idx;
    }
    return audio[idx];
}

static inline float get_padded_sample_f32(const float *audio, int idx) {
    if (idx < 0) {
        idx = -idx;
    } else if (idx >= AUDIO_CLIP_SAMPLES) {
        idx = 2 * (AUDIO_CLIP_SAMPLES - 1) - idx;
    }
    return audio[idx];
}

void feature_extraction_compute_frame(const float *frame_windowed, float *power_spec_out, float *mel_energies_out) {
    // 1. Real Discrete Fourier Transform (Power Spectrum) using 64-bit accumulators
    for (int k = 0; k < PREPROC_N_FFT_BINS; ++k) {
        double re_acc = 0.0;
        double im_acc = 0.0;
        for (int n = 0; n < PREPROC_N_FFT; ++n) {
            int twiddle_idx = (k * n) % PREPROC_N_FFT;
            double val = (double)frame_windowed[n];
            re_acc += val * (double)g_cos_400[twiddle_idx];
            im_acc -= val * (double)g_sin_400[twiddle_idx];
        }
        power_spec_out[k] = (float)(re_acc * re_acc + im_acc * im_acc);
    }

    // 2. Mel Filterbank Energy Accumulation
    for (int m = 0; m < PREPROC_N_MELS; ++m) {
        const MelBandInfo *band = &g_mel_bands[m];
        double mel_acc = 0.0;
        for (int b = 0; b < band->num_bins; ++b) {
            int bin_k = band->start_bin + b;
            mel_acc += (double)power_spec_out[bin_k] * (double)g_mel_weights[band->weight_offset + b];
        }
        mel_energies_out[m] = (float)mel_acc;
    }
}

void feature_extraction_stages_f32(
    const float *audio_pcm_f32_16k,
    float *s0_input_out,
    float *s1_power_out,
    float *s1_mel_out,
    float *s2_log_out,
    float *s3_norm_out
) {
    if (s0_input_out) {
        memcpy(s0_input_out, audio_pcm_f32_16k, AUDIO_CLIP_SAMPLES * sizeof(float));
    }

    static float frame_windowed[PREPROC_N_FFT];
    static float power_spec[PREPROC_N_FFT_BINS];
    static float mel_energies[PREPROC_N_MELS];

    for (int t = 0; t < PREPROC_N_FRAMES; ++t) {
        // 1. Frame Windowing with Reflect Padding
        int frame_start = t * PREPROC_HOP_LENGTH - (PREPROC_N_FFT / 2);
        for (int n = 0; n < PREPROC_N_FFT; ++n) {
            float sample_f = get_padded_sample_f32(audio_pcm_f32_16k, frame_start + n);
            frame_windowed[n] = sample_f * g_hann_window[n];
        }

        // 2. Compute Power Spectrum and Mel Energies
        feature_extraction_compute_frame(frame_windowed, power_spec, mel_energies);

        if (s1_power_out) {
            for (int k = 0; k < PREPROC_N_FFT_BINS; ++k) {
                s1_power_out[k * PREPROC_N_FRAMES + t] = power_spec[k];
            }
        }

        for (int m = 0; m < PREPROC_N_MELS; ++m) {
            if (s1_mel_out) {
                s1_mel_out[m * PREPROC_N_FRAMES + t] = mel_energies[m];
            }

            float log_val = logf(mel_energies[m] + LOG_EPS);
            if (s2_log_out) {
                s2_log_out[m * PREPROC_N_FRAMES + t] = log_val;
            }

            float norm_val = (log_val - NORM_MEAN) / NORM_STD;
            if (s3_norm_out) {
                s3_norm_out[m * PREPROC_N_FRAMES + t] = norm_val;
            }
        }
    }
}

void feature_extraction_run_f32(const float *audio_pcm_f32_16k, float *mel_features_out) {
    feature_extraction_stages_f32(audio_pcm_f32_16k, NULL, NULL, NULL, NULL, mel_features_out);
}

static int8_t quantize_feature_int8(float value) {
    // Round the float32 quotient before adding the integer zero point.
    float scaled = value / FEATURE_INPUT_SCALE;
    float lower = floorf(scaled);
    float frac = scaled - lower;
    int rounded;

    if (frac > 0.5f) {
        rounded = (int)lower + 1;
    } else if (frac < 0.5f) {
        rounded = (int)lower;
    } else {
        // Ties-to-even, matching NumPy's np.round behavior.
        int lower_i = (int)lower;
        rounded = (lower_i % 2 == 0) ? lower_i : (lower_i + 1);
    }

    rounded += FEATURE_INPUT_ZERO_POINT;

    if (rounded < FEATURE_INPUT_MIN) {
        rounded = FEATURE_INPUT_MIN;
    } else if (rounded > FEATURE_INPUT_MAX) {
        rounded = FEATURE_INPUT_MAX;
    }

    return (int8_t)rounded;
}

void feature_extraction_run(const int16_t *audio_pcm_16k, int8_t *mel_features_out) {
    static float frame_windowed[PREPROC_N_FFT];
    static float power_spec[PREPROC_N_FFT_BINS];
    static float mel_energies[PREPROC_N_MELS];

    for (int t = 0; t < PREPROC_N_FRAMES; ++t) {
        // 1. Frame Windowing with Reflect Padding
        int frame_start = t * PREPROC_HOP_LENGTH - (PREPROC_N_FFT / 2);
        for (int n = 0; n < PREPROC_N_FFT; ++n) {
            int16_t raw_pcm = get_padded_sample_i16(audio_pcm_16k, frame_start + n);
            float sample_f = ((float)raw_pcm) / 32768.0f;
            frame_windowed[n] = sample_f * g_hann_window[n];
        }

        // 2. Compute Power Spectrum and Mel Energies
        feature_extraction_compute_frame(frame_windowed, power_spec, mel_energies);

        // 3. Log Compression + Normalization
        for (int m = 0; m < PREPROC_N_MELS; ++m) {
            float log_val = logf(mel_energies[m] + LOG_EPS);
            float norm_val = (log_val - NORM_MEAN) / NORM_STD;

            // Store in row-major layout: (64 mels, 101 frames) -> [m * 101 + t]
            mel_features_out[m * PREPROC_N_FRAMES + t] = quantize_feature_int8(norm_val);
        }
    }
}

uint32_t feature_extraction_compute_checksum(const float *features, size_t count) {
    uint32_t hash = 2166136261UL; // FNV-1a offset basis
    const uint8_t *bytes = (const uint8_t*)features;
    size_t byte_count = count * sizeof(float);

    for (size_t i = 0; i < byte_count; ++i) {
        hash ^= bytes[i];
        hash *= 16777619UL; // FNV prime
    }
    return hash;
}
''')

    code_str = "\n".join(c_content)
    with open(PIPELINE_DIR / "feature_extraction.c", "w", encoding="utf-8") as f:
        f.write(code_str)


def write_audio_fixture():
    wav_path = REPO_ROOT / "reference" / "tiny-kws" / "app" / "examples" / "yes.wav"
    with wave.open(str(wav_path), "rb") as f:
        raw = f.readframes(f.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16)

    fixture_content = [
        '/* Auto-generated audio fixture from reference/tiny-kws/app/examples/yes.wav */',
        '#ifndef AUDIO_FIXTURE_YES_H',
        '#define AUDIO_FIXTURE_YES_H',
        '',
        '#include <stdint.h>',
        '',
        '#define AUDIO_FIXTURE_LABEL "yes"',
        '#define AUDIO_FIXTURE_LABEL_IDX 2',
        '#define AUDIO_FIXTURE_SAMPLES 16000',
        '',
        'alignas(16) static const int16_t g_audio_fixture_yes[16000] = {'
    ]

    for i in range(0, len(pcm), 12):
        chunk = ", ".join(str(pcm[j]) for j in range(i, min(i + 12, len(pcm))))
        fixture_content.append(f"    {chunk},")

    fixture_content.append('};\n')
    fixture_content.append('#endif // AUDIO_FIXTURE_YES_H\n')

    with open(FIXTURES_DIR / "audio_fixture_yes.h", "w", encoding="utf-8") as f:
        f.write("\n".join(fixture_content))


if __name__ == "__main__":
    write_header_and_c()
    write_audio_fixture()
    print("Successfully generated:")
    print(f"  - {PIPELINE_DIR / 'feature_extraction.h'}")
    print(f"  - {PIPELINE_DIR / 'feature_extraction.c'}")
    print(f"  - {FIXTURES_DIR / 'audio_fixture_yes.h'}")
