#!/usr/bin/env python3
"""
Generate C implementation and test fixtures for Code2Edge preprocessing (STFT -> Mel -> Log -> Normalize).
Adheres strictly to reference/tiny-kws/src/common.py and contracts/target/input-tensor.json.
"""

import math
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
LOG_EPS = 1e-6
N_FRAMES = 101
N_FFT_BINS = N_FFT // 2 + 1  # 201

# Normalization constants from checkpoints/best.pt
NORM_MEAN = -6.9023613929748535
NORM_STD = 4.81721305847168


def generate_tables():
    # 1. Periodic Hann Window
    n = np.arange(N_FFT)
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)).astype(np.float32)

    # 2. Twiddle tables for 400 points
    cos_table = np.cos(2 * np.pi * n / N_FFT).astype(np.float32)
    sin_table = np.sin(2 * np.pi * n / N_FFT).astype(np.float32)

    # 3. Mel Filterbank (HTK scale)
    m_min = 2595.0 * np.log10(1.0 + F_MIN / 700.0)
    m_max = 2595.0 * np.log10(1.0 + F_MAX / 700.0)
    m_pts = np.linspace(m_min, m_max, N_MELS + 2)
    f_pts = 700.0 * (10.0**(m_pts / 2595.0) - 1.0)
    freq_bins = np.linspace(0, SAMPLE_RATE // 2, N_FFT_BINS)

    mel_bands = []
    flat_weights = []

    for i in range(N_MELS):
        left, center, right = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        weights = []
        start_k = None
        end_k = None
        for k in range(N_FFT_BINS):
            f = freq_bins[k]
            w = 0.0
            if left <= f <= center and center > left:
                w = (f - left) / (center - left)
            elif center < f <= right and right > center:
                w = (right - f) / (right - center)
            if w > 0.0:
                if start_k is None:
                    start_k = k
                end_k = k
                weights.append(float(w))

        if start_k is None:
            start_k = 0
            num_bins = 0
            offset = len(flat_weights)
        else:
            num_bins = len(weights)
            offset = len(flat_weights)
            flat_weights.extend(weights)

        mel_bands.append({
            "start_bin": start_k,
            "num_bins": num_bins,
            "weight_offset": offset
        })

    return hann, cos_table, sin_table, mel_bands, flat_weights


def write_header_and_c():
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
#define PREPROC_OUTPUT_SIZE   (PREPROC_N_MELS * PREPROC_N_FRAMES) // {N_MELS * N_FRAMES} floats

#define NORM_MEAN             ({NORM_MEAN:.16f}f)
#define NORM_STD              ({NORM_STD:.16f}f)
#define LOG_EPS               ({LOG_EPS}f)

#ifdef __cplusplus
extern "C" {{
#endif

/**
 * Initialize preprocessing tables (no-op since tables are resident in Flash .rodata).
 */
void feature_extraction_init(void);

/**
 * Execute STFT -> Mel -> Log -> Normalize on 16 kHz 16-bit mono PCM audio.
 *
 * @param audio_pcm_16k Pointer to 16,000 int16_t PCM audio samples (1.0 sec @ 16 kHz)
 * @param mel_features_out Output buffer for 6,464 float32 normalized log-mel features.
 *                         Row-major layout: (64 mels, 101 frames) -> [mel_idx * 101 + frame_idx]
 */
void feature_extraction_run(const int16_t *audio_pcm_16k, float *mel_features_out);

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

static inline int16_t get_padded_sample(const int16_t *audio, int idx) {
    if (idx < 0) {
        idx = -idx;
    } else if (idx >= AUDIO_CLIP_SAMPLES) {
        idx = 2 * (AUDIO_CLIP_SAMPLES - 1) - idx;
    }
    return audio[idx];
}

void feature_extraction_run(const int16_t *audio_pcm_16k, float *mel_features_out) {
    static float frame_windowed[PREPROC_N_FFT];
    static float power_spec[PREPROC_N_FFT_BINS];

    for (int t = 0; t < PREPROC_N_FRAMES; ++t) {
        // 1. Frame Windowing with Reflect Padding
        int frame_start = t * PREPROC_HOP_LENGTH - (PREPROC_N_FFT / 2);
        for (int n = 0; n < PREPROC_N_FFT; ++n) {
            int16_t raw_pcm = get_padded_sample(audio_pcm_16k, frame_start + n);
            float sample_f = ((float)raw_pcm) / 32768.0f;
            frame_windowed[n] = sample_f * g_hann_window[n];
        }

        // 2. Real Discrete Fourier Transform (Power Spectrum)
        for (int k = 0; k < PREPROC_N_FFT_BINS; ++k) {
            float re = 0.0f;
            float im = 0.0f;
            for (int n = 0; n < PREPROC_N_FFT; ++n) {
                int twiddle_idx = (k * n) % PREPROC_N_FFT;
                float val = frame_windowed[n];
                re += val * g_cos_400[twiddle_idx];
                im -= val * g_sin_400[twiddle_idx];
            }
            power_spec[k] = re * re + im * im;
        }

        // 3. Mel Filterbank Multiplication + Log Compression + Normalization
        for (int m = 0; m < PREPROC_N_MELS; ++m) {
            const MelBandInfo *band = &g_mel_bands[m];
            float mel_energy = 0.0f;
            for (int b = 0; b < band->num_bins; ++b) {
                int bin_k = band->start_bin + b;
                mel_energy += power_spec[bin_k] * g_mel_weights[band->weight_offset + b];
            }

            float log_val = logf(mel_energy + LOG_EPS);
            float norm_val = (log_val - NORM_MEAN) / NORM_STD;

            // Store in row-major layout: (64 mels, 101 frames) -> [m * 101 + t]
            mel_features_out[m * PREPROC_N_FRAMES + t] = norm_val;
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

    with open(PIPELINE_DIR / "feature_extraction.c", "w", encoding="utf-8") as f:
        f.write("\n".join(c_content))


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
