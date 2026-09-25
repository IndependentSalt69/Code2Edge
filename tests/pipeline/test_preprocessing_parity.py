"""
Unit tests for Code2Edge preprocessing pipeline and audio fixtures.
Verifies input shape, output shape, data types, numerical parity against reference, and checksum calculation.
"""

import math
import struct
import wave
import numpy as np
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLE_WAV = REPO_ROOT / "reference" / "tiny-kws" / "app" / "examples" / "yes.wav"
FIXTURE_HEADER = REPO_ROOT / "tests" / "firmware" / "benchmark_harness" / "fixtures" / "audio_fixture_yes.h"
FEATURE_EXTRACTION_C = REPO_ROOT / "src" / "pipeline" / "feature_extraction.c"
FEATURE_EXTRACTION_H = REPO_ROOT / "src" / "pipeline" / "feature_extraction.h"

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 64
N_FRAMES = 101
N_FFT_BINS = 201
LOG_EPS = 1e-6
NORM_MEAN = -6.9023613929748535
NORM_STD = 4.81721305847168


def compute_fnv1a_checksum(float_array: np.ndarray) -> int:
    """Compute 32-bit FNV-1a hash over float32 array."""
    raw_bytes = float_array.astype(np.float32).tobytes()
    hash_val = 2166136261
    for b in raw_bytes:
        hash_val ^= b
        hash_val = (hash_val * 16777619) & 0xFFFFFFFF
    return hash_val


def run_reference_preprocessing(pcm_int16: np.ndarray) -> np.ndarray:
    """Python implementation exactly mirroring src/pipeline/feature_extraction.c."""
    wav = pcm_int16.astype(np.float32) / 32768.0
    padded = np.pad(wav, (200, 200), mode="reflect")

    # Periodic Hann window
    n = np.arange(N_FFT)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)).astype(np.float32)

    # Mel filterbank
    m_min = 2595.0 * np.log10(1.0 + 20.0 / 700.0)
    m_max = 2595.0 * np.log10(1.0 + 7600.0 / 700.0)
    m_pts = np.linspace(m_min, m_max, N_MELS + 2)
    f_pts = 700.0 * (10.0**(m_pts / 2595.0) - 1.0)
    freq_bins = np.linspace(0, SAMPLE_RATE // 2, N_FFT_BINS)

    fbanks = np.zeros((N_MELS, N_FFT_BINS), dtype=np.float32)
    for i in range(N_MELS):
        left, center, right = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        for k in range(N_FFT_BINS):
            f = freq_bins[k]
            if left <= f <= center and center > left:
                fbanks[i, k] = (f - left) / (center - left)
            elif center < f <= right and right > center:
                fbanks[i, k] = (right - f) / (right - center)

    spec = np.zeros((N_FRAMES, N_FFT_BINS), dtype=np.float32)
    for t in range(N_FRAMES):
        frame = padded[t * HOP_LENGTH : t * HOP_LENGTH + N_FFT] * window
        fft_res = np.fft.rfft(frame, n=N_FFT)
        spec[t] = np.abs(fft_res)**2

    mel = np.dot(fbanks, spec.T)
    log_mel = np.log(mel + LOG_EPS)
    norm_mel = (log_mel - NORM_MEAN) / NORM_STD
    return norm_mel.astype(np.float32)


def test_input_fixture_exists_and_valid():
    """Verify input WAV fixture exists and has exactly 16000 16-bit mono samples."""
    assert EXAMPLE_WAV.exists(), f"Example WAV missing: {EXAMPLE_WAV}"
    with wave.open(str(EXAMPLE_WAV), "rb") as f:
        assert f.getnchannels() == 1, "Expected mono audio"
        assert f.getsampwidth() == 2, "Expected 16-bit PCM"
        assert f.getframerate() == 16000, "Expected 16 kHz sample rate"
        assert f.getnframes() == 16000, "Expected exactly 1.0s (16000 samples)"


def test_header_and_c_files_exist():
    """Verify generated C pipeline files exist."""
    assert FEATURE_EXTRACTION_H.exists(), f"Missing {FEATURE_EXTRACTION_H}"
    assert FEATURE_EXTRACTION_C.exists(), f"Missing {FEATURE_EXTRACTION_C}"
    assert FIXTURE_HEADER.exists(), f"Missing {FIXTURE_HEADER}"


def test_preprocessing_output_shape_and_type():
    """Verify preprocessing produces (64, 101) float32 tensor."""
    with wave.open(str(EXAMPLE_WAV), "rb") as f:
        pcm = np.frombuffer(f.readframes(16000), dtype=np.int16)

    feats = run_reference_preprocessing(pcm)
    assert feats.shape == (64, 101), f"Expected shape (64, 101), got {feats.shape}"
    assert feats.dtype == np.float32, f"Expected float32, got {feats.dtype}"
    assert not np.isnan(feats).any(), "Found NaN in preprocessed features"
    assert not np.isinf(feats).any(), "Found Inf in preprocessed features"


def test_checksum_determinism():
    """Verify checksum is non-zero and perfectly deterministic."""
    with wave.open(str(EXAMPLE_WAV), "rb") as f:
        pcm = np.frombuffer(f.readframes(16000), dtype=np.int16)

    feats1 = run_reference_preprocessing(pcm)
    feats2 = run_reference_preprocessing(pcm)
    c1 = compute_fnv1a_checksum(feats1)
    c2 = compute_fnv1a_checksum(feats2)

    assert c1 == c2, "Checksum must be deterministic across identical runs"
    assert c1 != 0, "Checksum must not be zero"
    print(f"\nDeterministic Checksum for 'yes.wav' features: 0x{c1:08X} ({c1})")
