"""
Unit tests for Code2Edge Preprocessing Differential Parity against Frozen PyTorch Artifacts.
Executes the actual compiled C pipeline (src/pipeline/feature_extraction.c) on host
and validates stages S0 (Raw Input), S1 (Power Spectrum), S1_mel (Mel Energy), S2 (Log-Mel),
and S3 (Normalized Features) against frozen reference artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from tools.run_host_parity import (
    NativeCPipeline,
    compare_stage_tensors,
    evaluate_sample_parity,
    run_host_parity_suite,
    STAGE_TOLERANCES,
    REPO_ROOT,
    SMOKE_SAMPLE_DIR,
)


@pytest.fixture(scope="module")
def native_c_pipeline():
    """Provides a singleton compiled C pipeline instance."""
    return NativeCPipeline()


def test_frozen_reference_artifacts_exist():
    """Verify that the immutable frozen reference artifacts exist on disk."""
    assert SMOKE_SAMPLE_DIR.exists(), f"Missing sample artifact dir: {SMOKE_SAMPLE_DIR}"
    assert (SMOKE_SAMPLE_DIR / "post_input.npy").exists(), "Missing S0 post_input.npy"
    assert (SMOKE_SAMPLE_DIR / "post_power_spectrum.npy").exists(), "Missing S1 post_power_spectrum.npy"
    assert (SMOKE_SAMPLE_DIR / "post_mel.npy").exists(), "Missing S1_mel post_mel.npy"
    assert (SMOKE_SAMPLE_DIR / "post_log.npy").exists(), "Missing S2 post_log.npy"
    assert (SMOKE_SAMPLE_DIR / "post_normalize.npy").exists(), "Missing S3 post_normalize.npy"
    assert (SMOKE_SAMPLE_DIR / "window.npy").exists(), "Missing window.npy"
    assert (SMOKE_SAMPLE_DIR / "mel_filterbank.npy").exists(), "Missing mel_filterbank.npy"
    assert (REPO_ROOT / "reference" / "normalization.json").exists(), "Missing normalization.json"


def test_stage_s0_raw_waveform_parity(native_c_pipeline):
    """Verify Stage S0 (Raw audio waveform) matches ground truth."""
    ref_input = np.load(SMOKE_SAMPLE_DIR / "post_input.npy")
    assert ref_input.shape == (16000,), f"Expected shape (16000,), got {ref_input.shape}"
    assert ref_input.dtype == np.float32, f"Expected float32, got {ref_input.dtype}"
    assert np.isfinite(ref_input).all(), "Found non-finite values in S0 input"

    stages = native_c_pipeline.run_stages(ref_input)
    result = compare_stage_tensors(
        "S0_raw_waveform",
        ref_input,
        stages["S0_raw_waveform"],
        STAGE_TOLERANCES["S0_raw_waveform"],
    )
    assert result["status"] == "PASS", f"Stage S0 failed parity: {result}"
    assert result["max_abs_diff"] <= STAGE_TOLERANCES["S0_raw_waveform"]["max_abs_diff"]


def test_stage_s1_power_spectrum_parity(native_c_pipeline):
    """Verify Stage S1 (Power Spectrum STFT) matches ground truth within tolerance."""
    ref_input = np.load(SMOKE_SAMPLE_DIR / "post_input.npy")
    ref_power = np.load(SMOKE_SAMPLE_DIR / "post_power_spectrum.npy")

    stages = native_c_pipeline.run_stages(ref_input)
    result = compare_stage_tensors(
        "S1_power_spectrum",
        ref_power,
        stages["S1_power_spectrum"],
        STAGE_TOLERANCES["S1_power_spectrum"],
    )

    assert result["status"] == "PASS", f"Stage S1 failed parity: {result}"
    assert result["max_abs_diff"] <= STAGE_TOLERANCES["S1_power_spectrum"]["max_abs_diff"]
    assert result["cosine_similarity"] >= STAGE_TOLERANCES["S1_power_spectrum"]["min_cosine"]


def test_stage_s1_mel_energy_parity(native_c_pipeline):
    """Verify Stage S1_mel (Mel Energy representation) matches ground truth within tolerance."""
    ref_input = np.load(SMOKE_SAMPLE_DIR / "post_input.npy")
    ref_mel = np.load(SMOKE_SAMPLE_DIR / "post_mel.npy")

    stages = native_c_pipeline.run_stages(ref_input)
    result = compare_stage_tensors(
        "S1_mel_energy",
        ref_mel,
        stages["S1_mel_energy"],
        STAGE_TOLERANCES["S1_mel_energy"],
    )

    assert result["status"] == "PASS", f"Stage S1_mel failed parity: {result}"
    assert result["max_abs_diff"] <= STAGE_TOLERANCES["S1_mel_energy"]["max_abs_diff"]
    assert result["cosine_similarity"] >= STAGE_TOLERANCES["S1_mel_energy"]["min_cosine"]


def test_stage_s2_log_mel_parity(native_c_pipeline):
    """Verify Stage S2 (Log-Mel features) matches ground truth within tolerance."""
    ref_input = np.load(SMOKE_SAMPLE_DIR / "post_input.npy")
    ref_log = np.load(SMOKE_SAMPLE_DIR / "post_log.npy")

    stages = native_c_pipeline.run_stages(ref_input)
    result = compare_stage_tensors(
        "S2_log_mel",
        ref_log,
        stages["S2_log_mel"],
        STAGE_TOLERANCES["S2_log_mel"],
    )

    assert result["status"] == "PASS", f"Stage S2 failed parity: {result}"
    assert result["max_abs_diff"] <= STAGE_TOLERANCES["S2_log_mel"]["max_abs_diff"]
    assert result["cosine_similarity"] >= STAGE_TOLERANCES["S2_log_mel"]["min_cosine"]


def test_stage_s3_normalized_features_parity(native_c_pipeline):
    """Verify Stage S3 (Normalized features) matches ground truth within tolerance."""
    ref_input = np.load(SMOKE_SAMPLE_DIR / "post_input.npy")
    ref_norm = np.load(SMOKE_SAMPLE_DIR / "post_normalize.npy")

    stages = native_c_pipeline.run_stages(ref_input)
    result = compare_stage_tensors(
        "S3_normalized_features",
        ref_norm,
        stages["S3_normalized_features"],
        STAGE_TOLERANCES["S3_normalized_features"],
    )

    assert result["status"] == "PASS", f"Stage S3 failed parity: {result}"
    assert result["max_abs_diff"] <= STAGE_TOLERANCES["S3_normalized_features"]["max_abs_diff"]
    assert result["cosine_similarity"] >= STAGE_TOLERANCES["S3_normalized_features"]["min_cosine"]


def test_full_host_parity_report_generation():
    """Execute end-to-end host parity verification and assert report emission."""
    report = run_host_parity_suite(sample_dir=SMOKE_SAMPLE_DIR)
    assert report["summary"]["overall_status"] == "PASS", f"Host parity report failed: {report}"
    assert report["summary"]["passed_stages"] == 5
    assert report["summary"]["failed_stages"] == 0
    assert report["coverage"]["parity_status_smoke"] == "PASS"
    assert report["coverage"]["parity_status_full_corpus"] == "PENDING"
