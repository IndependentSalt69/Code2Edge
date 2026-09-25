"""
mcp_server/scenarios.py

Parity scenario engine for mock mode.

Controls behaviour via:
    CODE2EDGE_MOCK_PARITY_SCENARIO = pass | fail_then_pass | always_fail |
                                     fail_mel_scale | fail_framing_shape

fail_then_pass: attempt 1 → FAIL (fail_framing_shape numbers), attempt 2 → PASS.
  Attempt tracking is per run_id stored in a module-level dict (resets on process
  restart; good enough for demo/test).

All numeric divergence values are realistic for a DS-CNN KWS pipeline on a
Cortex-M33 implementation with a deliberate bug injected.
"""
from __future__ import annotations

import os
from typing import Any

# ── attempt tracker for fail_then_pass ───────────────────────────────────────
# Maps run_id → number of times run_parity_test has been called for it
_attempt_counter: dict[str, int] = {}

# ── shared corpus/tolerance/artifact constants ─────────────────────────────

_CORPUS = {
    "n_samples": 500,
    "corpus_id": "kws-corpus-v1",
    "sha256": "a3f1c8b2e9d04f7a6c5b3e2d1f8a9c0e4b7d6f3a2c1e0d9b8a7f6c5e4d3b2a10",
}

_TOLERANCES = {
    "resample":     {"max_abs_diff": 1e-5,  "mean_abs_diff": 1e-6},
    "pre_emphasis": {"max_abs_diff": 1e-5,  "mean_abs_diff": 1e-6},
    "framing":      {"max_abs_diff": 1e-5,  "mean_abs_diff": 1e-6},
    "fft":          {"max_abs_diff": 1e-4,  "mean_abs_diff": 1e-5},
    "mel":          {"max_abs_diff": 1e-4,  "mean_abs_diff": 1e-5},
    "log":          {"max_abs_diff": 1e-4,  "mean_abs_diff": 1e-5},
    "normalize":    {"max_abs_diff": 5e-4,  "mean_abs_diff": 1e-4},
    "min_prediction_agreement": 0.95,
    "max_accuracy_delta": 0.03,
}


def _stage(name: str, order: int,
           shape_ref: list, shape_impl: list,
           max_ad: float, mean_ad: float, cos: float,
           status: str,
           worst_sample: str | None = "sample_0042",
           worst_index: int | None = 1234) -> dict[str, Any]:
    stages_tols = {
        "resample":     {"max_abs_diff": 1e-5,  "mean_abs_diff": 1e-6},
        "pre_emphasis": {"max_abs_diff": 1e-5,  "mean_abs_diff": 1e-6},
        "framing":      {"max_abs_diff": 1e-5,  "mean_abs_diff": 1e-6},
        "fft":          {"max_abs_diff": 1e-4,  "mean_abs_diff": 1e-5},
        "mel":          {"max_abs_diff": 1e-4,  "mean_abs_diff": 1e-5},
        "log":          {"max_abs_diff": 1e-4,  "mean_abs_diff": 1e-5},
        "normalize":    {"max_abs_diff": 5e-4,  "mean_abs_diff": 1e-4},
    }
    return {
        "name": name,
        "order": order,
        "shape_ref":  shape_ref,
        "shape_impl": shape_impl,
        "dtype_ref":  "float32",
        "dtype_impl": "float32",
        "max_abs_diff":      round(max_ad, 8),
        "mean_abs_diff":     round(mean_ad, 9),
        "cosine_similarity": round(cos, 7),
        "tolerance":         stages_tols[name],
        "status":            status,
        "worst_sample_id":   worst_sample,
        "worst_index":       worst_index,
    }


def _passing_stages() -> list[dict]:
    """All 7 stages passing with near-zero divergence."""
    return [
        _stage("resample",     0, [500,16000],   [500,16000],   0.0,      0.0,      1.0,       "PASS", None, None),
        _stage("pre_emphasis", 1, [500,16000],   [500,16000],   2.3e-7,   4.1e-8,   0.9999998, "PASS"),
        _stage("framing",      2, [500,49,400],  [500,49,400],  8.1e-6,   1.2e-7,   0.9999991, "PASS"),
        _stage("fft",          3, [500,49,201],  [500,49,201],  6.4e-5,   9.3e-6,   0.9999921, "PASS"),
        _stage("mel",          4, [500,49,40],   [500,49,40],   7.2e-5,   8.1e-6,   0.9999912, "PASS"),
        _stage("log",          5, [500,49,40],   [500,49,40],   9.1e-5,   1.1e-5,   0.9999901, "PASS"),
        _stage("normalize",    6, [500,49,40],   [500,49,40],   4.3e-4,   8.7e-5,   0.9999880, "PASS"),
    ]


def _framing_shape_fail_stages() -> list[dict]:
    """Framing bug: impl produces 50 frames instead of 49.  Cascades forward."""
    return [
        _stage("resample",     0, [500,16000],   [500,16000],   0.0,      0.0,      1.0,       "PASS", None, None),
        _stage("pre_emphasis", 1, [500,16000],   [500,16000],   2.3e-7,   4.1e-8,   0.9999998, "PASS"),
        _stage("framing",      2, [500,49,400],  [500,50,400],  0.0312,   0.00841,  0.9921,    "FAIL", "sample_0117", 19592),
        _stage("fft",          3, [500,49,201],  [500,50,201],  0.0891,   0.0234,   0.9814,    "FAIL", "sample_0117", 9855),
        _stage("mel",          4, [500,49,40],   [500,50,40],   0.1243,   0.0412,   0.9731,    "FAIL", "sample_0312", 1959),
        _stage("log",          5, [500,49,40],   [500,50,40],   0.1891,   0.0623,   0.9652,    "FAIL", "sample_0312", 1959),
        _stage("normalize",    6, [500,49,40],   [500,50,40],   0.2014,   0.0701,   0.9621,    "FAIL", "sample_0312", 1959),
    ]


def _mel_scale_fail_stages() -> list[dict]:
    """Mel filterbank normalization bug: HTK vs Slaney produces constant scale error."""
    return [
        _stage("resample",     0, [500,16000],   [500,16000],   0.0,      0.0,      1.0,       "PASS", None, None),
        _stage("pre_emphasis", 1, [500,16000],   [500,16000],   2.3e-7,   4.1e-8,   0.9999998, "PASS"),
        _stage("framing",      2, [500,49,400],  [500,49,400],  8.1e-6,   1.2e-7,   0.9999991, "PASS"),
        _stage("fft",          3, [500,49,201],  [500,49,201],  6.4e-5,   9.3e-6,   0.9999921, "PASS"),
        _stage("mel",          4, [500,49,40],   [500,49,40],   1.843,    0.4712,   0.9999901, "FAIL", "sample_0200", 872),
        _stage("log",          5, [500,49,40],   [500,49,40],   0.6021,   0.1540,   0.9988120, "FAIL", "sample_0200", 872),
        _stage("normalize",    6, [500,49,40],   [500,49,40],   0.7214,   0.1843,   0.9981220, "FAIL", "sample_0200", 872),
    ]


def _build_payload(gate: str, attempt: int, run_id: str,
                   timestamp: str, status: str,
                   stages: list[dict],
                   first_divergent: str | None,
                   pred_agreement: float,
                   acc_ref: float, acc_impl: float,
                   diagnosis_hints: list[dict],
                   artifacts_suffix: str = "") -> dict[str, Any]:
    suffix = artifacts_suffix or f"attempt{attempt}"
    return {
        "schema_version": "1.0.0",
        "tool": "run_parity_test",
        "source": "mock",
        "run_id": run_id,
        "timestamp": timestamp,
        "status": status,
        "gate": gate,
        "attempt": attempt,
        "corpus": _CORPUS,
        "tolerances": _TOLERANCES,
        "stages": stages,
        "first_divergent_stage": first_divergent,
        "end_to_end": {
            "prediction_agreement": pred_agreement,
            "accuracy_ref":   acc_ref,
            "accuracy_impl":  acc_impl,
            "accuracy_delta": round(acc_impl - acc_ref, 4),
        },
        "diagnosis_hints": diagnosis_hints,
        "artifacts": {
            "report_path":    f"workflow/runs/{run_id}/parity_{gate}_{suffix}.json",
            "per_sample_csv": f"workflow/runs/{run_id}/parity_{gate}_{suffix}_per_sample.csv",
        },
    }


# ── Diagnosis hint banks ──────────────────────────────────────────────────────

_HINTS_FRAMING_SHAPE = [
    {
        "stage": "framing",
        "hypothesis": (
            "Wrong number of frames: impl produces 50 frames, ref produces 49. "
            "Likely cause: off-by-one in frame-count formula (floor vs ceil), "
            "or hop_length mismatch."
        ),
        "evidence": "shape_ref=[500,49,400] vs shape_impl=[500,50,400]; frame dimension differs by 1",
    },
    {
        "stage": "framing",
        "hypothesis": (
            "Incorrect padding: impl may be zero-padding the input before framing "
            "where ref does not (or vice versa)."
        ),
        "evidence": "Extra frame appears at the end of the sequence for all 500 samples",
    },
]

_HINTS_MEL_SCALE = [
    {
        "stage": "mel",
        "hypothesis": (
            "Filterbank normalization mismatch: ref uses Slaney normalization "
            "(area_norm=True) but impl uses HTK (no normalization), or vice versa. "
            "This produces a constant per-band scale factor ~2/(fmax-fmin)."
        ),
        "evidence": (
            "max_abs_diff=1.843 and mean_abs_diff=0.471 at mel stage with "
            "identical shapes — constant multiplicative offset, not an additive one"
        ),
    },
]

_HINTS_PASS: list[dict] = []


def _get_scenario() -> str:
    return os.environ.get("CODE2EDGE_MOCK_PARITY_SCENARIO", "pass").strip().lower()


VALID_SCENARIOS = {"pass", "fail_then_pass", "always_fail", "fail_mel_scale", "fail_framing_shape"}


def build_parity_payload(gate: str, attempt: int, run_id: str,
                          timestamp: str) -> dict[str, Any]:
    """Build the appropriate run_parity_test output dict for the active scenario."""
    scenario = _get_scenario()

    if scenario not in VALID_SCENARIOS:
        raise ValueError(
            f"Unknown CODE2EDGE_MOCK_PARITY_SCENARIO='{scenario}'. "
            f"Valid: {sorted(VALID_SCENARIOS)}"
        )

    # ── pass ─────────────────────────────────────────────────────────────────
    if scenario == "pass":
        return _build_payload(
            gate=gate, attempt=attempt, run_id=run_id, timestamp=timestamp,
            status="PASS",
            stages=_passing_stages(),
            first_divergent=None,
            pred_agreement=0.978, acc_ref=0.923, acc_impl=0.921,
            diagnosis_hints=_HINTS_PASS,
        )

    # ── always_fail ───────────────────────────────────────────────────────────
    if scenario == "always_fail":
        return _build_payload(
            gate=gate, attempt=attempt, run_id=run_id, timestamp=timestamp,
            status="FAIL",
            stages=_framing_shape_fail_stages(),
            first_divergent="framing",
            pred_agreement=0.834, acc_ref=0.923, acc_impl=0.761,
            diagnosis_hints=_HINTS_FRAMING_SHAPE,
        )

    # ── fail_framing_shape ────────────────────────────────────────────────────
    if scenario == "fail_framing_shape":
        return _build_payload(
            gate=gate, attempt=attempt, run_id=run_id, timestamp=timestamp,
            status="FAIL",
            stages=_framing_shape_fail_stages(),
            first_divergent="framing",
            pred_agreement=0.834, acc_ref=0.923, acc_impl=0.761,
            diagnosis_hints=_HINTS_FRAMING_SHAPE,
        )

    # ── fail_mel_scale ────────────────────────────────────────────────────────
    if scenario == "fail_mel_scale":
        return _build_payload(
            gate=gate, attempt=attempt, run_id=run_id, timestamp=timestamp,
            status="FAIL",
            stages=_mel_scale_fail_stages(),
            first_divergent="mel",
            pred_agreement=0.861, acc_ref=0.923, acc_impl=0.798,
            diagnosis_hints=_HINTS_MEL_SCALE,
        )

    # ── fail_then_pass ────────────────────────────────────────────────────────
    # attempt 1 (first call for this run_id) → FAIL, attempt 2+ → PASS
    call_count = _attempt_counter.get(run_id, 0) + 1
    _attempt_counter[run_id] = call_count

    if call_count == 1:
        return _build_payload(
            gate=gate, attempt=attempt, run_id=run_id, timestamp=timestamp,
            status="FAIL",
            stages=_framing_shape_fail_stages(),
            first_divergent="framing",
            pred_agreement=0.834, acc_ref=0.923, acc_impl=0.761,
            diagnosis_hints=_HINTS_FRAMING_SHAPE,
        )
    else:
        return _build_payload(
            gate=gate, attempt=attempt, run_id=run_id, timestamp=timestamp,
            status="PASS",
            stages=_passing_stages(),
            first_divergent=None,
            pred_agreement=0.978, acc_ref=0.923, acc_impl=0.921,
            diagnosis_hints=_HINTS_PASS,
        )
