"""
mcp_server/tests/test_real_pipeline_adapter.py

Real-mode profile_model / inspect_pipeline / run_parity_test(gate="host"),
wired to Person A's actual reference outputs (Prompts 10A and 11 prep).
Skipped automatically if those files are absent (e.g. running against an
older checkout of the repo).
"""
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MANIFEST = _REPO_ROOT / "reference" / "pipeline_manifest.json"
_HOST_PARITY_REPORT = _REPO_ROOT / "evidence" / "parity" / "host_parity_report.json"

pytestmark = pytest.mark.skipif(
    not _MANIFEST.exists(),
    reason="reference/pipeline_manifest.json not present (Person A's real outputs not checked out)",
)


@pytest.mark.asyncio
async def test_real_profile_model_matches_reported_param_count(monkeypatch):
    monkeypatch.setenv("CODE2EDGE_PROFILE_MODEL_MODE", "real")
    from mcp_server.tools import profile_model

    out = await profile_model.run(repo_path="reference/tiny-kws", model_file="src/model.py")
    assert out["source"] == "real"
    assert out["model"]["architecture"] == "DS-CNN"
    assert out["model"]["total_params"] == 119372  # cross-checked against assets/metrics.json
    assert out["model"]["total_macs"] > 0
    assert len(out["model"]["layers"]) > 0
    assert any("analytically" in w for w in out["model"]["warnings"])


@pytest.mark.asyncio
async def test_real_inspect_pipeline_maps_real_stages(monkeypatch):
    monkeypatch.setenv("CODE2EDGE_INSPECT_PIPELINE_MODE", "real")
    from mcp_server.tools import inspect_pipeline

    out = await inspect_pipeline.run(repo_path="reference/tiny-kws",
                                     manifest_path="reference/pipeline_manifest.json")
    assert out["source"] == "real"
    pipeline = out["pipeline"]
    assert pipeline["domain"] == "kws"
    stage_names = [s["name"] for s in pipeline["stages"]]
    assert stage_names == ["fft", "mel", "log", "normalize"]
    assert "resample" not in stage_names
    assert "pre_emphasis" not in stage_names
    assert pipeline["global_constants"]["sample_rate"] == 16000
    assert pipeline["global_constants"]["n_mels"] == 64
    assert pipeline["corpus"]["n_samples"] == 500
    assert len(pipeline["warnings"]) >= 1


@pytest.mark.skipif(not _HOST_PARITY_REPORT.exists(),
                    reason="evidence/parity/host_parity_report.json not present")
@pytest.mark.asyncio
async def test_real_run_parity_test_host_gate_matches_authoritative_report(monkeypatch):
    monkeypatch.setenv("CODE2EDGE_RUN_PARITY_TEST_MODE", "real")
    from mcp_server.tools import run_parity_test

    out = await run_parity_test.run(
        gate="host", attempt=1, corpus_dir="reference/artifacts",
        ref_pipeline_path="reference/tiny-kws/src/common.py",
        impl_pipeline_path="src/pipeline/feature_extraction.c",
        run_id="test-real-host",
    )
    assert out["source"] == "real"
    assert out["status"] == "PASS"
    assert out["first_divergent_stage"] is None
    assert [s["name"] for s in out["stages"]] == ["fft", "mel", "log", "normalize"]
    assert all(s["status"] == "PASS" for s in out["stages"])
    assert out["corpus"]["n_samples"] == 500

    # Real reference accuracy is present; model-level comparison is honestly
    # null rather than fabricated, since the harness only measures
    # preprocessing-stage parity.
    assert out["end_to_end"]["accuracy_ref"] == pytest.approx(0.9664621676891616)
    assert out["end_to_end"]["prediction_agreement"] is None
    assert out["end_to_end"]["accuracy_impl"] is None
    assert any("not measured" in w or "null" in w for w in out["warnings"])


@pytest.mark.asyncio
async def test_real_run_parity_test_device_gate_still_unwired(monkeypatch):
    monkeypatch.setenv("CODE2EDGE_RUN_PARITY_TEST_MODE", "real")
    from mcp_server.tools import run_parity_test

    out = await run_parity_test.run(
        gate="device", attempt=1, corpus_dir="reference/artifacts",
        ref_pipeline_path="reference/tiny-kws/src/common.py",
        impl_pipeline_path="src/pipeline/feature_extraction.c",
    )
    assert out["status"] == "ERROR"
    assert "device" in out["error_message"]
