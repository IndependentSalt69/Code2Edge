"""
mcp_server/tests/test_real_pipeline_adapter.py

Real-mode profile_model / inspect_pipeline, wired to Person A's actual
reference outputs (Prompt 10A). Skipped automatically if those files are
absent (e.g. running against an older checkout of the repo).
"""
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_MANIFEST = _REPO_ROOT / "reference" / "pipeline_manifest.json"

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
