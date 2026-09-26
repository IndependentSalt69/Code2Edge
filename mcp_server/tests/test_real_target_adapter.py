"""
mcp_server/tests/test_real_target_adapter.py

Unit tests for Person B's check_target MCP adapter and real-mode execution.
"""
import pytest
from pathlib import Path

from mcp_server.adapters import target_adapter
from mcp_server.tools import check_target, benchmark_target

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_target_adapter_constants():
    """Verify frozen model and target hardware constants in target_adapter."""
    assert target_adapter.MODEL_ARCH == "DS-CNN"
    assert target_adapter.MODEL_N_CLASSES == 12
    assert target_adapter.MODEL_WIDTH == 160
    assert target_adapter.MODEL_N_BLOCKS == 4
    assert target_adapter.MODEL_INPUT_SHAPE == [1, 1, 64, 101]
    assert target_adapter.MODEL_OUTPUT_SHAPE == [1, 12]
    assert len(target_adapter.MODEL_LABELS) == 12
    assert target_adapter.MODEL_CHECKPOINT_SHA256 == (
        "3ec8eed15c9db102f524022beae1455aca2ad46460f665fcf78c40410445c1df"
    )
    assert target_adapter.TARGET_SRAM_BUDGET_KB == 786.0
    assert target_adapter.TARGET_FLASH_BUDGET_KB == 2048.0


def test_target_adapter_run_check_target_standard_fit():
    """Verify memory fit computation with standard 128 KB tensor arena."""
    out = target_adapter.run_check_target(
        model_file="checkpoints/best.pt",
        arena_kb=128.0,
        target_id="STM32U585",
    )

    assert out["source"] == "real"
    assert out["tool"] == "check_target"
    assert out["schema_version"] == "1.0.0"

    t = out["target"]
    assert t["target_id"] == "STM32U585"
    assert t["sram_budget_kb"] == 786.0
    assert t["flash_budget_kb"] == 2048.0
    assert t["arena_kb"] == 128.0
    assert t["feature_buffer_kb"] == 25.3
    assert t["total_sram_kb"] == 153.3
    assert t["headroom_kb"] == 632.7
    assert t["fits"] is True
    assert isinstance(t["warnings"], list)


def test_target_adapter_run_check_target_sram_overflow():
    """Verify fits=False and warnings when requested arena exceeds 786 KB SRAM."""
    out = target_adapter.run_check_target(
        model_file="checkpoints/best.pt",
        arena_kb=800.0,
        target_id="STM32U585",
    )

    t = out["target"]
    assert t["arena_kb"] == 800.0
    assert t["fits"] is False
    assert t["headroom_kb"] < 0
    assert any("SRAM budget exceeded" in w for w in t["warnings"])


def test_target_adapter_run_check_target_zero_arena_warning():
    """Verify warning when arena_kb is 0."""
    out = target_adapter.run_check_target(
        model_file="checkpoints/best.pt",
        arena_kb=0.0,
        target_id="STM32U585",
    )

    t = out["target"]
    assert any("0 KB" in w for w in t["warnings"])


def test_target_adapter_run_check_target_missing_model_file():
    """Verify handling when model file is not on disk: falls back to the
    real size documented in src/pipeline/model_data.h's header comment,
    not a generic guess."""
    out = target_adapter.run_check_target(
        model_file="models/nonexistent_model.tflite",
        arena_kb=64.0,
        target_id="STM32U585",
    )

    t = out["target"]
    assert t["model_flash_kb"] == 168.2  # 172216 bytes, per model_data.h
    assert any("not found on disk" in w for w in t["warnings"])
    assert any("model_data.h" in w for w in t["warnings"])


@pytest.mark.asyncio
async def test_mcp_check_target_real_mode(monkeypatch):
    """Verify MCP check_target tool executes in real mode and passes schema validation."""
    monkeypatch.setenv("CODE2EDGE_CHECK_TARGET_MODE", "real")

    out = await check_target.run(
        model_file="checkpoints/best.pt",
        arena_kb=128.0,
        target_id="STM32U585",
    )

    assert out["source"] == "real"
    assert out["tool"] == "check_target"
    assert out["target"]["fits"] is True
    assert out["target"]["sram_budget_kb"] == 786.0


@pytest.mark.asyncio
async def test_mcp_check_target_mock_mode(monkeypatch):
    """Verify MCP check_target tool executes in mock mode."""
    monkeypatch.setenv("CODE2EDGE_CHECK_TARGET_MODE", "mock")

    out = await check_target.run(
        model_file="checkpoints/best.pt",
        arena_kb=128.0,
        target_id="STM32U585",
    )

    assert out["source"] == "mock"
    assert out["tool"] == "check_target"
    assert out["target"]["arena_kb"] == 128.0


def _fake_physical_result(**overrides):
    base = {
        "latency_ms": {"inference_avg_ms": 12.5, "inference_min_ms": 11.0, "inference_max_ms": 14.0},
        "memory_bytes": {"flash_used_bytes": 200000, "sram_used_bytes": 190000},
        "sample_prediction": {
            "test_fixture": "yes.wav",
            "predicted_class_index": 2,
            "predicted_label": "yes",
            "measured_logits": [1, 2, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        },
        "validation": {"status": "PASS"},
    }
    base.update(overrides)
    return base


def test_run_benchmark_target_computes_real_predicted_sram_and_confidence(monkeypatch):
    """predicted_vs_measured must not contain the old hardcoded 5000.0/236.5/304.0
    placeholders, and confidence must be a real softmax over measured_logits."""
    monkeypatch.setattr(target_adapter, "run_physical_benchmark", lambda **kw: _fake_physical_result())

    out = target_adapter.run_benchmark_target(
        model_file="src/pipeline/model_data.c", n_inferences=50, target_id="STM32U585",
    )
    b = out["benchmark"]

    assert b["predictions"][0]["confidence"] == pytest.approx(0.811067, abs=1e-5)
    assert b["predictions"][0]["class_id"] == 2
    assert b["predictions"][0]["label"] == "yes"

    rows = {r["metric"]: r for r in b["predicted_vs_measured"]}
    assert rows["latency_ms_mean"]["predicted"] == "not estimated"
    assert rows["latency_ms_mean"]["delta_pct"] is None
    assert rows["flash_used_kb"]["predicted"] == "not estimated"
    assert rows["flash_used_kb"]["delta_pct"] is None
    # sram is a real check_target-derived prediction, not a magic constant
    assert isinstance(rows["sram_peak_kb"]["predicted"], float)
    assert rows["sram_peak_kb"]["predicted"] != 236.5
    assert rows["sram_peak_kb"]["delta_pct"] is not None


def test_run_benchmark_target_missing_memory_bytes_raises(monkeypatch):
    """Must never substitute a canned flash/sram number for a measured quantity."""
    monkeypatch.setattr(
        target_adapter, "run_physical_benchmark",
        lambda **kw: _fake_physical_result(memory_bytes={}),
    )
    with pytest.raises(RuntimeError, match="did not report"):
        target_adapter.run_benchmark_target(
            model_file="src/pipeline/model_data.c", n_inferences=50, target_id="STM32U585",
        )


def test_run_benchmark_target_missing_logits_leaves_predictions_empty(monkeypatch):
    """Must never fabricate a confidence value when no logits were measured."""
    monkeypatch.setattr(
        target_adapter, "run_physical_benchmark",
        lambda **kw: _fake_physical_result(sample_prediction={}),
    )
    out = target_adapter.run_benchmark_target(
        model_file="src/pipeline/model_data.c", n_inferences=50, target_id="STM32U585",
    )
    assert out["benchmark"]["predictions"] == []
    assert any("no measured_logits" in w for w in out["benchmark"]["warnings"])


@pytest.mark.asyncio
async def test_mcp_benchmark_target_real_mode_reports_pending_artifact(monkeypatch):
    """Verify real-mode benchmark_target safely reports pending artifact rather than fabricating."""
    monkeypatch.setenv("CODE2EDGE_BENCHMARK_TARGET_MODE", "real")

    out = await benchmark_target.run(
        model_file="checkpoints/best.pt",
        n_inferences=50,
        target_id="STM32U585",
    )

    assert out["source"] == "real"
    assert out["status"] == "ERROR"
    assert "not found" in out["error_message"]
    assert "checkpoints" in out["error_message"] or "best.pt" in out["error_message"]
