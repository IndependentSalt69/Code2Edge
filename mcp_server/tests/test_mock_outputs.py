"""
mcp_server/tests/test_mock_outputs.py

Verify that every tool in mock mode returns schema-valid output.
"""
import pytest

from mcp_server._schema import validate_output
import jsonschema


# ── helpers ───────────────────────────────────────────────────────────────────

def _assert_valid(tool_name: str, payload: dict) -> None:
    """Run schema validation and give a readable assertion message on failure."""
    try:
        validate_output(tool_name, payload)
    except jsonschema.ValidationError as e:
        pytest.fail(f"[{tool_name}] schema validation failed:\n{e.message}")


def _assert_envelope(payload: dict, expected_tool: str) -> None:
    assert payload["tool"] == expected_tool
    assert payload["source"] == "mock"
    assert "run_id" in payload
    assert "timestamp" in payload
    assert "schema_version" in payload


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_profile_model_mock_valid():
    from mcp_server.tools import profile_model
    result = await profile_model.run(
        repo_path="/fake/repo", model_file="model.tflite"
    )
    _assert_envelope(result, "profile_model")
    _assert_valid("profile_model", result)
    assert result["model"]["architecture"] == "DS-CNN"
    assert result["model"]["quantized"] is True
    assert isinstance(result["model"]["layers"], list)
    assert len(result["model"]["layers"]) > 0


@pytest.mark.asyncio
async def test_inspect_pipeline_mock_valid():
    from mcp_server.tools import inspect_pipeline
    result = await inspect_pipeline.run(
        repo_path="/fake/repo", manifest_path="/fake/manifest.json"
    )
    _assert_envelope(result, "inspect_pipeline")
    _assert_valid("inspect_pipeline", result)
    assert result["pipeline"]["domain"] == "kws"
    assert result["pipeline"]["n_stages"] == 7
    stage_names = [s["name"] for s in result["pipeline"]["stages"]]
    assert stage_names == ["resample", "pre_emphasis", "framing", "fft", "mel", "log", "normalize"]


@pytest.mark.asyncio
async def test_check_target_mock_valid():
    from mcp_server.tools import check_target
    result = await check_target.run(model_file="model.tflite", arena_kb=128.0)
    _assert_envelope(result, "check_target")
    _assert_valid("check_target", result)
    assert result["target"]["sram_budget_kb"] == 786
    assert result["target"]["flash_budget_kb"] == 2048
    assert "fits" in result["target"]
    assert "headroom_kb" in result["target"]


@pytest.mark.asyncio
async def test_check_target_reflects_arena_kb():
    from mcp_server.tools import check_target
    result = await check_target.run(model_file="model.tflite", arena_kb=200.0)
    assert result["target"]["arena_kb"] == 200.0
    _assert_valid("check_target", result)


@pytest.mark.asyncio
async def test_check_target_over_budget_fits_false():
    from mcp_server.tools import check_target
    # 780 KB arena + 7.7 feature buffer = 787.7 > 786 budget
    result = await check_target.run(model_file="model.tflite", arena_kb=780.0)
    assert result["target"]["fits"] is False
    assert result["target"]["headroom_kb"] < 0
    _assert_valid("check_target", result)


@pytest.mark.asyncio
async def test_run_parity_test_mock_pass_valid():
    from mcp_server.tools import run_parity_test
    result = await run_parity_test.run(
        gate="host", attempt=1,
        corpus_dir="/fake/corpus",
        ref_pipeline_path="/fake/ref.py",
        impl_pipeline_path="/fake/impl",
        run_id="test-run-001",
    )
    _assert_envelope(result, "run_parity_test")
    _assert_valid("run_parity_test", result)
    assert result["status"] == "PASS"
    assert result["gate"] == "host"
    assert result["first_divergent_stage"] is None
    assert len(result["stages"]) == 7
    assert all(s["status"] == "PASS" for s in result["stages"])


@pytest.mark.asyncio
async def test_benchmark_target_mock_valid():
    from mcp_server.tools import benchmark_target
    result = await benchmark_target.run(
        model_file="model.tflite", n_inferences=50
    )
    _assert_envelope(result, "benchmark_target")
    _assert_valid("benchmark_target", result)
    assert result["benchmark"]["n_inferences"] == 50
    assert result["benchmark"]["latency_ms"]["n"] == 50
    assert isinstance(result["benchmark"]["predicted_vs_measured"], list)
    assert len(result["benchmark"]["predicted_vs_measured"]) > 0


@pytest.mark.asyncio
async def test_quantize_model_mock_valid():
    from mcp_server.tools import quantize_model
    result = await quantize_model.run(
        model_file="model.h5",
        representative_data_dir="/fake/data",
        n_calibration_samples=50,
        output_dir="workflow/runs/test",
    )
    _assert_envelope(result, "quantize_model")
    _assert_valid("quantize_model", result)
    assert result["quantization"]["output_dtype"] == "int8"
    assert result["quantization"]["n_calibration_samples"] == 50
    assert "workflow/runs/test" in result["quantization"]["output_path"]
