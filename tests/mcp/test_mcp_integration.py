"""
End-to-end integration tests for Code2Edge MCP server.

Execution Separation Contract:
  - pytest = Software / MCP / Adapter / Contract integration validation (Hardware boundaries mocked).
  - Explicit CLI / MCP run = Authoritative physical hardware execution on Arduino UNO Q (STM32U585 over COM3).
  - Normal pytest execution must NEVER silently connect to COM3 or trigger physical hardware.

Tests:
  - FastMCP stdio server initialization and tool discovery.
  - Real-mode execution of check_target_tool conforming to contracts/check_target.schema.json.
  - Real-mode execution of benchmark_target_tool with mocked physical benchmark boundary.
  - Focused target_adapter argument verification and stdout->stderr redirection validation.
"""

from __future__ import annotations

import json
import os
import sys
import unittest.mock as mock
from pathlib import Path

import jsonschema
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_server.adapters.target_adapter import run_benchmark_target
from mcp_server.server import mcp

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"


@pytest.mark.asyncio
async def test_mcp_server_end_to_end_check_target_real_mode(tmp_path):
    """Start MCP server over stdio, invoke check_target_tool, and validate output schema."""
    env = os.environ.copy()
    env["CODE2EDGE_CHECK_TARGET_MODE"] = "real"
    env["CODE2EDGE_PROFILE_MODEL_MODE"] = "real"
    env["CODE2EDGE_INSPECT_PIPELINE_MODE"] = "real"

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env=env,
        cwd=str(REPO_ROOT),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Discover registered tools
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            assert "check_target_tool" in tool_names
            assert "benchmark_target_tool" in tool_names
            assert "profile_model_tool" in tool_names
            assert "inspect_pipeline_tool" in tool_names

            # 2. Invoke check_target_tool in real mode
            call_result = await session.call_tool(
                "check_target_tool",
                arguments={
                    "model_file": "checkpoints/best.pt",
                    "arena_kb": 128.0,
                    "target_id": "STM32U585",
                },
            )

            assert len(call_result.content) >= 1
            payload = json.loads(call_result.content[0].text)

            # 3. Assert real mode values (no mock data)
            assert payload["tool"] == "check_target"
            assert payload["source"] == "real"
            assert payload["schema_version"] == "1.0.0"

            target = payload["target"]
            assert target["target_id"] == "STM32U585"
            assert target["sram_budget_kb"] == 786.0
            assert target["flash_budget_kb"] == 2048.0
            assert target["arena_kb"] == 128.0
            assert target["feature_buffer_kb"] == 25.3
            assert target["total_sram_kb"] == 153.3
            assert target["headroom_kb"] == 632.7
            assert target["fits"] is True
            assert isinstance(target["warnings"], list)
            assert isinstance(target["unsupported_ops"], list)

            # 4. Validate output schema against contracts/check_target.schema.json
            schema_file = CONTRACTS_DIR / "check_target.schema.json"
            schema_doc = json.loads(schema_file.read_text(encoding="utf-8"))
            output_schema = {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$defs": schema_doc["$defs"],
                "$ref": "#/$defs/Output",
            }
            jsonschema.validate(instance=payload, schema=output_schema)

            # 5. Save test evidence report to tmp_path
            test_evidence_path = tmp_path / "check_target_test_response.json"
            test_evidence_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            assert test_evidence_path.exists()


@pytest.mark.asyncio
async def test_mcp_server_end_to_end_benchmark_target_mocked_hardware(tmp_path):
    """
    Verify benchmark_target_tool invocation through FastMCP server interface
    with the physical hardware runner mocked to avoid hardware access during pytest.
    
    Path:
      MCP Client -> benchmark_target tool -> target_adapter.run_benchmark_target()
      -> mocked run_physical_benchmark() -> deterministic structured result -> JSON valid
    """
    mock_benchmark_report = {
        "$schema": "contracts/target/benchmark-result.schema.json",
        "target_id": "arduino_uno_q_stm32u585",
        "model_id": "tiny-kws-dscnn-int8",
        "run_type": "MEASURED",
        "timestamp": "2026-09-26T08:44:00+00:00",
        "latency_ms": {
            "inference_avg_ms": 5000.0,
            "inference_min_ms": 4990.0,
            "inference_max_ms": 5010.0,
            "inference_std_ms": 5.0,
            "preprocessing_avg_ms": None,
            "end_to_end_avg_ms": None,
            "clock_cycles_inference": 800000000,
        },
        "memory_bytes": {
            "flash_used_bytes": 311336,
            "flash_total_bytes": 2097152,
            "flash_headroom_bytes": 1785816,
            "sram_used_bytes": 242224,
            "sram_total_bytes": 804864,
            "sram_headroom_bytes": 562640,
            "tensor_arena_bytes": 166560,
            "feature_buffer_bytes": 25856,
            "static_ram_bytes": 75664,
            "stack_peak_bytes": None,
            "application_partition": {
                "flash_partition_max_bytes": 786432,
                "flash_partition_usage_pct": 39.59,
                "sram_partition_max_bytes": 262144,
                "sram_partition_usage_pct": 92.40,
            },
        },
        "execution_metadata": {
            "clock_mhz": 160.0,
            "num_iterations": 3,
            "warmup_iterations": 1,
            "runtime": "native_static_dscnn_runner",
            "quantization": "int8",
            "toolchain_version": "arduino-cli (1.5.2-rc.1), core: arduino:zephyr (1.0.0)",
            "measurement_source": "physical_stm32u585",
            "serial_port": "COM3",
            "fixture": "yes.wav",
        },
        "sample_prediction": {
            "test_fixture": "yes.wav",
            "predicted_class_index": 2,
            "predicted_label": "yes",
            "measured_logits": [-30, -21, 83, -39, -28, -27, -20, -28, -30, -16, -29, -23],
            "dequantized": [
                -1.649778, -1.199838, 3.999462, -2.099717, -1.549791, -1.499798,
                -1.149845, -1.549791, -1.649778, -0.949872, -1.599785, -1.299825,
            ],
        },
        "validation": {
            "expected_logits": [-30, -21, 83, -39, -28, -27, -20, -28, -30, -16, -29, -23],
            "expected_predicted_index": 2,
            "expected_predicted_label": "yes",
            "all_logits_matched": True,
            "all_argmax_matched": True,
            "all_labels_matched": True,
            "total_iterations_verified": 3,
            "status": "PASS",
        },
    }

    with mock.patch("mcp_server.adapters.target_adapter.run_physical_benchmark") as mock_runner:
        mock_runner.return_value = mock_benchmark_report

        with mock.patch.dict(os.environ, {"CODE2EDGE_BENCHMARK_TARGET_MODE": "real"}):
            # 1. Discover registered tools
            tools = await mcp.list_tools()
            tool_names = [t.name for t in tools]
            assert "benchmark_target_tool" in tool_names

            # 2. Invoke tool via FastMCP server
            raw_result = await mcp.call_tool(
                "benchmark_target_tool",
                arguments={
                    "model_file": "src/pipeline/model_data.c",
                    "n_inferences": 3,
                    "target_id": "STM32U585",
                },
            )

            assert len(raw_result) >= 1
            response_text = raw_result[0].text if hasattr(raw_result[0], "text") else json.dumps(raw_result[0])
            payload = json.loads(response_text)

            # 3. Assert MCP response structure and real values
            assert payload["tool"] == "benchmark_target"
            assert payload["source"] == "real"
            assert payload["schema_version"] == "1.0.0"

            bench = payload["benchmark"]
            assert bench["target_id"] == "STM32U585"
            assert bench["n_inferences"] == 3
            assert bench["latency_ms"]["mean"] == 5000.0
            assert bench["latency_ms"]["min"] == 4990.0
            assert bench["latency_ms"]["max"] == 5010.0
            assert bench["latency_ms"]["n"] == 3
            assert bench["flash_used_kb"] == pytest.approx(304.0, rel=1e-1)
            assert bench["sram_peak_kb"] == pytest.approx(236.5, rel=1e-1)
            assert len(bench["predictions"]) == 1
            assert bench["predictions"][0]["label"] == "yes"
            assert bench["predictions"][0]["class_id"] == 2
            assert len(bench["predicted_vs_measured"]) == 3

            # 4. Validate output schema against contracts/benchmark_target.schema.json
            schema_file = CONTRACTS_DIR / "benchmark_target.schema.json"
            schema_doc = json.loads(schema_file.read_text(encoding="utf-8"))
            output_schema = {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$defs": schema_doc["$defs"],
                "$ref": "#/$defs/Output",
            }
            jsonschema.validate(instance=payload, schema=output_schema)

            # 5. Verify run_physical_benchmark was invoked with expected parameters
            mock_runner.assert_called_once_with(
                port="COM3",
                baud_rate=115200,
                num_iterations=3,
                warmup_iterations=5,
                timeout_per_inference=40.0,
                sketch_path=REPO_ROOT / "tests" / "firmware" / "benchmark_harness",
                fqbn="arduino:zephyr:unoq",
                tensor_arena_bytes=166560,
                feature_buffer_bytes=25856,
                output_file=REPO_ROOT / "evidence" / "benchmarks" / "stm32u585_benchmark_report.json",
                skip_compile=False,
            )


def test_run_benchmark_target_adapter_invocation_and_redirection(capsys):
    """
    Focused adapter test verifying run_benchmark_target() invokes
    run_physical_benchmark() with exact arguments and redirects diagnostic
    stdout to stderr to prevent MCP JSON-RPC stdio pollution.
    """
    mock_benchmark_report = {
        "$schema": "contracts/target/benchmark-result.schema.json",
        "target_id": "arduino_uno_q_stm32u585",
        "model_id": "tiny-kws-dscnn-int8",
        "run_type": "MEASURED",
        "timestamp": "2026-09-26T08:44:00+00:00",
        "latency_ms": {
            "inference_avg_ms": 5024.45,
            "inference_min_ms": 5024.45,
            "inference_max_ms": 5024.45,
            "inference_std_ms": 0.0,
            "preprocessing_avg_ms": None,
            "end_to_end_avg_ms": None,
            "clock_cycles_inference": 803912575,
        },
        "memory_bytes": {
            "flash_used_bytes": 311336,
            "flash_total_bytes": 2097152,
            "flash_headroom_bytes": 1785816,
            "sram_used_bytes": 242224,
            "sram_total_bytes": 804864,
            "sram_headroom_bytes": 562640,
            "tensor_arena_bytes": 166560,
            "feature_buffer_bytes": 25856,
            "static_ram_bytes": 75664,
            "stack_peak_bytes": None,
        },
        "execution_metadata": {
            "clock_mhz": 160.0,
            "num_iterations": 10,
            "warmup_iterations": 5,
            "runtime": "native_static_dscnn_runner",
            "quantization": "int8",
            "toolchain_version": "arduino-cli (1.5.2-rc.1), core: arduino:zephyr (1.0.0)",
            "measurement_source": "physical_stm32u585",
            "serial_port": "COM3",
            "fixture": "yes.wav",
        },
        "sample_prediction": {
            "test_fixture": "yes.wav",
            "predicted_class_index": 2,
            "predicted_label": "yes",
            "measured_logits": [-30, -21, 83, -39, -28, -27, -20, -28, -30, -16, -29, -23],
            "dequantized": [],
        },
        "validation": {
            "status": "PASS",
        },
    }

    def fake_physical_runner(**kwargs):
        # Simulate stdout diagnostic prints from real benchmark runner
        print("[Code2Edge] Compiling firmware...")
        print("[Code2Edge] Connecting to target MCU on COM3 @ 115200 baud...")
        print("[Code2Edge] Running 5 warmup iterations...")
        print("  [Warmup 1/5] DWT Cycles: 803,912,575 | Latency: 5024.45 ms")
        print("[Code2Edge] Running 10 measured iterations...")
        print("  [Iter  1/10] Cycles: 803,912,575 | 5024.45 ms | Predicted: 'yes' (2) [MATCH]")
        return mock_benchmark_report

    with mock.patch("mcp_server.adapters.target_adapter.run_physical_benchmark", side_effect=fake_physical_runner) as mock_runner:
        result = run_benchmark_target(
            model_file="src/pipeline/model_data.c",
            n_inferences=10,
            target_id="STM32U585",
        )

        # 1. Verify exact argument contract
        mock_runner.assert_called_once_with(
            port="COM3",
            baud_rate=115200,
            num_iterations=10,
            warmup_iterations=5,
            timeout_per_inference=40.0,
            sketch_path=REPO_ROOT / "tests" / "firmware" / "benchmark_harness",
            fqbn="arduino:zephyr:unoq",
            tensor_arena_bytes=166560,
            feature_buffer_bytes=25856,
            output_file=REPO_ROOT / "evidence" / "benchmarks" / "stm32u585_benchmark_report.json",
            skip_compile=False,
        )

        # 2. Verify stdout remains completely clean while stderr receives progress
        captured = capsys.readouterr()
        assert captured.out == "", "stdout must be empty to avoid corrupting MCP JSON-RPC"
        assert "[Code2Edge] Compiling firmware..." in captured.err
        assert "[Code2Edge] Connecting to target MCU" in captured.err
        assert "[Warmup 1/5]" in captured.err
        assert "[Iter  1/10]" in captured.err

        # 3. Verify returned payload
        assert result["tool"] == "benchmark_target"
        assert result["source"] == "real"
        assert result["benchmark"]["n_inferences"] == 10
        assert result["benchmark"]["latency_ms"]["mean"] == 5024.45
