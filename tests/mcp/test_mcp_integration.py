"""
End-to-end integration tests for Code2Edge MCP server.

Tests:
  - FastMCP stdio server initialization and tool discovery.
  - Real-mode execution of check_target_tool conforming to contracts/check_target.schema.json.
  - Verification that no mock values are returned in real mode.
  - Real-mode execution of benchmark_target_tool reporting pending model artifact.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jsonschema
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
async def test_mcp_server_end_to_end_benchmark_target_safe_handling():
    """Verify benchmark_target_tool returns clean error when model artifact is pending."""
    env = os.environ.copy()
    env["CODE2EDGE_BENCHMARK_TARGET_MODE"] = "real"

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env=env,
        cwd=str(REPO_ROOT),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            call_result = await session.call_tool(
                "benchmark_target_tool",
                arguments={
                    "model_file": "checkpoints/best.pt",
                    "n_inferences": 50,
                    "target_id": "STM32U585",
                },
            )

            assert len(call_result.content) >= 1
            payload = json.loads(call_result.content[0].text)

            assert payload["tool"] == "benchmark_target"
            assert payload["source"] == "real"
            assert payload["status"] == "ERROR"
            assert "pending from Person A" in payload["error_message"]
            assert "STM32U585 is verified" in payload["error_message"]
