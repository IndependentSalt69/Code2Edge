"""
mcp_server/server.py

Code2Edge MCP server — FastMCP over stdio.

Run:
    python -m mcp_server.server          # stdio transport (default)
    python mcp_server/server.py

Environment variables (per-tool mode):
    CODE2EDGE_PROFILE_MODEL_MODE     = mock | real   (default: mock)
    CODE2EDGE_INSPECT_PIPELINE_MODE  = mock | real
    CODE2EDGE_CHECK_TARGET_MODE      = mock | real
    CODE2EDGE_RUN_PARITY_TEST_MODE   = mock | real
    CODE2EDGE_BENCHMARK_TARGET_MODE  = mock | real
    CODE2EDGE_QUANTIZE_MODEL_MODE    = mock | real

Parity scenario (only used when run_parity_test is in mock mode):
    CODE2EDGE_MOCK_PARITY_SCENARIO   = pass | fail_then_pass | always_fail |
                                       fail_mel_scale | fail_framing_shape
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server.tools import (
    benchmark_target,
    check_target,
    gate_step,
    get_run_status,
    inspect_pipeline,
    profile_model,
    quantize_model,
    record_approval,
    run_parity_test,
    start_run,
)

mcp = FastMCP(
    name="code2edge",
    instructions=(
        "Code2Edge hardware-aware deployment tools. "
        "Use profile_model → inspect_pipeline → check_target in parallel, "
        "then run_parity_test (with gate_step retry logic), "
        "then benchmark_target."
    ),
)

# ── Register tools ────────────────────────────────────────────────────────────

@mcp.tool()
async def profile_model_tool(repo_path: str, model_file: str,
                              labels_file: str = "", config_file: str = "") -> dict:
    """Analyse a model repository: architecture, layer summary, quantization."""
    return await profile_model.run(repo_path=repo_path, model_file=model_file,
                                   labels_file=labels_file, config_file=config_file)


@mcp.tool()
async def inspect_pipeline_tool(repo_path: str, manifest_path: str,
                                 corpus_dir: str = "") -> dict:
    """Trace the Python preprocessing pipeline and return stage constants."""
    return await inspect_pipeline.run(repo_path=repo_path,
                                      manifest_path=manifest_path,
                                      corpus_dir=corpus_dir)


@mcp.tool()
async def check_target_tool(model_file: str, arena_kb: float,
                             target_id: str = "STM32U585") -> dict:
    """Check whether a model fits in the STM32U585 memory budget."""
    return await check_target.run(model_file=model_file, arena_kb=arena_kb,
                                  target_id=target_id)


@mcp.tool()
async def run_parity_test_tool(gate: str, attempt: int,
                                corpus_dir: str, ref_pipeline_path: str,
                                impl_pipeline_path: str,
                                run_id: str = "") -> dict:
    """Run differential parity test (Python ref vs generated C++) stage-by-stage."""
    return await run_parity_test.run(gate=gate, attempt=attempt,
                                     corpus_dir=corpus_dir,
                                     ref_pipeline_path=ref_pipeline_path,
                                     impl_pipeline_path=impl_pipeline_path,
                                     run_id=run_id)


@mcp.tool()
async def benchmark_target_tool(model_file: str, n_inferences: int,
                                 target_id: str = "STM32U585",
                                 corpus_dir: str = "") -> dict:
    """Flash and benchmark the model on STM32U585: latency + memory."""
    return await benchmark_target.run(model_file=model_file,
                                      n_inferences=n_inferences,
                                      target_id=target_id,
                                      corpus_dir=corpus_dir)


@mcp.tool()
async def quantize_model_tool(model_file: str, representative_data_dir: str,
                               n_calibration_samples: int = 100,
                               output_dir: str = "workflow/runs/quantized") -> dict:
    """Post-training quantize float32 model to int8 (optional tool)."""
    return await quantize_model.run(model_file=model_file,
                                    representative_data_dir=representative_data_dir,
                                    n_calibration_samples=n_calibration_samples,
                                    output_dir=output_dir)


@mcp.tool()
async def start_run_tool(requested_by: str = "") -> dict:
    """Create a new Code2Edge deployment run and return its run_id."""
    return await start_run.run(requested_by=requested_by)


@mcp.tool()
async def get_run_status_tool(run_id: str) -> dict:
    """Read back a run's current stage, status and history."""
    return await get_run_status.run(run_id=run_id)


@mcp.tool()
async def gate_step_tool(run_id: str, gate: str, parity_result: dict) -> dict:
    """Evaluate a run_parity_test result: continue, repair (attempts < 3), or escalate."""
    return await gate_step.run(run_id=run_id, gate=gate, parity_result=parity_result)


@mcp.tool()
async def record_approval_tool(run_id: str, checkpoint: str, approved: bool,
                                approver: str = "", notes: str = "") -> dict:
    """Record a human approval/rejection at a workflow checkpoint."""
    return await record_approval.run(run_id=run_id, checkpoint=checkpoint,
                                     approved=approved, approver=approver, notes=notes)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
