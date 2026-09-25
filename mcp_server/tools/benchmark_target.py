"""mcp_server/tools/benchmark_target.py"""
from __future__ import annotations

from mcp_server._mock_loader import load_mock
from mcp_server._mode import get_mode
from mcp_server._schema import validate_output
from mcp_server._ids import new_run_id, utcnow_iso

_TOOL = "benchmark_target"


async def run(model_file: str, n_inferences: int,
              target_id: str = "STM32U585", corpus_dir: str = "") -> dict:
    mode = get_mode(_TOOL)
    run_id = new_run_id(_TOOL)

    if mode == "mock":
        payload = load_mock(_TOOL, run_id=run_id)
        payload["benchmark"]["n_inferences"] = n_inferences
        payload["benchmark"]["target_id"] = target_id
        payload["benchmark"]["latency_ms"]["n"] = n_inferences
    else:
        from mcp_server.adapters import target_adapter
        try:
            payload = target_adapter.run_benchmark_target(
                model_file=model_file, n_inferences=n_inferences,
                target_id=target_id, corpus_dir=corpus_dir,
            )
        except NotImplementedError as e:
            return {
                "schema_version": "1.0.0", "tool": _TOOL, "source": "real",
                "run_id": run_id, "timestamp": utcnow_iso(),
                "status": "ERROR", "error_message": str(e),
            }

    validate_output(_TOOL, payload)
    return payload
