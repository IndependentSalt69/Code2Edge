"""mcp_server/tools/check_target.py"""
from __future__ import annotations

from mcp_server._mock_loader import load_mock
from mcp_server._mode import get_mode
from mcp_server._schema import validate_output
from mcp_server._ids import new_run_id, utcnow_iso

_TOOL = "check_target"


async def run(model_file: str, arena_kb: float,
              target_id: str = "STM32U585") -> dict:
    mode = get_mode(_TOOL)
    run_id = new_run_id(_TOOL)

    if mode == "mock":
        payload = load_mock(_TOOL, run_id=run_id)
        # Reflect requested arena_kb and target_id into the mock
        payload["target"]["arena_kb"] = arena_kb
        payload["target"]["target_id"] = target_id
        sram_budget = payload["target"]["sram_budget_kb"]
        total_sram = arena_kb + payload["target"]["feature_buffer_kb"]
        payload["target"]["total_sram_kb"] = round(total_sram, 1)
        payload["target"]["fits"] = total_sram <= sram_budget
        payload["target"]["headroom_kb"] = round(sram_budget - total_sram, 1)
    else:
        from mcp_server.adapters import target_adapter
        try:
            payload = target_adapter.run_check_target(
                model_file=model_file, arena_kb=arena_kb, target_id=target_id,
            )
        except NotImplementedError as e:
            return {
                "schema_version": "1.0.0", "tool": _TOOL, "source": "real",
                "run_id": run_id, "timestamp": utcnow_iso(),
                "status": "ERROR", "error_message": str(e),
            }

    validate_output(_TOOL, payload)
    return payload
