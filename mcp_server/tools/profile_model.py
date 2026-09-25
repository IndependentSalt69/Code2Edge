"""mcp_server/tools/profile_model.py"""
from __future__ import annotations

from mcp_server._mock_loader import load_mock
from mcp_server._mode import get_mode
from mcp_server._schema import validate_output
from mcp_server._ids import new_run_id, utcnow_iso

_TOOL = "profile_model"


async def run(repo_path: str, model_file: str,
              labels_file: str = "", config_file: str = "") -> dict:
    mode = get_mode(_TOOL)
    run_id = new_run_id(_TOOL)

    if mode == "mock":
        payload = load_mock(_TOOL, run_id=run_id)
    else:
        from mcp_server.adapters import pipeline_adapter
        try:
            payload = pipeline_adapter.run_profile_model(
                repo_path=repo_path, model_file=model_file,
                labels_file=labels_file, config_file=config_file,
            )
        except NotImplementedError as e:
            return {
                "schema_version": "1.0.0", "tool": _TOOL, "source": "real",
                "run_id": run_id, "timestamp": utcnow_iso(),
                "status": "ERROR", "error_message": str(e),
            }

    validate_output(_TOOL, payload)
    return payload
