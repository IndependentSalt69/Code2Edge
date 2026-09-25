"""mcp_server/tools/inspect_pipeline.py"""
from __future__ import annotations

from mcp_server._mock_loader import load_mock
from mcp_server._mode import get_mode
from mcp_server._schema import validate_output
from mcp_server._ids import new_run_id, utcnow_iso

_TOOL = "inspect_pipeline"


async def run(repo_path: str, manifest_path: str, corpus_dir: str = "") -> dict:
    mode = get_mode(_TOOL)
    run_id = new_run_id(_TOOL)

    if mode == "mock":
        payload = load_mock(_TOOL, run_id=run_id)
    else:
        from mcp_server.adapters import pipeline_adapter
        try:
            payload = pipeline_adapter.run_inspect_pipeline(
                repo_path=repo_path, manifest_path=manifest_path,
                corpus_dir=corpus_dir,
            )
        except NotImplementedError as e:
            return {
                "schema_version": "1.0.0", "tool": _TOOL, "source": "real",
                "run_id": run_id, "timestamp": utcnow_iso(),
                "status": "ERROR", "error_message": str(e),
            }

    validate_output(_TOOL, payload)
    return payload
