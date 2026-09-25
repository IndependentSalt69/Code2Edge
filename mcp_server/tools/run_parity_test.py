"""mcp_server/tools/run_parity_test.py"""
from __future__ import annotations

from mcp_server._mode import get_mode
from mcp_server._schema import validate_output
from mcp_server._ids import new_run_id, utcnow_iso

_TOOL = "run_parity_test"


async def run(gate: str, attempt: int,
              corpus_dir: str, ref_pipeline_path: str,
              impl_pipeline_path: str, run_id: str = "") -> dict:
    mode = get_mode(_TOOL)
    rid = run_id or new_run_id(_TOOL)
    ts = utcnow_iso()

    if mode == "mock":
        from mcp_server.scenarios import build_parity_payload
        payload = build_parity_payload(gate=gate, attempt=attempt,
                                       run_id=rid, timestamp=ts)
    else:
        from mcp_server.adapters import pipeline_adapter
        try:
            payload = pipeline_adapter.run_run_parity_test(
                gate=gate, attempt=attempt,
                corpus_dir=corpus_dir,
                ref_pipeline_path=ref_pipeline_path,
                impl_pipeline_path=impl_pipeline_path,
                run_id=rid,
            )
        except NotImplementedError as e:
            return {
                "schema_version": "1.0.0", "tool": _TOOL, "source": "real",
                "run_id": rid, "timestamp": ts,
                "status": "ERROR", "error_message": str(e),
            }

    validate_output(_TOOL, payload)
    return payload
