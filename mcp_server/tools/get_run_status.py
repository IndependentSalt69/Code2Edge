"""mcp_server/tools/get_run_status.py"""
from __future__ import annotations

from mcp_server._schema import validate_output
from workflow import state

_TOOL = "get_run_status"


async def run(run_id: str) -> dict:
    """Raises workflow.state.RunNotFoundError (surfaced as an MCP tool error)
    if run_id does not exist — there is no schema-valid ERROR shape for a
    lookup on a run that was never created."""
    payload = state.get_run_status(run_id)
    validate_output(_TOOL, payload)
    return payload
