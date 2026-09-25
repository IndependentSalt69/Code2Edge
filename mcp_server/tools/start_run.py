"""mcp_server/tools/start_run.py"""
from __future__ import annotations

from mcp_server._schema import validate_output
from workflow import state

_TOOL = "start_run"


async def run(requested_by: str = "") -> dict:
    payload = state.start_run(requested_by=requested_by)
    validate_output(_TOOL, payload)
    return payload
