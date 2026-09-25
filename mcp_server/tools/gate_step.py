"""mcp_server/tools/gate_step.py"""
from __future__ import annotations

from mcp_server._schema import validate_output
from workflow import gates

_TOOL = "gate_step"


async def run(run_id: str, gate: str, parity_result: dict) -> dict:
    payload = gates.gate_step(run_id=run_id, gate=gate, parity_result=parity_result)
    validate_output(_TOOL, payload)
    return payload
