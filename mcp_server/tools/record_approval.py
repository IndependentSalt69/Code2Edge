"""mcp_server/tools/record_approval.py"""
from __future__ import annotations

from mcp_server._schema import validate_output
from workflow import approval

_TOOL = "record_approval"


async def run(run_id: str, checkpoint: str, approved: bool,
              approver: str = "", notes: str = "") -> dict:
    payload = approval.record_approval(
        run_id=run_id, checkpoint=checkpoint, approved=approved,
        approver=approver or None, notes=notes or None,
    )
    validate_output(_TOOL, payload)
    return payload
