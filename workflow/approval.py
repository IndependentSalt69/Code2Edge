"""
workflow/approval.py

Human approval checkpoints. Code generation must not start until an
APPROVE has been recorded for the 'edge_readiness' checkpoint on this run.
"""
from __future__ import annotations

from typing import Any

from mcp_server._ids import utcnow_iso
from workflow import state

CHECKPOINTS = ("edge_readiness", "post_host_parity", "post_device_parity", "pre_submission")

GENERATION_CHECKPOINT = "edge_readiness"


class ApprovalRequiredError(PermissionError):
    pass


def record_approval(
    run_id: str,
    checkpoint: str,
    approved: bool,
    approver: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint {checkpoint!r}. Must be one of: {CHECKPOINTS}")

    ts = utcnow_iso()
    payload = {
        "schema_version": "1.0.0",
        "tool": "record_approval",
        "source": "real",
        "run_id": run_id,
        "timestamp": ts,
        "checkpoint": checkpoint,
        "approved": approved,
        "approver": approver,
        "notes": notes,
        "workflow_action": "proceed" if approved else "abort",
    }

    state.append_approval(run_id, payload)
    state.transition(
        run_id, "approval", "passed" if approved else "failed",
        detail=f"{checkpoint}: {'APPROVE' if approved else 'REJECT'} by {approver or 'unknown'}",
    )
    if not approved:
        state.set_run_status(run_id, "aborted", detail=f"{checkpoint} rejected by {approver or 'unknown'}")

    return payload


def is_approved(run_id: str, checkpoint: str) -> bool:
    """True if the most recent record_approval call for *checkpoint* was an APPROVE."""
    approvals = state.get_approvals(run_id, checkpoint=checkpoint)
    if not approvals:
        return False
    return bool(approvals[-1]["approved"])


def assert_generation_approved(run_id: str) -> None:
    """Raise ApprovalRequiredError unless the edge_readiness checkpoint is APPROVEd.

    Generation (workflow/WORKFLOW.md step e) must call this before writing
    anything into the deploy worktree.
    """
    if not is_approved(run_id, GENERATION_CHECKPOINT):
        raise ApprovalRequiredError(
            f"Generation blocked for run_id={run_id!r}: no APPROVE recorded for "
            f"checkpoint={GENERATION_CHECKPOINT!r}."
        )
