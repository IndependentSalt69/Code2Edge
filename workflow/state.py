"""
workflow/state.py

Run state machine for the Code2Edge deployment workflow.

Every run gets a directory workflow/runs/<run_id>/ (gitignored) holding
state.json.  All stage transitions, gate decisions, approvals and parity
results are timestamped and appended to that file so a run's full history
survives process restarts.

Stage order (fixed, matches workflow/WORKFLOW.md):
    analysis -> edge_readiness -> approval -> generation -> host_parity ->
    mcu_build -> device_parity -> benchmark -> report -> pr
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp_server._ids import new_run_id, utcnow_iso

STAGES: list[str] = [
    "analysis",
    "edge_readiness",
    "approval",
    "generation",
    "host_parity",
    "mcu_build",
    "device_parity",
    "benchmark",
    "report",
    "pr",
]

STAGE_STATUSES = {"pending", "in_progress", "passed", "failed", "skipped"}
RUN_STATUSES = {"running", "completed", "failed", "escalated", "aborted"}

_REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _REPO_ROOT / "workflow" / "runs"


class RunNotFoundError(FileNotFoundError):
    pass


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def _state_path(run_id: str) -> Path:
    return run_dir(run_id) / "state.json"


def _write_state(run_id: str, state: dict[str, Any]) -> None:
    path = _state_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _read_state(run_id: str) -> dict[str, Any]:
    path = _state_path(run_id)
    if not path.exists():
        raise RunNotFoundError(f"No run found for run_id={run_id!r}")
    return json.loads(path.read_text(encoding="utf-8"))


# ── run lifecycle ────────────────────────────────────────────────────────────

def start_run(requested_by: str = "") -> dict[str, Any]:
    """Create a new run, initialise its state file, return the start_run output."""
    run_id = new_run_id("run")
    ts = utcnow_iso()
    state: dict[str, Any] = {
        "run_id": run_id,
        "created_at": ts,
        "updated_at": ts,
        "requested_by": requested_by or None,
        "current_stage": STAGES[0],
        "status": "running",
        "stages": [
            {"name": s, "status": "pending", "entered_at": None, "exited_at": None}
            for s in STAGES
        ],
        "history": [
            {"stage": STAGES[0], "event": "run_started", "timestamp": ts, "detail": None}
        ],
        "approvals": [],
        "parity_history": {"host": [], "device": []},
        "gate_decisions": [],
    }
    state["stages"][0]["status"] = "in_progress"
    state["stages"][0]["entered_at"] = ts
    _write_state(run_id, state)

    return {
        "schema_version": "1.0.0",
        "tool": "start_run",
        "source": "real",
        "run_id": run_id,
        "timestamp": ts,
        "status": "created",
        "current_stage": STAGES[0],
        "stages": list(STAGES),
    }


def get_run_status(run_id: str) -> dict[str, Any]:
    state = _read_state(run_id)
    return {
        "schema_version": "1.0.0",
        "tool": "get_run_status",
        "source": "real",
        "run_id": run_id,
        "timestamp": utcnow_iso(),
        "status": state["status"],
        "current_stage": state["current_stage"],
        "stages": state["stages"],
        "history": state["history"],
    }


def transition(run_id: str, stage: str, status: str, detail: str | None = None) -> None:
    """Record a stage transition. status is one of STAGE_STATUSES."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Must be one of: {STAGES}")
    if status not in STAGE_STATUSES:
        raise ValueError(f"Unknown stage status {status!r}. Must be one of: {sorted(STAGE_STATUSES)}")

    state = _read_state(run_id)
    ts = utcnow_iso()
    entry = next(s for s in state["stages"] if s["name"] == stage)
    if status == "in_progress" and entry["entered_at"] is None:
        entry["entered_at"] = ts
    entry["status"] = status
    if status in ("passed", "failed", "skipped"):
        entry["exited_at"] = ts

    state["current_stage"] = stage
    state["updated_at"] = ts
    state["history"].append({"stage": stage, "event": status, "timestamp": ts, "detail": detail})
    _write_state(run_id, state)


def set_run_status(run_id: str, status: str, detail: str | None = None) -> None:
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status {status!r}. Must be one of: {sorted(RUN_STATUSES)}")
    state = _read_state(run_id)
    ts = utcnow_iso()
    state["status"] = status
    state["updated_at"] = ts
    state["history"].append(
        {"stage": state["current_stage"], "event": f"run_{status}", "timestamp": ts, "detail": detail}
    )
    _write_state(run_id, state)


# ── parity history (per gate, across repair attempts) ──────────────────────

def append_parity_result(run_id: str, gate: str, parity_result: dict[str, Any]) -> None:
    state = _read_state(run_id)
    state["parity_history"].setdefault(gate, []).append(parity_result)
    state["updated_at"] = utcnow_iso()
    _write_state(run_id, state)


def get_parity_history(run_id: str, gate: str) -> list[dict[str, Any]]:
    state = _read_state(run_id)
    return list(state["parity_history"].get(gate, []))


# ── gate decisions ──────────────────────────────────────────────────────────

def append_gate_decision(run_id: str, decision: dict[str, Any]) -> None:
    state = _read_state(run_id)
    state["gate_decisions"].append(decision)
    state["updated_at"] = utcnow_iso()
    _write_state(run_id, state)


def get_gate_decisions(run_id: str, gate: str | None = None) -> list[dict[str, Any]]:
    state = _read_state(run_id)
    decisions = state["gate_decisions"]
    if gate is not None:
        decisions = [d for d in decisions if d.get("gate") == gate]
    return list(decisions)


# ── approvals ────────────────────────────────────────────────────────────────

def append_approval(run_id: str, approval: dict[str, Any]) -> None:
    state = _read_state(run_id)
    state["approvals"].append(approval)
    state["updated_at"] = utcnow_iso()
    _write_state(run_id, state)


def get_approvals(run_id: str, checkpoint: str | None = None) -> list[dict[str, Any]]:
    state = _read_state(run_id)
    approvals = state["approvals"]
    if checkpoint is not None:
        approvals = [a for a in approvals if a.get("checkpoint") == checkpoint]
    return list(approvals)
