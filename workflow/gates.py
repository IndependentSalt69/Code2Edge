"""
workflow/gates.py

Generic parity gate: evaluates a run_parity_test output and decides whether
to continue, request a repair attempt, or escalate to a human. Written once,
used by both the host and device gates (workflow/WORKFLOW.md steps f and g).

The retry cap (3 attempts) is enforced HERE, in code — never left to the
calling agent to self-police.
"""
from __future__ import annotations

from typing import Any

from mcp_server._ids import utcnow_iso
from workflow import state

MAX_ATTEMPTS = 3


def gate_step(run_id: str, gate: str, parity_result: dict[str, Any]) -> dict[str, Any]:
    if gate not in ("host", "device"):
        raise ValueError(f"gate must be 'host' or 'device', got {gate!r}")
    if parity_result.get("gate") != gate:
        raise ValueError(
            f"parity_result['gate']={parity_result.get('gate')!r} does not match gate={gate!r}"
        )

    status = parity_result["status"]
    attempt = parity_result["attempt"]
    first_divergent = parity_result.get("first_divergent_stage")
    diagnosis_hints = parity_result.get("diagnosis_hints") or []
    ts = utcnow_iso()

    # Record this attempt in the run's parity history before deciding
    state.append_parity_result(run_id, gate, parity_result)

    focus_stage: str | None = None
    repair_hint: str | None = None
    escalation_report_path: str | None = None

    if status == "PASS":
        decision = "continue"
        remaining = 0

    elif status == "ERROR":
        decision = "escalate"
        remaining = 0
        escalation_report_path = _write_escalation_report(
            run_id, gate, reason="tool returned ERROR", parity_result=parity_result
        )

    else:  # FAIL
        focus_stage = first_divergent
        repair_hint = diagnosis_hints[0]["hypothesis"] if diagnosis_hints else None

        if attempt >= MAX_ATTEMPTS:
            decision = "escalate"
            remaining = 0
            escalation_report_path = _write_escalation_report(
                run_id, gate, reason="attempts exhausted", parity_result=parity_result
            )
        else:
            decision = "repair"
            remaining = MAX_ATTEMPTS - attempt

    payload = {
        "schema_version": "1.0.0",
        "tool": "gate_step",
        "source": "real",
        "run_id": run_id,
        "timestamp": ts,
        "decision": decision,
        "attempt": attempt,
        "max_attempts": MAX_ATTEMPTS,
        "remaining": remaining,
        "focus_stage": focus_stage,
        "repair_hint": repair_hint,
        "escalation_report_path": escalation_report_path,
    }

    state.append_gate_decision(run_id, {**payload, "gate": gate})

    if decision == "escalate":
        state.set_run_status(
            run_id, "escalated", detail=f"{gate} gate escalated at attempt {attempt} ({status})"
        )
    elif decision == "continue":
        state.transition(run_id, f"{gate}_parity", "passed", detail=f"attempt {attempt}")

    return payload


def _write_escalation_report(run_id: str, gate: str, reason: str, parity_result: dict[str, Any]) -> str:
    """Write workflow/runs/<run_id>/escalation_<gate>.md with every attempt's
    stage table and diagnosis hints, then return its path relative to the
    repo root (posix-style, matching the escalation_report_path field)."""
    history = state.get_parity_history(run_id, gate)
    report_dir = state.run_dir(run_id)
    report_path = report_dir / f"escalation_{gate}.md"

    lines: list[str] = [
        f"# Escalation report — {gate} parity gate",
        "",
        f"Run: `{run_id}`  Reason: **{reason}**  Attempts recorded: {len(history)}",
        "",
    ]

    for result in history:
        attempt = result.get("attempt")
        status = result.get("status")
        first_divergent = result.get("first_divergent_stage") or "none"
        end_to_end = result.get("end_to_end", {})
        lines += [
            f"## Attempt {attempt} — {status}",
            "",
            f"First divergent stage: `{first_divergent}`  "
            f"Prediction agreement: {end_to_end.get('prediction_agreement', 'n/a')}  "
            f"Accuracy delta: {end_to_end.get('accuracy_delta', 'n/a')}",
            "",
            "| stage | status | max_abs_diff | mean_abs_diff | shape_ref | shape_impl |",
            "|---|---|---|---|---|---|",
        ]
        for s in result.get("stages", []):
            lines.append(
                f"| {s.get('name')} | {s.get('status')} | {s.get('max_abs_diff')} | "
                f"{s.get('mean_abs_diff')} | {s.get('shape_ref')} | {s.get('shape_impl')} |"
            )
        lines.append("")

        hints = result.get("diagnosis_hints") or []
        if hints:
            lines.append("Diagnosis hints:")
            for h in hints:
                lines.append(f"- **{h.get('stage')}**: {h.get('hypothesis')} (evidence: {h.get('evidence')})")
            lines.append("")

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # RUNS_DIR is always <root>/workflow/runs, so two levels up from RUNS_DIR
    # is the root report_path should be expressed relative to (repo root in
    # production, a temp dir under tests).
    root = state.RUNS_DIR.parent.parent
    return report_path.relative_to(root).as_posix()
