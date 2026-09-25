"""
workflow/tests/test_gates.py

Covers the retry/escalation contract from contracts/gate_step.schema.json:
  PASS                        -> continue
  FAIL, attempt < 3           -> repair, remaining = 3 - attempt
  FAIL, attempt == 3          -> escalate, escalation report written
  ERROR (any attempt)         -> escalate immediately, escalation report written
"""
import pytest

from workflow import gates, state


def _parity_result(status: str, attempt: int, gate: str = "host",
                   first_divergent_stage: str | None = "mel") -> dict:
    return {
        "status": status,
        "gate": gate,
        "attempt": attempt,
        "first_divergent_stage": None if status == "PASS" else first_divergent_stage,
        "end_to_end": {"prediction_agreement": 0.97 if status == "PASS" else 0.80,
                       "accuracy_delta": 0.0 if status == "PASS" else -0.05},
        "diagnosis_hints": [] if status == "PASS" else [
            {"stage": first_divergent_stage, "hypothesis": "filterbank scale mismatch",
             "evidence": "max_abs_diff=1.8 at mel stage"}
        ],
        "stages": [
            {"name": "mel", "status": "PASS" if status == "PASS" else "FAIL",
             "max_abs_diff": 0.0001 if status == "PASS" else 1.8,
             "mean_abs_diff": 0.00001 if status == "PASS" else 0.47,
             "shape_ref": [500, 49, 40], "shape_impl": [500, 49, 40]},
        ],
    }


def _run_id() -> str:
    return state.start_run()["run_id"]


def test_pass_continues():
    run_id = _run_id()
    decision = gates.gate_step(run_id, "host", _parity_result("PASS", attempt=1))
    assert decision["decision"] == "continue"
    assert decision["remaining"] == 0
    assert decision["focus_stage"] is None
    assert decision["escalation_report_path"] is None


def test_fail_then_pass_repairs_then_continues():
    run_id = _run_id()

    d1 = gates.gate_step(run_id, "host", _parity_result("FAIL", attempt=1))
    assert d1["decision"] == "repair"
    assert d1["remaining"] == 2
    assert d1["focus_stage"] == "mel"
    assert d1["escalation_report_path"] is None

    d2 = gates.gate_step(run_id, "host", _parity_result("PASS", attempt=2))
    assert d2["decision"] == "continue"
    assert d2["remaining"] == 0


def test_fail_three_times_escalates_with_report():
    run_id = _run_id()

    d1 = gates.gate_step(run_id, "host", _parity_result("FAIL", attempt=1))
    assert d1["decision"] == "repair"
    assert d1["remaining"] == 2

    d2 = gates.gate_step(run_id, "host", _parity_result("FAIL", attempt=2))
    assert d2["decision"] == "repair"
    assert d2["remaining"] == 1

    d3 = gates.gate_step(run_id, "host", _parity_result("FAIL", attempt=3))
    assert d3["decision"] == "escalate"
    assert d3["remaining"] == 0
    assert d3["escalation_report_path"] is not None

    # escalation_report_path is relative to the root two levels above RUNS_DIR
    root = state.RUNS_DIR.parent.parent
    report_path = root / d3["escalation_report_path"]
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Attempt 1" in text and "Attempt 2" in text and "Attempt 3" in text
    assert "filterbank scale mismatch" in text

    status = state.get_run_status(run_id)
    assert status["status"] == "escalated"


def test_error_escalates_immediately_on_first_attempt():
    run_id = _run_id()
    parity_result = _parity_result("ERROR", attempt=1)
    parity_result["first_divergent_stage"] = None
    parity_result["diagnosis_hints"] = []

    decision = gates.gate_step(run_id, "host", parity_result)
    assert decision["decision"] == "escalate"
    assert decision["remaining"] == 0
    assert decision["escalation_report_path"] is not None

    status = state.get_run_status(run_id)
    assert status["status"] == "escalated"


def test_gate_mismatch_raises():
    run_id = _run_id()
    with pytest.raises(ValueError):
        gates.gate_step(run_id, "device", _parity_result("PASS", attempt=1, gate="host"))


def test_host_and_device_gates_track_independent_attempt_caps():
    run_id = _run_id()
    gates.gate_step(run_id, "host", _parity_result("FAIL", attempt=1, gate="host"))
    gates.gate_step(run_id, "host", _parity_result("PASS", attempt=2, gate="host"))

    # device gate for the same run starts its own attempt sequence
    d1 = gates.gate_step(run_id, "device", _parity_result("FAIL", attempt=1, gate="device"))
    assert d1["decision"] == "repair"
    assert d1["remaining"] == 2

    host_history = state.get_parity_history(run_id, "host")
    device_history = state.get_parity_history(run_id, "device")
    assert len(host_history) == 2
    assert len(device_history) == 1
