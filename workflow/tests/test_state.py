"""workflow/tests/test_state.py"""
from workflow import state


def test_start_run_creates_state_and_schema_shape():
    out = state.start_run(requested_by="tester")
    assert out["tool"] == "start_run"
    assert out["status"] == "created"
    assert out["current_stage"] == state.STAGES[0]
    assert out["stages"] == state.STAGES

    status = state.get_run_status(out["run_id"])
    assert status["run_id"] == out["run_id"]
    assert status["status"] == "running"
    assert status["current_stage"] == state.STAGES[0]
    assert status["stages"][0]["status"] == "in_progress"
    assert status["stages"][0]["entered_at"] is not None
    assert status["stages"][1]["status"] == "pending"
    assert len(status["history"]) == 1


def test_get_run_status_unknown_run_id_raises():
    try:
        state.get_run_status("run-does-not-exist")
        assert False, "expected RunNotFoundError"
    except state.RunNotFoundError:
        pass


def test_transition_updates_stage_and_history():
    out = state.start_run()
    run_id = out["run_id"]

    state.transition(run_id, "edge_readiness", "in_progress", detail="checking budget")
    status = state.get_run_status(run_id)
    assert status["current_stage"] == "edge_readiness"
    edge = next(s for s in status["stages"] if s["name"] == "edge_readiness")
    assert edge["status"] == "in_progress"
    assert edge["entered_at"] is not None
    assert edge["exited_at"] is None

    state.transition(run_id, "edge_readiness", "passed")
    status = state.get_run_status(run_id)
    edge = next(s for s in status["stages"] if s["name"] == "edge_readiness")
    assert edge["status"] == "passed"
    assert edge["exited_at"] is not None
    assert len(status["history"]) == 3  # run_started, in_progress, passed


def test_transition_unknown_stage_raises():
    out = state.start_run()
    try:
        state.transition(out["run_id"], "not_a_stage", "passed")
        assert False, "expected ValueError"
    except ValueError:
        pass
