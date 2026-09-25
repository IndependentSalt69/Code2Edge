"""workflow/tests/test_approval.py"""
import pytest

from workflow import approval, state


def _run_id() -> str:
    return state.start_run()["run_id"]


def test_generation_blocked_before_any_approval():
    run_id = _run_id()
    with pytest.raises(approval.ApprovalRequiredError):
        approval.assert_generation_approved(run_id)


def test_approve_unblocks_generation():
    run_id = _run_id()
    out = approval.record_approval(run_id, "edge_readiness", approved=True, approver="person-c")
    assert out["workflow_action"] == "proceed"
    approval.assert_generation_approved(run_id)  # must not raise


def test_reject_keeps_generation_blocked():
    run_id = _run_id()
    out = approval.record_approval(run_id, "edge_readiness", approved=False, approver="person-c",
                                    notes="headroom too tight")
    assert out["workflow_action"] == "abort"
    with pytest.raises(approval.ApprovalRequiredError):
        approval.assert_generation_approved(run_id)

    status = state.get_run_status(run_id)
    assert status["status"] == "aborted"


def test_later_approve_overrides_earlier_reject():
    run_id = _run_id()
    approval.record_approval(run_id, "edge_readiness", approved=False, approver="person-c")
    approval.record_approval(run_id, "edge_readiness", approved=True, approver="person-c",
                              notes="re-ran after fixing headroom")
    approval.assert_generation_approved(run_id)  # must not raise


def test_approval_for_other_checkpoint_does_not_unblock_generation():
    run_id = _run_id()
    approval.record_approval(run_id, "post_host_parity", approved=True, approver="person-c")
    with pytest.raises(approval.ApprovalRequiredError):
        approval.assert_generation_approved(run_id)


def test_unknown_checkpoint_raises():
    run_id = _run_id()
    with pytest.raises(ValueError):
        approval.record_approval(run_id, "not_a_checkpoint", approved=True)
