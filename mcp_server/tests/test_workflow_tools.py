"""
mcp_server/tests/test_workflow_tools.py

start_run, get_run_status, gate_step and record_approval are local workflow
logic (no mock/real split) exposed as MCP tools. Verify each returns
schema-valid output and that gate_step's retry cap is reachable through MCP.
"""
import pytest

from workflow import state as workflow_state


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow_state, "RUNS_DIR", tmp_path / "runs")
    yield


@pytest.mark.asyncio
async def test_start_run_tool_schema_valid():
    from mcp_server.tools import start_run
    out = await start_run.run(requested_by="test-suite")
    assert out["tool"] == "start_run"
    assert out["status"] == "created"


@pytest.mark.asyncio
async def test_get_run_status_tool_schema_valid():
    from mcp_server.tools import get_run_status, start_run
    created = await start_run.run()
    out = await get_run_status.run(run_id=created["run_id"])
    assert out["tool"] == "get_run_status"
    assert out["run_id"] == created["run_id"]
    assert out["status"] == "running"


@pytest.mark.asyncio
async def test_record_approval_tool_schema_valid():
    from mcp_server.tools import record_approval, start_run
    created = await start_run.run()
    out = await record_approval.run(run_id=created["run_id"], checkpoint="edge_readiness",
                                    approved=True, approver="person-c")
    assert out["tool"] == "record_approval"
    assert out["workflow_action"] == "proceed"


@pytest.mark.asyncio
async def test_gate_step_tool_repair_then_escalate_schema_valid():
    from mcp_server.tools import gate_step, start_run
    created = await start_run.run()
    run_id = created["run_id"]

    def parity(status: str, attempt: int) -> dict:
        return {
            "status": status, "gate": "host", "attempt": attempt,
            "first_divergent_stage": None if status == "PASS" else "mel",
            "end_to_end": {"prediction_agreement": 0.97 if status == "PASS" else 0.8,
                           "accuracy_delta": 0.0 if status == "PASS" else -0.05},
            "diagnosis_hints": [] if status == "PASS" else [
                {"stage": "mel", "hypothesis": "filterbank scale mismatch", "evidence": "max_abs_diff=1.8"}
            ],
            "stages": [{"name": "mel", "status": "PASS" if status == "PASS" else "FAIL",
                        "max_abs_diff": 0.0001, "mean_abs_diff": 0.00001,
                        "shape_ref": [1], "shape_impl": [1]}],
        }

    d1 = await gate_step.run(run_id=run_id, gate="host", parity_result=parity("FAIL", 1))
    assert d1["decision"] == "repair"

    d2 = await gate_step.run(run_id=run_id, gate="host", parity_result=parity("FAIL", 2))
    assert d2["decision"] == "repair"

    d3 = await gate_step.run(run_id=run_id, gate="host", parity_result=parity("FAIL", 3))
    assert d3["decision"] == "escalate"
    assert d3["escalation_report_path"] is not None
