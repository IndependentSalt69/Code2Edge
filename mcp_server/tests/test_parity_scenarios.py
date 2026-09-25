"""
mcp_server/tests/test_parity_scenarios.py

Verify that every parity scenario behaves as specified:
  pass              → status PASS, all stages PASS, first_divergent_stage None
  always_fail       → status FAIL, first_divergent_stage='framing', diagnosis_hints present
  fail_framing_shape→ status FAIL, first_divergent_stage='framing', shape mismatch visible
  fail_mel_scale    → status FAIL, first_divergent_stage='mel', mel stage FAIL
  fail_then_pass    → attempt 1 → FAIL, attempt 2 (same run_id) → PASS
"""
import pytest


def _call_parity(monkeypatch, scenario: str, run_id: str = "test-run",
                 gate: str = "host", attempt: int = 1):
    monkeypatch.setenv("CODE2EDGE_MOCK_PARITY_SCENARIO", scenario)
    from mcp_server import scenarios
    import importlib
    importlib.reload(scenarios)  # pick up env var change
    from mcp_server._ids import utcnow_iso
    return scenarios.build_parity_payload(
        gate=gate, attempt=attempt, run_id=run_id, timestamp=utcnow_iso()
    )


def _validate(payload: dict) -> None:
    from mcp_server._schema import validate_output
    import jsonschema
    try:
        validate_output("run_parity_test", payload)
    except jsonschema.ValidationError as e:
        pytest.fail(f"schema validation failed: {e.message}")


# ── pass ──────────────────────────────────────────────────────────────────────

def test_scenario_pass(monkeypatch):
    p = _call_parity(monkeypatch, "pass", run_id="r-pass-001")
    _validate(p)
    assert p["status"] == "PASS"
    assert p["first_divergent_stage"] is None
    assert all(s["status"] == "PASS" for s in p["stages"])
    assert p["end_to_end"]["prediction_agreement"] >= 0.95
    assert len(p["diagnosis_hints"]) == 0


# ── always_fail ───────────────────────────────────────────────────────────────

def test_scenario_always_fail(monkeypatch):
    p = _call_parity(monkeypatch, "always_fail", run_id="r-af-001")
    _validate(p)
    assert p["status"] == "FAIL"
    assert p["first_divergent_stage"] == "framing"
    assert len(p["diagnosis_hints"]) > 0
    failing = [s for s in p["stages"] if s["status"] == "FAIL"]
    assert len(failing) >= 1
    assert p["end_to_end"]["prediction_agreement"] < 0.95


def test_scenario_always_fail_second_call_still_fails(monkeypatch):
    """always_fail must not flip to PASS on repeat calls."""
    for i in range(3):
        p = _call_parity(monkeypatch, "always_fail", run_id="r-af-repeat", attempt=i+1)
        assert p["status"] == "FAIL"


# ── fail_framing_shape ────────────────────────────────────────────────────────

def test_scenario_fail_framing_shape(monkeypatch):
    p = _call_parity(monkeypatch, "fail_framing_shape", run_id="r-fs-001")
    _validate(p)
    assert p["status"] == "FAIL"
    assert p["first_divergent_stage"] == "framing"

    framing = next(s for s in p["stages"] if s["name"] == "framing")
    assert framing["status"] == "FAIL"
    assert framing["shape_ref"] != framing["shape_impl"], "shape mismatch must be present"
    assert framing["shape_impl"][1] == 50  # impl produces 50 frames

    # Stages after framing must also FAIL (cascade)
    post_framing = [s for s in p["stages"] if s["order"] > 2]
    assert all(s["status"] == "FAIL" for s in post_framing)

    # Diagnosis hints must mention framing
    hint_stages = {h["stage"] for h in p["diagnosis_hints"]}
    assert "framing" in hint_stages


# ── fail_mel_scale ────────────────────────────────────────────────────────────

def test_scenario_fail_mel_scale(monkeypatch):
    p = _call_parity(monkeypatch, "fail_mel_scale", run_id="r-ms-001")
    _validate(p)
    assert p["status"] == "FAIL"
    assert p["first_divergent_stage"] == "mel"

    mel = next(s for s in p["stages"] if s["name"] == "mel")
    assert mel["status"] == "FAIL"
    assert mel["max_abs_diff"] > 1.0, "mel scale error should be large (>1.0)"

    # Stages before mel must PASS
    pre_mel = [s for s in p["stages"] if s["order"] < 4]
    assert all(s["status"] == "PASS" for s in pre_mel)

    # Hint must mention mel/slaney/htk
    assert any("mel" in h["stage"] or "Slaney" in h["hypothesis"]
               for h in p["diagnosis_hints"])


# ── fail_then_pass ────────────────────────────────────────────────────────────

def test_scenario_fail_then_pass(monkeypatch):
    """Same run_id: call 1 → FAIL, call 2 → PASS."""
    monkeypatch.setenv("CODE2EDGE_MOCK_PARITY_SCENARIO", "fail_then_pass")

    # Reset attempt counter so this test is isolated
    import mcp_server.scenarios as sc
    sc._attempt_counter.clear()
    from mcp_server._ids import utcnow_iso

    run_id = "r-ftp-isolated"

    p1 = sc.build_parity_payload(gate="host", attempt=1,
                                  run_id=run_id, timestamp=utcnow_iso())
    _validate(p1)
    assert p1["status"] == "FAIL", "first attempt must FAIL"

    p2 = sc.build_parity_payload(gate="host", attempt=2,
                                  run_id=run_id, timestamp=utcnow_iso())
    _validate(p2)
    assert p2["status"] == "PASS", "second attempt must PASS"


def test_scenario_fail_then_pass_different_run_ids_independent(monkeypatch):
    """Different run_ids must each start at attempt 1."""
    monkeypatch.setenv("CODE2EDGE_MOCK_PARITY_SCENARIO", "fail_then_pass")

    import mcp_server.scenarios as sc
    sc._attempt_counter.clear()
    from mcp_server._ids import utcnow_iso

    # run A: attempt 1 → FAIL
    pA1 = sc.build_parity_payload(gate="host", attempt=1,
                                   run_id="run-A", timestamp=utcnow_iso())
    assert pA1["status"] == "FAIL"

    # run B: attempt 1 → FAIL (independent counter)
    pB1 = sc.build_parity_payload(gate="host", attempt=1,
                                   run_id="run-B", timestamp=utcnow_iso())
    assert pB1["status"] == "FAIL"

    # run A: attempt 2 → PASS
    pA2 = sc.build_parity_payload(gate="host", attempt=2,
                                   run_id="run-A", timestamp=utcnow_iso())
    assert pA2["status"] == "PASS"


# ── invalid scenario ──────────────────────────────────────────────────────────

def test_invalid_scenario_raises(monkeypatch):
    monkeypatch.setenv("CODE2EDGE_MOCK_PARITY_SCENARIO", "not_a_scenario")
    import mcp_server.scenarios as sc
    from mcp_server._ids import utcnow_iso
    with pytest.raises(ValueError, match="Unknown CODE2EDGE_MOCK_PARITY_SCENARIO"):
        sc.build_parity_payload(gate="host", attempt=1,
                                run_id="x", timestamp=utcnow_iso())
