"""
workflow/tests/conftest.py

Redirect workflow.state.RUNS_DIR to a temp directory for every test, so
tests never touch the real workflow/runs/ on disk.
"""
import pytest

from workflow import state


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "RUNS_DIR", tmp_path / "runs")
    yield
