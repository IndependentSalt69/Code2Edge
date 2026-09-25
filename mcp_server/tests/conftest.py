"""
mcp_server/tests/conftest.py

Shared fixtures: force all tools into mock mode before every test.
"""
import os
import pytest

@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    """Set every tool to mock mode and reset the parity scenario to 'pass'."""
    for var in [
        "CODE2EDGE_PROFILE_MODEL_MODE",
        "CODE2EDGE_INSPECT_PIPELINE_MODE",
        "CODE2EDGE_CHECK_TARGET_MODE",
        "CODE2EDGE_RUN_PARITY_TEST_MODE",
        "CODE2EDGE_BENCHMARK_TARGET_MODE",
        "CODE2EDGE_QUANTIZE_MODEL_MODE",
    ]:
        monkeypatch.setenv(var, "mock")
    monkeypatch.setenv("CODE2EDGE_MOCK_PARITY_SCENARIO", "pass")
    yield
