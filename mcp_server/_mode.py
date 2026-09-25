"""
mcp_server/_mode.py

Per-tool mode resolution.

Each tool reads CODE2EDGE_<TOOL_UPPER>_MODE from the environment.
  mock  → load fixture from mcp_server/mocks/<tool>.json
  real  → delegate to the matching adapter

NEVER fall back silently from real to mock.
"""
from __future__ import annotations

import os

_TOOL_ENV_NAMES: dict[str, str] = {
    "profile_model":    "CODE2EDGE_PROFILE_MODEL_MODE",
    "inspect_pipeline": "CODE2EDGE_INSPECT_PIPELINE_MODE",
    "check_target":     "CODE2EDGE_CHECK_TARGET_MODE",
    "run_parity_test":  "CODE2EDGE_RUN_PARITY_TEST_MODE",
    "benchmark_target": "CODE2EDGE_BENCHMARK_TARGET_MODE",
    "quantize_model":   "CODE2EDGE_QUANTIZE_MODEL_MODE",
}

VALID_MODES = {"mock", "real"}


def get_mode(tool_name: str) -> str:
    """Return 'mock' or 'real' for *tool_name*.  Raises ValueError on bad value."""
    env_var = _TOOL_ENV_NAMES.get(tool_name, f"CODE2EDGE_{tool_name.upper()}_MODE")
    raw = os.environ.get(env_var, "mock").strip().lower()
    if raw not in VALID_MODES:
        raise ValueError(
            f"Invalid mode '{raw}' for {env_var}. Must be one of: {VALID_MODES}"
        )
    return raw
