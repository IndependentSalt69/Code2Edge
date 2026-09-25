"""
mcp_server/_mock_loader.py

Load and stamp a fixture JSON file from mcp_server/mocks/<tool>.json.

Fixture files use __RUN_ID__ and __TIMESTAMP__ as sentinels which are
replaced at load time so every invocation gets fresh values.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from mcp_server._ids import new_run_id, utcnow_iso

_MOCKS_DIR = Path(__file__).resolve().parent / "mocks"


def load_mock(tool_name: str, run_id: str | None = None,
              overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the fixture for *tool_name*, stamp it, and return a deep copy.

    *overrides* is a flat dict of top-level keys to replace after stamping.
    """
    path = _MOCKS_DIR / f"{tool_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No mock fixture found: {path}")

    raw = path.read_text(encoding="utf-8")
    rid = run_id or new_run_id(tool_name)
    raw = raw.replace("__RUN_ID__", rid).replace("__TIMESTAMP__", utcnow_iso())
    payload: dict[str, Any] = json.loads(raw)

    if overrides:
        payload.update(overrides)

    return payload
