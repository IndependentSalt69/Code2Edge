"""
mcp_server/_ids.py

Shared run-ID and timestamp utilities.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
