"""
mcp_server/_schema.py

Schema validation helpers.  Each MCP tool calls validate_output() before
returning its result.  Validation errors surface as MCP tool errors so the
agent sees structured feedback, not a silent bad payload.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator

# Contracts directory: two levels up from this file (mcp_server/ → repo root → contracts/)
_CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "contracts"


def _build_registry():
    """Build a referencing.Registry from all contracts/*.schema.json files."""
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    resources: list[tuple[str, Any]] = []
    for schema_file in _CONTRACTS_DIR.glob("*.schema.json"):
        doc = json.loads(schema_file.read_text(encoding="utf-8"))
        schema_id = doc.get("$id")
        if schema_id:
            resources.append(
                (schema_id, Resource.from_contents(doc, default_specification=DRAFT202012))
            )
        file_uri = schema_file.as_uri()
        resources.append(
            (file_uri, Resource.from_contents(doc, default_specification=DRAFT202012))
        )
    return Registry().with_resources(resources)


# Built once at import time; cheap after first load
_REGISTRY = _build_registry()

# Cache: tool_name → wrapper schema (full $defs + $ref to Output)
_SCHEMA_CACHE: dict[str, dict] = {}

_TOOL_SCHEMA_FILES = {
    "profile_model":    "profile_model.schema.json",
    "inspect_pipeline": "inspect_pipeline.schema.json",
    "check_target":     "check_target.schema.json",
    "run_parity_test":  "run_parity_test.schema.json",
    "gate_step":        "gate_step.schema.json",
    "record_approval":  "record_approval.schema.json",
    "benchmark_target": "benchmark_target.schema.json",
    "quantize_model":   "quantize_model.schema.json",
    "start_run":        "start_run.schema.json",
    "get_run_status":   "get_run_status.schema.json",
}


def _get_wrapper_schema(tool_name: str) -> dict:
    if tool_name not in _SCHEMA_CACHE:
        schema_file = _CONTRACTS_DIR / _TOOL_SCHEMA_FILES[tool_name]
        full = json.loads(schema_file.read_text(encoding="utf-8"))
        _SCHEMA_CACHE[tool_name] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": full["$defs"],
            "$ref": "#/$defs/Output",
        }
    return _SCHEMA_CACHE[tool_name]


def validate_output(tool_name: str, payload: dict) -> None:
    """Validate *payload* against the Output schema for *tool_name*.

    Raises jsonschema.ValidationError on failure.  The caller (tool handler)
    is responsible for converting this to an MCP error response.
    """
    wrapper = _get_wrapper_schema(tool_name)
    validator = Draft202012Validator(wrapper, registry=_REGISTRY)
    errors = list(validator.iter_errors(payload))
    if errors:
        worst = max(errors, key=lambda e: len(e.absolute_path))
        path_str = " -> ".join(str(p) for p in worst.absolute_path) or "(root)"
        raise jsonschema.ValidationError(
            f"[{tool_name}] schema validation failed at [{path_str}]: {worst.message}"
        )
