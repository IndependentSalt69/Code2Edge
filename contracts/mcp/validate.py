#!/usr/bin/env python3
"""
contracts/validate.py

Validates every example JSON in contracts/examples/ against its corresponding
JSON Schema in contracts/.

Usage:
    python contracts/validate.py            # from repo root
    python validate.py                      # from inside contracts/

Exit code 0 = all valid, 1 = one or more failures.
"""

import importlib.metadata
import json
import sys
from pathlib import Path

import jsonschema
from jsonschema import Draft202012Validator

# ---------------------------------------------------------------------------
# Locate directories regardless of CWD
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
CONTRACTS_DIR = THIS_FILE.parent
EXAMPLES_DIR = CONTRACTS_DIR / "examples"

# Map tool name → schema filename (Output schema lives in $defs/Output)
TOOL_SCHEMAS = {
    "benchmark_target": "benchmark_target.schema.json",
    "check_target":     "check_target.schema.json",
    "gate_step":        "gate_step.schema.json",
    "get_run_status":   "get_run_status.schema.json",
    "inspect_pipeline": "inspect_pipeline.schema.json",
    "profile_model":    "profile_model.schema.json",
    "quantize_model":   "quantize_model.schema.json",
    "record_approval":  "record_approval.schema.json",
    "run_parity_test":  "run_parity_test.schema.json",
    "start_run":        "start_run.schema.json",
}

EXAMPLE_FILES = {
    "benchmark_target": "benchmark_target.example.json",
    "check_target":     "check_target.example.json",
    "gate_step":        "gate_step.example.json",
    "get_run_status":   "get_run_status.example.json",
    "inspect_pipeline": "inspect_pipeline.example.json",
    "profile_model":    "profile_model.example.json",
    "quantize_model":   "quantize_model.example.json",
    "record_approval":  "record_approval.example.json",
    "run_parity_test":  "run_parity_test.example.json",
    "start_run":        "start_run.example.json",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_registry():
    """Build a jsonschema Registry (referencing library) from all schema files."""
    try:
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        resources = []
        for schema_file in CONTRACTS_DIR.glob("*.schema.json"):
            doc = load_json(schema_file)
            schema_id = doc.get("$id")
            if schema_id:
                resources.append((schema_id, Resource.from_contents(doc, default_specification=DRAFT202012)))
            # Also register by file URI
            file_uri = schema_file.as_uri()
            resources.append((file_uri, Resource.from_contents(doc, default_specification=DRAFT202012)))

        registry = Registry().with_resources(resources)
        return registry, True
    except ImportError:
        return None, False


def validate_example(tool: str, registry, use_registry: bool) -> tuple[bool, str]:
    schema_path  = CONTRACTS_DIR / TOOL_SCHEMAS[tool]
    example_path = EXAMPLES_DIR / EXAMPLE_FILES[tool]

    if not schema_path.exists():
        return False, f"MISSING schema: {schema_path.name}"
    if not example_path.exists():
        return False, f"MISSING example: {example_path.name}"

    full_schema = load_json(schema_path)
    example     = load_json(example_path)

    if "$defs" not in full_schema or "Output" not in full_schema["$defs"]:
        return False, f"Schema {schema_path.name} has no $defs/Output"

    # Build a self-contained schema that wraps the full document as the root
    # and immediately $refs into Output, so all sibling $defs are resolvable.
    wrapper_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": full_schema["$defs"],
        "$ref": "#/$defs/Output",
    }

    try:
        if use_registry:
            validator = Draft202012Validator(wrapper_schema, registry=registry)
        else:
            validator = Draft202012Validator(wrapper_schema)

        errors = list(validator.iter_errors(example))
        if not errors:
            return True, "OK"

        # Report the most specific (deepest) error
        worst = max(errors, key=lambda e: len(e.absolute_path))
        path_str = " -> ".join(str(p) for p in worst.absolute_path) or "(root)"
        return False, f"ValidationError at [{path_str}]: {worst.message}"

    except jsonschema.SchemaError as e:
        return False, f"SchemaError: {e.message}"
    except Exception as e:  # noqa: BLE001
        return False, f"Error: {e}"


def main() -> int:
    print(f"\nCode2Edge contract validation")
    print(f"Schemas dir : {CONTRACTS_DIR}")
    print(f"Examples dir: {EXAMPLES_DIR}")
    print(f"jsonschema  : {importlib.metadata.version('jsonschema')}")
    print("-" * 64)

    registry, use_registry = build_registry()
    if use_registry:
        print(f"  Using referencing.Registry for $ref resolution")
    else:
        print(f"  referencing not available, using inline wrapper schema")
    print("-" * 64)

    passed = 0
    failed = 0

    for tool in sorted(TOOL_SCHEMAS.keys()):
        ok, msg = validate_example(tool, registry, use_registry)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tool:25s}  {msg}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("-" * 64)
    print(f"  {passed} passed, {failed} failed\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
