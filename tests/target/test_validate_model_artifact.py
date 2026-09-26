"""
Unit tests for Code2Edge model artifact validator (tools/target/validate_model_artifact.py).

NOTE ON SYNTHETIC TEST ARTIFACTS:
These unit tests utilize programmatically generated SYNTHETIC TFLite FlatBuffer
binaries (via tests.target.synthetic_tflite_builder) to verify all static validation
rules and contract gates. These synthetic models are mock structures designed strictly
for tooling testing and are NOT trained deployment models.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.target.synthetic_tflite_builder import build_synthetic_tflite
from tools.target.validate_model_artifact import (
    EXPECTED_INPUT_SHAPE,
    EXPECTED_LABELS,
    EXPECTED_OUTPUT_SHAPE,
    extract_bytes_from_c_header,
    validate_artifact_file,
    validate_model_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_valid_synthetic_quantized_model(tmp_path: Path):
    """Test that a fully compliant synthetic DS-CNN int8 model passes all validation checks."""
    model_bytes = build_synthetic_tflite(
        input_shape=(1, 1, 64, 101),
        output_shape=(1, 12),
        input_dtype=9,  # INT8
        output_dtype=9,  # INT8
        weight_dtype=9,  # INT8
        input_scale=0.05,
        input_zp=0,
        output_scale=0.00390625,
        output_zp=-128,
        opcodes=(3, 4, 9, 25),  # CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, SOFTMAX
        labels=EXPECTED_LABELS,
    )

    model_file = tmp_path / "valid_model.tflite"
    model_file.write_bytes(model_bytes)

    out_json = tmp_path / "evidence" / "model_validation.json"
    passed, report = validate_artifact_file(
        artifact_path=model_file,
        output_json=out_json,
        strict_quantization=True,
        quiet=True,
    )

    assert passed is True
    assert report["passed"] is True
    assert len(report["errors"]) == 0
    assert report["sha256"] == hashlib.sha256(model_bytes).hexdigest()
    assert report["file_size_bytes"] == len(model_bytes)

    # Check input / output tensor contracts
    assert report["input_tensor"]["shape"] == [1, 1, 64, 101]
    assert report["input_tensor"]["dtype"] == "INT8"
    assert report["input_tensor"]["scales"] == [pytest.approx(0.05)]
    assert report["input_tensor"]["zero_points"] == [0]

    assert report["output_tensor"]["shape"] == [1, 12]
    assert report["output_tensor"]["dtype"] == "INT8"
    assert report["output_tensor"]["scales"] == [pytest.approx(0.00390625)]
    assert report["output_tensor"]["zero_points"] == [-128]

    # Check quantization & target compatibility
    assert report["is_fully_quantized"] is True
    assert "CONV_2D" in report["operators_used"]
    assert "DEPTHWISE_CONV_2D" in report["operators_used"]
    assert "FULLY_CONNECTED" in report["operators_used"]
    assert "SOFTMAX" in report["operators_used"]
    assert report["unsupported_operators"] == []
    assert report["target"]["compatible"] is True

    # Check 12-class label contract
    assert report["labels"]["contract_passed"] is True
    assert report["labels"]["found_in_metadata"] == EXPECTED_LABELS

    # Check evidence JSON written to disk
    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["passed"] is True
    assert loaded["artifact_path"] == str(model_file)


def test_input_shape_mismatch_fails(tmp_path: Path):
    """Test that non-compliant input shape triggers validation failure."""
    bad_bytes = build_synthetic_tflite(input_shape=(1, 1, 32, 50))
    model_file = tmp_path / "bad_input_shape.tflite"
    model_file.write_bytes(bad_bytes)

    passed, report = validate_artifact_file(model_file, strict_quantization=True, quiet=True)
    assert passed is False
    assert any("Input shape mismatch" in err for err in report["errors"])


def test_output_shape_mismatch_fails(tmp_path: Path):
    """Test that non-compliant output shape triggers validation failure."""
    bad_bytes = build_synthetic_tflite(output_shape=(1, 10))
    model_file = tmp_path / "bad_output_shape.tflite"
    model_file.write_bytes(bad_bytes)

    passed, report = validate_artifact_file(model_file, strict_quantization=True, quiet=True)
    assert passed is False
    assert any("Output shape mismatch" in err for err in report["errors"])


def test_unquantized_float_model_handling(tmp_path: Path):
    """Test that float32 model is rejected in strict mode and warned in non-strict mode."""
    float_bytes = build_synthetic_tflite(
        input_dtype=0,  # FLOAT32
        output_dtype=0,  # FLOAT32
        weight_dtype=0,  # FLOAT32
    )
    model_file = tmp_path / "float32_model.tflite"
    model_file.write_bytes(float_bytes)

    # 1. Strict mode must fail
    passed_strict, report_strict = validate_artifact_file(
        model_file, strict_quantization=True, quiet=True
    )
    assert passed_strict is False
    assert report_strict["is_fully_quantized"] is False
    assert any("not fully quantized" in err for err in report_strict["errors"])

    # 2. Non-strict mode should allow with warning
    passed_loose, report_loose = validate_artifact_file(
        model_file, strict_quantization=False, quiet=True
    )
    assert passed_loose is True
    assert report_loose["is_fully_quantized"] is False
    assert any("unquantized" in w for w in report_loose["warnings"])


def test_unsupported_operator_detection(tmp_path: Path):
    """Test that operators unsupported by TFLite Micro / CMSIS-NN are flagged."""
    bad_op_bytes = build_synthetic_tflite(
        opcodes=(3, 120),  # CONV_2D, NON_MAX_SUPPRESSION_V4 (unsupported on MCU)
    )
    model_file = tmp_path / "unsupported_op_model.tflite"
    model_file.write_bytes(bad_op_bytes)

    passed, report = validate_artifact_file(model_file, strict_quantization=True, quiet=True)
    assert passed is False
    assert "NON_MAX_SUPPRESSION_V4" in report["unsupported_operators"]
    assert any("unsupported" in err for err in report["errors"])


def test_c_header_cross_verification(tmp_path: Path):
    """Test extracting and validating C header byte array against binary .tflite."""
    model_bytes = build_synthetic_tflite()
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(model_bytes)

    # Generate matching C header
    hex_bytes = ", ".join(f"0x{b:02x}" for b in model_bytes)
    header_content = f"""
#ifndef MODEL_DATA_H
#define MODEL_DATA_H
#include <stdint.h>
alignas(16) const unsigned char g_model_data[] = {{
  {hex_bytes}
}};
const int g_model_data_len = {len(model_bytes)};
#endif
"""
    header_file = tmp_path / "model_data.h"
    header_file.write_text(header_content, encoding="utf-8")

    # 1. Matching header passes
    passed, report = validate_artifact_file(
        artifact_path=model_file,
        header_path=header_file,
        quiet=True,
    )
    assert passed is True
    assert report["header_file"]["matches_tflite_bytes"] is True
    assert report["header_file"]["sha256"] == report["sha256"]

    # 2. Header with mismatching byte fails
    mismatch_bytes = bytearray(model_bytes)
    mismatch_bytes[10] ^= 0xFF
    bad_hex_bytes = ", ".join(f"0x{b:02x}" for b in mismatch_bytes)
    bad_header_file = tmp_path / "bad_model_data.h"
    bad_header_file.write_text(
        f"const unsigned char g_model_data[] = {{ {bad_hex_bytes} }};",
        encoding="utf-8",
    )

    passed_bad, report_bad = validate_artifact_file(
        artifact_path=model_file,
        header_path=bad_header_file,
        quiet=True,
    )
    assert passed_bad is False
    assert any("SHA-256 mismatch" in err for err in report_bad["errors"])


def test_label_contract_mismatch(tmp_path: Path):
    """Test that mismatched 12-class label names in metadata trigger error."""
    bad_labels = ["yes", "no", "up", "down"]  # only 4 labels
    model_bytes = build_synthetic_tflite(labels=bad_labels)
    model_file = tmp_path / "bad_labels_model.tflite"
    model_file.write_bytes(model_bytes)

    passed, report = validate_artifact_file(model_file, quiet=True)
    assert passed is False
    assert any("label contract mismatch" in err for err in report["errors"])


def test_cli_execution_success_and_failure(tmp_path: Path):
    """Test validate_model_artifact.py CLI entrypoint and exit codes."""
    valid_bytes = build_synthetic_tflite()
    valid_file = tmp_path / "cli_valid.tflite"
    valid_file.write_bytes(valid_bytes)

    bad_bytes = build_synthetic_tflite(input_shape=(1, 1, 10, 10))
    bad_file = tmp_path / "cli_bad.tflite"
    bad_file.write_bytes(bad_bytes)

    script_path = REPO_ROOT / "tools" / "target" / "validate_model_artifact.py"
    out_evidence = tmp_path / "cli_evidence.json"

    # 1. Valid model -> exit code 0
    res_valid = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(valid_file),
            "--output-json",
            str(out_evidence),
        ],
        capture_output=True,
        text=True,
    )
    assert res_valid.returncode == 0
    assert "PASS" in res_valid.stdout
    assert out_evidence.exists()

    # 2. Invalid model -> exit code 1
    out_bad_evidence = tmp_path / "cli_bad_evidence.json"
    res_bad = subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(bad_file),
            "--output-json",
            str(out_bad_evidence),
        ],
        capture_output=True,
        text=True,
    )
    assert res_bad.returncode == 1
    assert "FAIL" in res_bad.stdout
