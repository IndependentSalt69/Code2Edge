"""
Unit tests for Code2Edge Target Profile & Benchmark Tools.

Tests:
  - tools/target/check_target.py
  - tools/target/benchmark_target.py
  - contracts/target/target-profile.json schema conformance
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from tools.target.check_target import check_target, find_arduino_cli, get_installed_cores
from tools.target.benchmark_target import generate_benchmark_report

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_check_target_profile_structure():
    """Verify check_target() generates a well-structured target profile."""
    profile = check_target()

    assert profile["target_id"] == "arduino_uno_q_stm32u585"
    assert profile["board_name"] == "Arduino UNO Q"

    mcu = profile["mcu"]
    assert mcu["core"] == "ARM Cortex-M33"
    assert mcu["part_number"] == "STM32U585"
    assert mcu["clock_hz"] == 160000000
    assert mcu["flash_bytes"] == 2097152
    assert mcu["sram_bytes"] == 804864

    mem = profile["memory_limits"]
    assert mem["flash_limit_bytes"] == 2097152
    assert mem["sram_limit_bytes"] == 804864
    assert mem["max_tensor_arena_bytes"] == 524288
    assert mem["max_feature_buffer_bytes"] == 65536

    toolchain = profile["toolchain"]
    assert toolchain["name"] == "arduino-cli"
    assert toolchain["core_platform"] == "arduino:zephyr (1.0.0)"
    assert toolchain["board_fqbn"] == "arduino:zephyr:unoq"
    assert toolchain["status"] == "VERIFIED"

    bridge = profile["bridge"]
    assert bridge["transport"] == "UART"
    assert bridge["device_path"] == "/dev/ttyHS1"
    assert bridge["status"] == "VERIFIED"

    v_status = profile["verification_status"]
    assert v_status["hardware_bringup"] == "VERIFIED"
    assert v_status["dwt_benchmark_infrastructure"] == "VERIFIED"
    assert v_status["preprocessing_target_execution"] == "VERIFIED"
    assert v_status["host_parity"] == "VERIFIED"
    assert v_status["device_parity"] == "VERIFIED"
    assert v_status["model_inference_dscnn"] == "NOT_VERIFIED"


def test_target_profile_json_schema_validation():
    """Verify contracts/target/target-profile.json against target-profile.schema.json."""
    schema_path = REPO_ROOT / "contracts" / "target" / "target-profile.schema.json"
    profile_path = REPO_ROOT / "contracts" / "target" / "target-profile.json"

    assert schema_path.exists(), f"Missing schema: {schema_path}"
    assert profile_path.exists(), f"Missing profile: {profile_path}"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    jsonschema.validate(instance=profile, schema=schema)


def test_generate_benchmark_report_schema_validation():
    """Verify generate_benchmark_report() produces output matching benchmark-result.schema.json."""
    schema_path = REPO_ROOT / "contracts" / "target" / "benchmark-result.schema.json"
    assert schema_path.exists(), f"Missing schema: {schema_path}"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    report = generate_benchmark_report(
        target_id="arduino_uno_q_stm32u585",
        model_id="tiny-kws-dscnn-int8",
        run_type="MEASURED",
        inference_avg_ms=12.5,
        inference_min_ms=12.4,
        inference_max_ms=12.6,
        inference_std_ms=0.05,
        preprocessing_avg_ms=38.2,
        clock_cycles_inference=2000000,
        flash_used_bytes=135000,
        sram_used_bytes=70000,
        tensor_arena_bytes=35000,
        feature_buffer_bytes=25856,
        num_iterations=50,
        runtime="tflite_micro_cmsis_nn",
        quantization="int8",
    )

    jsonschema.validate(instance=report, schema=schema)
    assert report["run_type"] == "MEASURED"
    assert report["latency_ms"]["end_to_end_avg_ms"] == 50.7
    assert report["memory_bytes"]["flash_headroom_bytes"] == 2097152 - 135000
    assert report["memory_bytes"]["sram_headroom_bytes"] == 804864 - 70000
