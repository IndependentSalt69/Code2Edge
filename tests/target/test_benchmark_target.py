"""
Unit tests for Code2Edge Target Benchmark Runner (tools/target/benchmark_target.py).

Tests:
  - Compiler memory output parsing (application partition Flash/SRAM)
  - Statistical calculations (min, max, mean, sample standard deviation)
  - Report formatting and JSON Schema conformance against contracts/target/benchmark-result.schema.json
  - Contract validation and headroom calculations
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema
import pytest

from tools.target.benchmark_target import (
    STM32U585_FLASH_BYTES,
    STM32U585_SRAM_BYTES,
    calculate_sample_stats,
    generate_benchmark_report,
    parse_compiler_memory_usage,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_parse_compiler_memory_usage():
    """Verify parsing of standard Arduino CLI compiler memory usage output."""
    raw_output = (
        "Sketch uses 311336 bytes (39%) of program storage space. Maximum is 786432 bytes.\n"
        "Global variables use 242224 bytes (92%) of dynamic memory, leaving 19920 bytes for local variables. Maximum is 262144 bytes.\n"
        "Low memory available, stability problems may occur."
    )

    parsed = parse_compiler_memory_usage(raw_output)

    assert parsed["flash_used_bytes"] == 311336
    assert parsed["flash_partition_max_bytes"] == 786432
    assert parsed["flash_partition_pct"] == 39.0
    assert parsed["sram_used_bytes"] == 242224
    assert parsed["sram_partition_max_bytes"] == 262144
    assert parsed["sram_partition_pct"] == 92.0


def test_parse_compiler_memory_usage_invalid():
    """Verify ValueError is raised on invalid/empty compiler output."""
    with pytest.raises(ValueError, match="Failed to parse compiler memory usage"):
        parse_compiler_memory_usage("Compilation successful. No memory statistics reported.")


def test_calculate_sample_stats_empty():
    """Verify handling of empty list."""
    stats = calculate_sample_stats([])
    assert stats["min"] == 0.0
    assert stats["max"] == 0.0
    assert stats["mean"] == 0.0
    assert stats["std"] == 0.0


def test_calculate_sample_stats_single():
    """Verify handling of single element."""
    stats = calculate_sample_stats([100.0])
    assert stats["min"] == 100.0
    assert stats["max"] == 100.0
    assert stats["mean"] == 100.0
    assert stats["std"] == 0.0


def test_calculate_sample_stats_multiple():
    """Verify sample standard deviation calculation with N-1 degrees of freedom."""
    values = [10.0, 12.0, 23.0, 23.0, 16.0, 23.0, 21.0, 16.0]
    stats = calculate_sample_stats(values)

    expected_mean = sum(values) / len(values)  # 18.0
    expected_variance = sum((x - expected_mean) ** 2 for x in values) / (len(values) - 1)
    expected_std = math.sqrt(expected_variance)

    assert stats["min"] == 10.0
    assert stats["max"] == 23.0
    assert stats["mean"] == pytest.approx(expected_mean, rel=1e-5)
    assert stats["std"] == pytest.approx(expected_std, rel=1e-5)


def test_generate_benchmark_report_conformance():
    """Verify generated report conforms to benchmark-result.schema.json."""
    schema_path = REPO_ROOT / "contracts" / "target" / "benchmark-result.schema.json"
    assert schema_path.exists(), f"Missing schema: {schema_path}"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    report = generate_benchmark_report(
        target_id="arduino_uno_q_stm32u585",
        model_id="tiny-kws-dscnn-int8",
        run_type="MEASURED",
        inference_avg_ms=5024.4536,
        inference_min_ms=5024.4536,
        inference_max_ms=5024.4536,
        inference_std_ms=0.0,
        preprocessing_avg_ms=None,
        clock_cycles_inference=803912575,
        flash_used_bytes=311336,
        sram_used_bytes=242224,
        tensor_arena_bytes=166560,
        feature_buffer_bytes=25856,
        num_iterations=50,
        warmup_iterations=5,
        runtime="native_static_dscnn_runner",
        quantization="int8",
        toolchain_version="arduino-cli (1.5.2-rc.1), core: arduino:zephyr (1.0.0)",
        sample_prediction={
            "test_fixture": "yes.wav",
            "predicted_class_index": 2,
            "predicted_label": "yes",
        },
        validation={
            "expected_logits": [-30, -21, 83, -39, -28, -27, -20, -28, -30, -16, -29, -23],
            "expected_predicted_index": 2,
            "expected_predicted_label": "yes",
            "all_logits_matched": True,
            "all_argmax_matched": True,
            "all_labels_matched": True,
            "status": "PASS",
        },
        measurement_source="physical_stm32u585",
        serial_port="COM3",
        fixture="yes.wav",
        compiler_partition={
            "flash_partition_max_bytes": 786432,
            "flash_partition_usage_pct": 39.59,
            "sram_partition_max_bytes": 262144,
            "sram_partition_usage_pct": 92.40,
        },
    )

    # Validate against JSON schema
    jsonschema.validate(instance=report, schema=schema)

    assert report["run_type"] == "MEASURED"
    assert report["target_id"] == "arduino_uno_q_stm32u585"
    assert report["model_id"] == "tiny-kws-dscnn-int8"

    # Latency checks
    lat = report["latency_ms"]
    assert lat["inference_avg_ms"] == 5024.4536
    assert lat["preprocessing_avg_ms"] is None
    assert lat["end_to_end_avg_ms"] is None
    assert lat["clock_cycles_inference"] == 803912575

    # Memory checks
    mem = report["memory_bytes"]
    assert mem["flash_used_bytes"] == 311336
    assert mem["flash_total_bytes"] == STM32U585_FLASH_BYTES
    assert mem["flash_headroom_bytes"] == STM32U585_FLASH_BYTES - 311336
    assert mem["sram_used_bytes"] == 242224
    assert mem["sram_total_bytes"] == STM32U585_SRAM_BYTES
    assert mem["sram_headroom_bytes"] == STM32U585_SRAM_BYTES - 242224
    assert mem["tensor_arena_bytes"] == 166560
    assert mem["feature_buffer_bytes"] == 25856
    assert mem["static_ram_bytes"] == 242224 - 166560
    assert mem["stack_peak_bytes"] is None

    # Metadata checks
    meta = report["execution_metadata"]
    assert meta["clock_mhz"] == 160.0
    assert meta["num_iterations"] == 50
    assert meta["warmup_iterations"] == 5
    assert meta["measurement_source"] == "physical_stm32u585"
    assert meta["serial_port"] == "COM3"
    assert meta["fixture"] == "yes.wav"


def test_open_serial_connection_query_and_immediate_ready():
    """Verify open_serial_connection sends single-byte '?' query and completes on exact CODE2EDGE_READY."""
    import unittest.mock as mock
    from tools.target.benchmark_target import open_serial_connection

    mock_ser = mock.MagicMock()
    mock_ser.readline.side_effect = [b"CODE2EDGE_READY\r\n"]

    with mock.patch("serial.Serial", return_value=mock_ser):
        ser = open_serial_connection(port="COM3", baud_rate=115200, readiness_timeout=5.0)

        # 1. Verify single-byte initial query write
        mock_ser.write.assert_called_with(b"?")
        assert ser is mock_ser


def test_open_serial_connection_repeated_query_after_delay():
    """Verify open_serial_connection re-sends single-byte '?' query after 500 ms when waiting for readiness."""
    import unittest.mock as mock
    from tools.target.benchmark_target import open_serial_connection

    mock_ser = mock.MagicMock()
    # First returns empty/boot log, then after delay returns readiness marker
    mock_ser.readline.side_effect = [
        b"",
        b"[Boot] Feature extraction initialized\r\n",
        b"",
        b"CODE2EDGE_READY\r\n",
    ]

    with mock.patch("serial.Serial", return_value=mock_ser):
        ser = open_serial_connection(port="COM3", baud_rate=115200, readiness_timeout=5.0)
        assert ser is mock_ser
        # Verify single-byte query was sent at least once
        assert mock_ser.write.call_count >= 1
        assert mock_ser.write.call_args[0][0] == b"?"


def test_open_serial_connection_exact_marker_matching_and_timeout():
    """Verify open_serial_connection rejects partial substrings and raises TimeoutError on expiry."""
    import unittest.mock as mock
    from tools.target.benchmark_target import open_serial_connection

    mock_ser = mock.MagicMock()
    responses = [
        b"NOT_CODE2EDGE_READY\r\n",
        b"PREFIX_CODE2EDGE_READY_SUFFIX\r\n",
    ]
    mock_ser.readline.side_effect = lambda: responses.pop(0) if responses else b""

    with mock.patch("serial.Serial", return_value=mock_ser):
        with pytest.raises(TimeoutError, match="did not send 'CODE2EDGE_READY'"):
            open_serial_connection(port="COM3", baud_rate=115200, readiness_timeout=0.2)

        # Ensure serial port was closed on timeout
        mock_ser.close.assert_called_once()


def test_trigger_single_inference_sends_single_byte_and_parses():
    """Verify trigger_single_inference sends single-byte 'I' command and parses INFERENCE_JSON correctly."""
    import unittest.mock as mock
    from tools.target.benchmark_target import trigger_single_inference

    mock_ser = mock.MagicMock()
    mock_ser.in_waiting = 0
    mock_ser.readline.side_effect = [
        b"CODE2EDGE_INFERENCE_START\r\n",
        b"================================================================\r\n",
        b" OUTPUT INT8 LOGITS: [-30, -21, 83, -39, -28, -27, -20, -28, -30, -16, -29, -23]\r\n",
        b'INFERENCE_JSON={"fixture":"yes","predicted_index":2,"predicted_label":"yes","inference_cycles":803912575,"inference_us":5024453.60,"arena_bytes":166560,"logits":[-30,-21,83,-39,-28,-27,-20,-28,-30,-16,-29,-23],"dequantized":[0.0]}\r\n',
        b"CODE2EDGE_INFERENCE_END\r\n",
    ]

    result = trigger_single_inference(mock_ser, timeout_sec=5.0)

    # 1. Verify single-byte 'I' write
    mock_ser.write.assert_called_with(b"I")
    mock_ser.flush.assert_called_once()

    # 2. Verify parsed result
    assert result["predicted_index"] == 2
    assert result["predicted_label"] == "yes"
    assert result["inference_cycles"] == 803912575
    assert result["inference_us"] == 5024453.60
    assert result["logits"][2] == 83


def test_trigger_single_inference_timeout_diagnostic():
    """Verify trigger_single_inference raises TimeoutError with diagnostic details on timeout."""
    import unittest.mock as mock
    from tools.target.benchmark_target import trigger_single_inference

    mock_ser = mock.MagicMock()
    mock_ser.in_waiting = 0
    responses = [
        b"CODE2EDGE_READY\r\n",
    ]
    mock_ser.readline.side_effect = lambda: responses.pop(0) if responses else b""

    with pytest.raises(TimeoutError, match="Did not receive complete INFERENCE_JSON block"):
        trigger_single_inference(mock_ser, timeout_sec=0.1)
