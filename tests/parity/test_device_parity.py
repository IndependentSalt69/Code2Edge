"""
Unit tests for Code2Edge Tier-2 On-Device Differential Parity Comparator.

Validates parsing, error handling, IEEE-754 float32 decoding, reference generation,
and stage-wise tolerance checks for STM32U585 UART tensor dumps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.target.run_device_parity import (
    DEFAULT_FIXTURE_WAV,
    DEFAULT_UART_DUMP,
    STAGE_S3_DEVICE_TOLERANCES,
    DeviceDump,
    DeviceDumpParser,
    DeviceParityComparator,
    HostReferenceGenerator,
    run_device_parity,
)


def create_mock_uart_dump(
    fixture_label: str = "yes",
    fixture_checksum: str = "0x8B8CD8",
    tensor_shape: str = "[1,1,64,101]",
    tensor_size: int = 6464,
    float_values: np.ndarray | None = None,
    corrupt_hex: bool = False,
    truncate_hex: int = 0,
    omit_start_marker: bool = False,
    omit_end_marker: bool = False,
) -> str:
    """Helper to generate synthetic or perturbed UART parity dumps."""
    if float_values is None:
        float_values = np.zeros(tensor_size, dtype=np.float32)

    lines = []
    if not omit_start_marker:
        lines.append("CODE2EDGE_PARITY_DUMP_START")
    lines.append(f"FIXTURE_LABEL={fixture_label}")
    lines.append(f"FIXTURE_CHECKSUM={fixture_checksum}")
    lines.append(f"TENSOR_SHAPE={tensor_shape}")
    lines.append(f"TENSOR_SIZE={tensor_size}")

    head_str = ",".join(f"{float_values[i]:.6f}" for i in range(min(5, len(float_values))))
    tail_str = ",".join(f"{float_values[-5+i]:.6f}" for i in range(min(5, len(float_values))))
    lines.append(f"HEAD_VALUES=[{head_str}]")
    lines.append(f"TAIL_VALUES=[{tail_str}]")

    lines.append("CODE2EDGE_PARITY_TENSOR_HEX_START")

    # Encode float32 to big-endian hex strings
    hex_words = []
    for val in float_values:
        raw_bytes = struct_pack_float(val)
        hex_words.append(raw_bytes.hex().upper())

    if corrupt_hex:
        hex_words[10] = "NOT_HEX!"

    if truncate_hex > 0:
        hex_words = hex_words[:-truncate_hex]

    # Chunk into 16 words per line
    for i in range(0, len(hex_words), 16):
        chunk = hex_words[i:i + 16]
        lines.append(" ".join(chunk))

    lines.append("CODE2EDGE_PARITY_TENSOR_HEX_END")
    if not omit_end_marker:
        lines.append("CODE2EDGE_PARITY_DUMP_END")

    return "\n".join(lines)


def struct_pack_float(val: float) -> bytes:
    import struct
    return struct.pack(">f", float(val))


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_valid_6464_uart_dump_parsing():
    """Verify parsing and decoding of the physical yes.wav UART dump."""
    assert DEFAULT_UART_DUMP.exists(), f"Missing physical UART dump file: {DEFAULT_UART_DUMP}"

    dump = DeviceDumpParser.parse_file(DEFAULT_UART_DUMP, expected_label="yes")

    assert dump.fixture_label == "yes"
    assert dump.fixture_checksum == "0x8B8CD8"
    assert dump.tensor_shape == [1, 1, 64, 101]
    assert dump.tensor_size == 6464
    assert dump.raw_hex_count == 6464
    assert dump.tensor.shape == (64, 101)
    assert dump.tensor.dtype == np.float32
    assert np.isfinite(dump.tensor).all()

    # Verify head and tail values
    assert len(dump.head_values) == 5
    assert len(dump.tail_values) == 5
    np.testing.assert_allclose(dump.tensor.flat[:5], dump.head_values, atol=1e-4)
    np.testing.assert_allclose(dump.tensor.flat[-5:], dump.tail_values, atol=1e-4)


def test_malformed_truncated_dump_missing_markers():
    """Verify that missing start or end markers raise ValueError."""
    # Missing dump start marker
    bad_dump1 = create_mock_uart_dump(omit_start_marker=True)
    with pytest.raises(ValueError, match="Missing start marker"):
        DeviceDumpParser.parse_text(bad_dump1)

    # Missing dump end marker
    bad_dump2 = create_mock_uart_dump(omit_end_marker=True)
    with pytest.raises(ValueError, match="Missing end marker"):
        DeviceDumpParser.parse_text(bad_dump2)


def test_malformed_dump_corrupt_hex():
    """Verify that invalid hex words raise ValueError."""
    bad_dump = create_mock_uart_dump(corrupt_hex=True)
    with pytest.raises(ValueError, match="Invalid IEEE-754 32-bit hex word"):
        DeviceDumpParser.parse_text(bad_dump)


def test_wrong_element_count_dump():
    """Verify that a truncated or oversized hex list raises ValueError."""
    # 4 elements truncated (6460 instead of 6464)
    truncated_dump = create_mock_uart_dump(tensor_size=6464, truncate_hex=4)
    with pytest.raises(ValueError, match="Element count mismatch"):
        DeviceDumpParser.parse_text(truncated_dump)


def test_wrong_shape_metadata():
    """Verify that inconsistent shape and size raises ValueError."""
    # Shape product 3232 != size 6464
    bad_shape_dump = create_mock_uart_dump(
        tensor_shape="[1,1,32,101]",
        tensor_size=6464,
    )
    with pytest.raises(ValueError, match="Inconsistent shape and size"):
        DeviceDumpParser.parse_text(bad_shape_dump)


def test_wrong_fixture_label():
    """Verify that fixture label mismatch raises ValueError."""
    dump_text = create_mock_uart_dump(fixture_label="no")
    with pytest.raises(ValueError, match="Fixture label mismatch"):
        DeviceDumpParser.parse_text(dump_text, expected_label="yes")


def test_host_reference_generation_from_fixture_wav():
    """Verify that HostReferenceGenerator generates correct (64, 101) tensor from yes.wav."""
    assert DEFAULT_FIXTURE_WAV.exists(), f"Missing fixture WAV: {DEFAULT_FIXTURE_WAV}"

    audio_f32 = HostReferenceGenerator.load_audio_wav(DEFAULT_FIXTURE_WAV)
    assert audio_f32.shape == (16000,)
    assert audio_f32.dtype == np.float32

    ref_tensor = HostReferenceGenerator.generate_pytorch_reference(audio_f32)
    assert ref_tensor.shape == (64, 101)
    assert ref_tensor.dtype == np.float32
    assert np.isfinite(ref_tensor).all()


def test_device_parity_comparison_pass_on_physical_dump():
    """Verify that the physical yes.wav UART dump passes Tier-2 parity checks."""
    dump = DeviceDumpParser.parse_file(DEFAULT_UART_DUMP)
    audio_f32 = HostReferenceGenerator.load_audio_wav(DEFAULT_FIXTURE_WAV)
    ref_tensor = HostReferenceGenerator.generate_pytorch_reference(audio_f32)

    result = DeviceParityComparator.compare(
        device_tensor=dump.tensor,
        reference_tensor=ref_tensor,
    )

    assert result.passed is True
    assert result.max_abs_error <= STAGE_S3_DEVICE_TOLERANCES["max_abs_error"]
    assert result.relative_error <= STAGE_S3_DEVICE_TOLERANCES["relative_error"]
    assert result.cosine_similarity >= STAGE_S3_DEVICE_TOLERANCES["min_cosine"]
    assert result.first_divergent_index is None


def test_device_parity_comparison_perturbed_tensor_fail():
    """Verify that a deliberately perturbed tensor fails and identifies the first divergent index."""
    dump = DeviceDumpParser.parse_file(DEFAULT_UART_DUMP)
    audio_f32 = HostReferenceGenerator.load_audio_wav(DEFAULT_FIXTURE_WAV)
    ref_tensor = HostReferenceGenerator.generate_pytorch_reference(audio_f32)

    perturbed_dev_tensor = dump.tensor.copy()
    # Inject perturbation at [10, 25] exceeding 1e-3 tolerance
    perturbed_dev_tensor[10, 25] += 0.05

    result = DeviceParityComparator.compare(
        device_tensor=perturbed_dev_tensor,
        reference_tensor=ref_tensor,
    )

    assert result.passed is False
    assert result.max_abs_error >= 0.049
    assert result.first_divergent_index == [10, 25]
    assert result.first_divergent_details is not None
    assert result.first_divergent_details["index"] == [10, 25]
    assert result.first_divergent_details["abs_diff"] >= 0.049


def test_end_to_end_run_device_parity(tmp_path):
    """Verify full end-to-end execution and report generation."""
    out_report = tmp_path / "device_parity_report.json"

    passed, report = run_device_parity(
        uart_dump_path=DEFAULT_UART_DUMP,
        fixture_wav_path=DEFAULT_FIXTURE_WAV,
        output_report_path=out_report,
        reference_mode="pytorch",
        verbose=False,
    )

    assert passed is True
    assert out_report.exists()
    assert report["overall_pass"] is True
    assert report["total_samples_evaluated"] == 1
    assert report["samples_matched"] == 1
    assert report["samples_diverged"] == 0
    assert report["prediction_agreement_rate"] == 1.0
    assert len(report["stage_results"]) == 1
    assert report["stage_results"][0]["passed"] is True
    assert report["stage_results"][0]["stage_id"] == "S3"
