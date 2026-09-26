#!/usr/bin/env python3
"""
Code2Edge Tier-2 On-Device Differential Parity Engine.

Parses physical UART tensor dumps emitted by the STM32U585 benchmark harness
(from tests/firmware/benchmark_harness/benchmark_harness.ino), decodes IEEE-754
float32 hex representations, and validates on-device feature extraction parity
against golden host reference tensors for the compiled fixture (e.g. yes.wav).

Conforms to:
  - contracts/target/device-parity-result.schema.json
  - contracts/target/target-profile.json
  - docs/validation.md (Tier-2 Stage-Wise Parity Tolerance Matrix)
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import struct
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UART_DUMP = REPO_ROOT / "evidence" / "parity" / "device_parity_yes_uart.txt"
DEFAULT_FIXTURE_WAV = REPO_ROOT / "reference" / "tiny-kws" / "app" / "examples" / "yes.wav"
DEFAULT_OUTPUT_REPORT = REPO_ROOT / "evidence" / "parity" / "device_parity_report.json"
NORMALIZATION_STATS = REPO_ROOT / "reference" / "normalization.json"
MEL_FILTERBANK_NPY = REPO_ROOT / "reference" / "artifacts" / "mel_filterbank.npy"
TARGET_PROFILE_PATH = REPO_ROOT / "contracts" / "target" / "target-profile.json"

# Tier-2 Device Parity Tolerances for Stage S3 (Normalized Features)
STAGE_S3_DEVICE_TOLERANCES = {
    "max_abs_error": 1e-3,      # 0.001
    "relative_error": 5e-3,     # 0.005 (0.5%)
    "min_cosine": 0.999990,     # 0.999990
    "tolerance_threshold": 1e-3,
}


@dataclass
class DeviceDump:
    """Parsed representation of a physical MCU UART parity tensor dump."""

    fixture_label: str
    fixture_checksum: str
    tensor_shape: List[int]
    tensor_size: int
    head_values: List[float]
    tail_values: List[float]
    tensor: np.ndarray  # float32 array of shape (64, 101) or tensor_shape
    raw_hex_count: int
    raw_metadata: Dict[str, str] = field(default_factory=dict)


class DeviceDumpParser:
    """Parser for UART parity dumps emitted by Code2Edge MCU firmware."""

    DUMP_START_MARKER = "CODE2EDGE_PARITY_DUMP_START"
    DUMP_END_MARKER = "CODE2EDGE_PARITY_DUMP_END"
    HEX_START_MARKER = "CODE2EDGE_PARITY_TENSOR_HEX_START"
    HEX_END_MARKER = "CODE2EDGE_PARITY_TENSOR_HEX_END"

    @classmethod
    def parse_text(cls, text: str, expected_label: Optional[str] = None) -> DeviceDump:
        """Parses UART text dump containing metadata and IEEE-754 hex values."""
        if cls.DUMP_START_MARKER not in text:
            raise ValueError(f"Malformed dump: Missing start marker '{cls.DUMP_START_MARKER}'")
        if cls.DUMP_END_MARKER not in text:
            raise ValueError(f"Malformed dump: Missing end marker '{cls.DUMP_END_MARKER}'")

        # Extract content between DUMP_START and DUMP_END
        start_pos = text.index(cls.DUMP_START_MARKER) + len(cls.DUMP_START_MARKER)
        end_pos = text.index(cls.DUMP_END_MARKER)
        dump_body = text[start_pos:end_pos]

        lines = [line.strip() for line in dump_body.splitlines() if line.strip()]

        if cls.HEX_START_MARKER not in lines:
            raise ValueError(f"Malformed dump: Missing hex start marker '{cls.HEX_START_MARKER}'")
        if cls.HEX_END_MARKER not in lines:
            raise ValueError(f"Malformed dump: Missing hex end marker '{cls.HEX_END_MARKER}'")

        hex_start_idx = lines.index(cls.HEX_START_MARKER)
        hex_end_idx = lines.index(cls.HEX_END_MARKER)

        if hex_end_idx <= hex_start_idx:
            raise ValueError("Malformed dump: Hex end marker precedes hex start marker")

        header_lines = lines[:hex_start_idx]
        hex_lines = lines[hex_start_idx + 1:hex_end_idx]

        # Parse key=value metadata
        metadata: Dict[str, str] = {}
        for line in header_lines:
            if "=" in line:
                k, v = line.split("=", 1)
                metadata[k.strip()] = v.strip()

        fixture_label = metadata.get("FIXTURE_LABEL", "")
        if not fixture_label:
            raise ValueError("Malformed dump: Missing 'FIXTURE_LABEL' in header")

        if expected_label and fixture_label != expected_label:
            raise ValueError(
                f"Fixture label mismatch: expected '{expected_label}', got '{fixture_label}'"
            )

        fixture_checksum = metadata.get("FIXTURE_CHECKSUM", "")

        # Parse tensor shape
        shape_str = metadata.get("TENSOR_SHAPE", "")
        if not shape_str:
            raise ValueError("Malformed dump: Missing 'TENSOR_SHAPE' in header")
        try:
            tensor_shape = [int(x.strip()) for x in shape_str.strip("[]").split(",") if x.strip()]
        except Exception as e:
            raise ValueError(f"Invalid TENSOR_SHAPE format '{shape_str}': {e}") from e

        # Parse tensor size
        size_str = metadata.get("TENSOR_SIZE", "")
        if not size_str:
            raise ValueError("Malformed dump: Missing 'TENSOR_SIZE' in header")
        try:
            tensor_size = int(size_str)
        except Exception as e:
            raise ValueError(f"Invalid TENSOR_SIZE format '{size_str}': {e}") from e

        # Validate shape vs size
        shape_product = int(np.prod(tensor_shape))
        if shape_product != tensor_size:
            raise ValueError(
                f"Inconsistent shape and size: shape {tensor_shape} has product {shape_product}, "
                f"but TENSOR_SIZE is {tensor_size}"
            )

        # Parse HEAD_VALUES and TAIL_VALUES
        head_values: List[float] = []
        if "HEAD_VALUES" in metadata:
            head_str = metadata["HEAD_VALUES"].strip("[]")
            head_values = [float(x.strip()) for x in head_str.split(",") if x.strip()]

        tail_values: List[float] = []
        if "TAIL_VALUES" in metadata:
            tail_str = metadata["TAIL_VALUES"].strip("[]")
            tail_values = [float(x.strip()) for x in tail_str.split(",") if x.strip()]

        # Parse hex words
        hex_tokens: List[str] = []
        for line in hex_lines:
            tokens = line.split()
            for token in tokens:
                clean_token = token.strip().upper()
                if clean_token:
                    if len(clean_token) != 8 or not re.match(r"^[0-9A-F]{8}$", clean_token):
                        raise ValueError(f"Invalid IEEE-754 32-bit hex word '{token}'")
                    hex_tokens.append(clean_token)

        raw_count = len(hex_tokens)
        if raw_count != tensor_size:
            raise ValueError(
                f"Element count mismatch: expected {tensor_size} hex words, but parsed {raw_count}"
            )

        # Decode big-endian IEEE-754 float32 values
        raw_bytes = bytes.fromhex("".join(hex_tokens))
        decoded_array = np.frombuffer(raw_bytes, dtype=">f4").astype(np.float32)

        if len(decoded_array) != tensor_size:
            raise ValueError(
                f"Decoded float count mismatch: expected {tensor_size}, got {len(decoded_array)}"
            )

        # Reshape to (64, 101) or target shape
        if len(tensor_shape) == 4 and tensor_shape[:2] == [1, 1]:
            reshaped_tensor = decoded_array.reshape((tensor_shape[2], tensor_shape[3]))
        else:
            reshaped_tensor = decoded_array.reshape(tuple(tensor_shape))

        # Validate head & tail values if present
        if head_values:
            actual_head = decoded_array[:len(head_values)]
            for i, (exp, act) in enumerate(zip(head_values, actual_head)):
                if abs(exp - act) > 1e-4:
                    raise ValueError(
                        f"Head value verification failed at [{i}]: expected {exp}, got {act}"
                    )

        if tail_values:
            actual_tail = decoded_array[-len(tail_values):]
            for i, (exp, act) in enumerate(zip(tail_values, actual_tail)):
                if abs(exp - act) > 1e-4:
                    raise ValueError(
                        f"Tail value verification failed at [{-len(tail_values)+i}]: expected {exp}, got {act}"
                    )

        return DeviceDump(
            fixture_label=fixture_label,
            fixture_checksum=fixture_checksum,
            tensor_shape=tensor_shape,
            tensor_size=tensor_size,
            head_values=head_values,
            tail_values=tail_values,
            tensor=reshaped_tensor,
            raw_hex_count=raw_count,
            raw_metadata=metadata,
        )

    @classmethod
    def parse_file(cls, file_path: Path, expected_label: Optional[str] = None) -> DeviceDump:
        """Parses a UART dump file."""
        if not file_path.exists():
            raise FileNotFoundError(f"UART dump file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return cls.parse_text(text, expected_label=expected_label)


class HostReferenceGenerator:
    """Generates ground-truth reference feature tensors directly from audio fixtures."""

    @staticmethod
    def load_audio_wav(wav_path: Path) -> np.ndarray:
        """Loads 16 kHz 16-bit mono PCM audio and normalizes to float32 in [-1.0, 1.0]."""
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio fixture WAV not found: {wav_path}")

        with wave.open(str(wav_path), "rb") as f:
            n_channels = f.getnchannels()
            sample_width = f.getsampwidth()
            framerate = f.getframerate()
            n_frames = f.getnframes()

            if n_channels != 1:
                raise ValueError(f"Expected mono audio (1 channel), got {n_channels}")
            if sample_width != 2:
                raise ValueError(f"Expected 16-bit PCM (sample width 2), got {sample_width}")
            if framerate != 16000:
                raise ValueError(f"Expected 16000 Hz sample rate, got {framerate}")

            raw_data = f.readframes(n_frames)
            pcm_i16 = np.frombuffer(raw_data, dtype=np.int16)

        if len(pcm_i16) != 16000:
            if len(pcm_i16) < 16000:
                pcm_i16 = np.pad(pcm_i16, (0, 16000 - len(pcm_i16)), mode="constant")
            else:
                pcm_i16 = pcm_i16[:16000]

        return pcm_i16.astype(np.float32) / 32768.0

    @classmethod
    def generate_pytorch_reference(cls, audio_f32: np.ndarray) -> np.ndarray:
        """
        Executes the frozen PyTorch reference pipeline (reference/tiny-kws/src/common.py)
        reproducing torchaudio LogMel and train-set normalization.
        """
        if audio_f32.shape != (16000,):
            raise ValueError(f"Expected audio shape (16000,), got {audio_f32.shape}")

        # 1. Periodic Hann STFT (n_fft=400, hop=160, reflect padding)
        wav_tensor = torch.from_numpy(audio_f32).unsqueeze(0)  # (1, 16000)
        window = torch.hann_window(400, periodic=True)
        stft = torch.stft(
            wav_tensor,
            n_fft=400,
            hop_length=160,
            win_length=400,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )  # (1, 201, 101)
        power_spec = stft.abs().pow(2.0)  # (1, 201, 101)

        # 2. Mel Filterbank Multiplication (201 -> 64 mels)
        if MEL_FILTERBANK_NPY.exists():
            mel_fb = np.load(str(MEL_FILTERBANK_NPY))
            mel_fb_tensor = torch.from_numpy(mel_fb).float()
        else:
            raise FileNotFoundError(f"Mel filterbank artifact not found: {MEL_FILTERBANK_NPY}")

        # (1, 101, 201) x (201, 64) -> (1, 101, 64) -> transpose to (1, 64, 101)
        mel_energy = torch.matmul(power_spec.transpose(1, 2), mel_fb_tensor).transpose(1, 2)

        # 3. Log Compression with epsilon = 1e-6
        log_mel = torch.log(mel_energy + 1e-6)

        # 4. Normalization using train stats
        if not NORMALIZATION_STATS.exists():
            raise FileNotFoundError(f"Normalization stats file not found: {NORMALIZATION_STATS}")

        with open(NORMALIZATION_STATS, "r", encoding="utf-8") as f:
            stats = json.load(f)

        norm_features = (log_mel - stats["mean"]) / stats["std"]
        return norm_features.squeeze(0).numpy().astype(np.float32)  # (64, 101)

    @classmethod
    def generate_host_c_reference(cls, audio_f32: np.ndarray) -> np.ndarray:
        """Executes the host-compiled C pipeline (src/pipeline/feature_extraction.c)."""
        from tools.run_host_parity import NativeCPipeline  # noqa: E402
        pipeline = NativeCPipeline()
        stages = pipeline.run_stages(audio_f32)
        return stages["S3_normalized_features"]


@dataclass
class StageParityResult:
    """Numerical metrics comparing a device tensor stage against host reference."""

    stage_id: str
    stage_name: str
    max_abs_error: float
    mean_abs_error: float
    relative_error: float
    cosine_similarity: float
    tolerance_threshold: float
    passed: bool
    first_divergent_index: Optional[List[int]] = None
    first_divergent_details: Optional[Dict[str, Any]] = None


class DeviceParityComparator:
    """Compares MCU on-device feature tensors against host reference ground-truth."""

    @staticmethod
    def compare(
        device_tensor: np.ndarray,
        reference_tensor: np.ndarray,
        tolerances: Optional[Dict[str, float]] = None,
        stage_id: str = "S3",
        stage_name: str = "S3_normalized_features",
    ) -> StageParityResult:
        """Calculates stage-wise differential parity metrics and verifies tolerances."""
        if device_tensor.shape != reference_tensor.shape:
            raise ValueError(
                f"Shape mismatch: device tensor {device_tensor.shape} != "
                f"reference tensor {reference_tensor.shape}"
            )

        tol = tolerances or STAGE_S3_DEVICE_TOLERANCES
        max_abs_tol = tol.get("max_abs_error", 1e-3)
        rel_tol = tol.get("relative_error", 5e-3)
        min_cosine_tol = tol.get("min_cosine", 0.999990)

        abs_diff = np.abs(reference_tensor - device_tensor)
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))

        ref_norm = float(np.linalg.norm(reference_tensor))
        diff_norm = float(np.linalg.norm(reference_tensor - device_tensor))
        rel_l2 = float(diff_norm / ref_norm) if ref_norm > 1e-12 else 0.0

        dot_prod = float(np.sum(reference_tensor * device_tensor))
        dev_norm = float(np.linalg.norm(device_tensor))
        if ref_norm > 1e-12 and dev_norm > 1e-12:
            cosine = float(dot_prod / (ref_norm * dev_norm))
        else:
            cosine = 1.0

        # Check thresholds
        passed = (
            max_abs <= max_abs_tol
            and rel_l2 <= rel_tol
            and cosine >= min_cosine_tol
        )

        first_divergent_idx = None
        first_divergent_details = None

        if not passed:
            # Find first index exceeding tolerance
            divergent_coords = np.argwhere(abs_diff > max_abs_tol)
            if len(divergent_coords) > 0:
                coord = divergent_coords[0]
                first_divergent_idx = [int(c) for c in coord]
                ref_val = float(reference_tensor[tuple(coord)])
                dev_val = float(device_tensor[tuple(coord)])
                diff_val = float(abs_diff[tuple(coord)])
                first_divergent_details = {
                    "index": first_divergent_idx,
                    "ref_value": ref_val,
                    "device_value": dev_val,
                    "abs_diff": diff_val,
                    "tolerance": max_abs_tol,
                }

        return StageParityResult(
            stage_id=stage_id,
            stage_name=stage_name,
            max_abs_error=max_abs,
            mean_abs_error=mean_abs,
            relative_error=rel_l2,
            cosine_similarity=cosine,
            tolerance_threshold=max_abs_tol,
            passed=passed,
            first_divergent_index=first_divergent_idx,
            first_divergent_details=first_divergent_details,
        )


def run_device_parity(
    uart_dump_path: Path = DEFAULT_UART_DUMP,
    fixture_wav_path: Path = DEFAULT_FIXTURE_WAV,
    output_report_path: Path = DEFAULT_OUTPUT_REPORT,
    reference_mode: str = "pytorch",
    verbose: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Executes end-to-end Tier-2 On-Device Differential Parity verification.

    Returns:
        (passed, report_dict)
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. Parse physical MCU UART dump
    if verbose:
        print("=" * 70)
        print("CODE2EDGE TIER-2 ON-DEVICE DIFFERENTIAL PARITY VERIFICATION")
        print("=" * 70)
        print(f"UART Dump Source:       {uart_dump_path}")
        print(f"Fixture Audio Source:    {fixture_wav_path}")
        print(f"Reference Backend:       {reference_mode}")

    device_dump = DeviceDumpParser.parse_file(uart_dump_path, expected_label="yes")

    if verbose:
        print(f"Decoded Device Tensor:   shape={device_dump.tensor.shape}, elements={device_dump.tensor_size}")
        print(f"Fixture Label / Checksum: {device_dump.fixture_label} / {device_dump.fixture_checksum}")

    # 2. Generate Host Reference Tensor
    audio_f32 = HostReferenceGenerator.load_audio_wav(fixture_wav_path)

    if reference_mode.lower() == "pytorch":
        ref_tensor = HostReferenceGenerator.generate_pytorch_reference(audio_f32)
        ref_description = "Frozen PyTorch reference pipeline (reference/tiny-kws/src/common.py)"
    elif reference_mode.lower() == "c_pipeline":
        ref_tensor = HostReferenceGenerator.generate_host_c_reference(audio_f32)
        ref_description = "Host-compiled C pipeline (src/pipeline/feature_extraction.c)"
    elif reference_mode.lower() == "both":
        ref_py = HostReferenceGenerator.generate_pytorch_reference(audio_f32)
        ref_c = HostReferenceGenerator.generate_host_c_reference(audio_f32)
        ref_tensor = ref_py
        ref_description = "PyTorch (primary) with Host C cross-validation"
    else:
        raise ValueError(f"Unknown reference mode: {reference_mode}")

    # 3. Compare Device Tensor vs Reference Tensor
    stage_result = DeviceParityComparator.compare(
        device_tensor=device_dump.tensor,
        reference_tensor=ref_tensor,
        tolerances=STAGE_S3_DEVICE_TOLERANCES,
        stage_id="S3",
        stage_name="S3_normalized_features",
    )

    overall_pass = stage_result.passed

    # 4. Format Machine-Readable Report matching device-parity-result.schema.json
    report: Dict[str, Any] = {
        "$schema": "contracts/target/device-parity-result.schema.json",
        "target_id": "arduino_uno_q_stm32u585",
        "corpus_id": "fixture_yes_v1",
        "timestamp": timestamp,
        "host_parity_precheck": True,
        "overall_pass": overall_pass,
        "total_samples_evaluated": 1,
        "samples_matched": 1 if overall_pass else 0,
        "samples_diverged": 0 if overall_pass else 1,
        "prediction_agreement_rate": 1.0 if overall_pass else 0.0,
        "first_divergent_stage": None if overall_pass else stage_result.stage_id,
        "stage_results": [
            {
                "stage_id": stage_result.stage_id,
                "stage_name": stage_result.stage_name,
                "max_abs_error": stage_result.max_abs_error,
                "mean_abs_error": stage_result.mean_abs_error,
                "relative_error": stage_result.relative_error,
                "cosine_similarity": stage_result.cosine_similarity,
                "tolerance_threshold": stage_result.tolerance_threshold,
                "passed": stage_result.passed,
            }
        ],
        "divergent_samples": [] if overall_pass else [
            {
                "sample_id": device_dump.fixture_label,
                "expected_label": device_dump.fixture_label,
                "actual_label": device_dump.fixture_label,
                "divergent_stage": stage_result.stage_id,
                "divergent_details": stage_result.first_divergent_details,
            }
        ],
        "metadata": {
            "fixture_label": device_dump.fixture_label,
            "fixture_checksum": device_dump.fixture_checksum,
            "tensor_shape": device_dump.tensor_shape,
            "tensor_elements": device_dump.tensor_size,
            "reference_description": ref_description,
            "uart_dump_path": str(uart_dump_path),
            "fixture_wav_path": str(fixture_wav_path),
        },
    }

    # Write output evidence report
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 5. Console Output
    if verbose:
        status_str = "PASS" if overall_pass else "FAIL"
        print("-" * 70)
        print(f"STAGE EVALUATION RESULTS [{stage_result.stage_id} - {stage_result.stage_name}]:")
        print(f"  Status:             [{status_str}]")
        print(f"  Max Absolute Error: {stage_result.max_abs_error:.6e}  (Tol: <= {STAGE_S3_DEVICE_TOLERANCES['max_abs_error']:.6e})")
        print(f"  Mean Absolute Error:{stage_result.mean_abs_error:.6e}")
        print(f"  Relative L2 Error:  {stage_result.relative_error:.6e}  (Tol: <= {STAGE_S3_DEVICE_TOLERANCES['relative_error']:.6e})")
        print(f"  Cosine Similarity:  {stage_result.cosine_similarity:.8f} (Tol: >= {STAGE_S3_DEVICE_TOLERANCES['min_cosine']:.6f})")

        if not overall_pass and stage_result.first_divergent_details:
            d = stage_result.first_divergent_details
            print(f"  First Divergence:   Index {d['index']} | Ref={d['ref_value']:.6f} | Dev={d['device_value']:.6f} | Diff={d['abs_diff']:.6e}")

        print("-" * 70)
        print(f"Overall Tier-2 Parity Status: {status_str}")
        print(f"Evidence Report Written To:   {output_report_path}")
        print("=" * 70)

    return overall_pass, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Code2Edge Tier-2 On-Device Differential Parity Verification Tool"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_UART_DUMP,
        help="Path to physical UART parity dump file (default: evidence/parity/device_parity_yes_uart.txt)",
    )
    parser.add_argument(
        "--fixture", "-f",
        type=Path,
        default=DEFAULT_FIXTURE_WAV,
        help="Path to fixture audio WAV file (default: reference/tiny-kws/app/examples/yes.wav)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT,
        help="Path for generated parity evidence JSON report (default: evidence/parity/device_parity_report.json)",
    )
    parser.add_argument(
        "--reference-mode", "-r",
        choices=["pytorch", "c_pipeline", "both"],
        default="pytorch",
        help="Reference generator mode (default: pytorch)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose diagnostic output",
    )

    args = parser.parse_args()

    try:
        passed, _ = run_device_parity(
            uart_dump_path=args.input,
            fixture_wav_path=args.fixture,
            output_report_path=args.output,
            reference_mode=args.reference_mode,
            verbose=not args.quiet,
        )
        return 0 if passed else 1
    except Exception as e:
        print(f"Error during device parity verification: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
