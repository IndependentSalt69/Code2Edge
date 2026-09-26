#!/usr/bin/env python3
"""
Code2Edge: Physical Hardware DS-CNN Inference Runner & Comparator.
Executes physical on-device inference on the STM32U585 via USB CDC serial (COM3),
captures UART telemetry, and compares on-device output against reference predictions.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import serial

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_OUTPUT = REPO_ROOT / "evidence" / "parity" / "device_inference_report.json"

EXPECTED_OUTPUT_LOGITS = [-30, -21, 83, -39, -28, -27, -20, -28, -30, -16, -29, -23]
EXPECTED_ARGMAX_INDEX = 2
EXPECTED_PREDICTED_LABEL = "yes"
EXPECTED_CLASSES = [
    "silence", "unknown", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go"
]


def run_hardware_inference(
    port: str = "COM3",
    baud_rate: int = 115200,
    timeout_sec: float = 10.0,
    output_json: Optional[Path] = EVIDENCE_OUTPUT,
) -> Dict[str, Any]:
    print(f"[Code2Edge] Connecting to target MCU on {port} @ {baud_rate} baud...")

    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud_rate,
            timeout=0.5,
            write_timeout=1.0,
        )
    except Exception as e:
        print(f"[Code2Edge] Error: Failed to open {port}: {e}", file=sys.stderr)
        raise

    ser.setDTR(False)
    time.sleep(0.1)
    ser.setDTR(True)
    time.sleep(0.6)

    # Drain any startup banner / leftover bytes
    ser.reset_input_buffer()
    while ser.in_waiting > 0:
        ser.read(ser.in_waiting)
        time.sleep(0.1)

    # Send trigger command 'I' (Inference Report)
    print("[Code2Edge] Sending inference trigger command 'I' to STM32U585...")
    ser.write(b"I\r\n")
    ser.flush()

    raw_lines: List[str] = []
    start_time = time.time()
    inference_json_data: Optional[Dict[str, Any]] = None
    capture_done = False
    in_block = False

    while time.time() - start_time < timeout_sec:
        line_bytes = ser.readline()
        if not line_bytes:
            continue

        line_str = line_bytes.decode("utf-8", errors="replace").strip()
        if not line_str:
            continue

        if line_str == "CODE2EDGE_INFERENCE_START":
            in_block = True
            raw_lines = [line_str]
            print(f"  [MCU] {line_str}")
            continue

        if in_block:
            print(f"  [MCU] {line_str}")
            raw_lines.append(line_str)

            if line_str.startswith("INFERENCE_JSON="):
                try:
                    json_str = line_str[len("INFERENCE_JSON="):]
                    inference_json_data = json.loads(json_str)
                except Exception as e:
                    print(f"[Code2Edge] Warning: Failed to parse INFERENCE_JSON: {e}")

            if "CODE2EDGE_INFERENCE_END" in line_str:
                capture_done = True
                break

    ser.close()

    if not raw_lines:
        raise RuntimeError(f"No response received from MCU on {port} within {timeout_sec}s")

    # Parse and validate result
    report: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_port": port,
        "baud_rate": baud_rate,
        "raw_log": raw_lines,
        "inference_data": inference_json_data,
        "validation": {},
    }

    if inference_json_data:
        actual_logits = inference_json_data.get("logits", [])
        actual_dequant = inference_json_data.get("dequantized", [])
        actual_idx = inference_json_data.get("predicted_index", -1)
        actual_label = inference_json_data.get("predicted_label", "")
        actual_cycles = inference_json_data.get("inference_cycles", 0)
        actual_us = inference_json_data.get("inference_us", 0.0)

        # Check exact integer parity
        logits_match = (actual_logits == EXPECTED_OUTPUT_LOGITS)
        argmax_match = (actual_idx == EXPECTED_ARGMAX_INDEX)
        label_match = (actual_label == EXPECTED_PREDICTED_LABEL)

        report["validation"] = {
            "expected_logits": EXPECTED_OUTPUT_LOGITS,
            "actual_logits": actual_logits,
            "logits_exact_match": logits_match,
            "expected_argmax_index": EXPECTED_ARGMAX_INDEX,
            "actual_argmax_index": actual_idx,
            "argmax_match": argmax_match,
            "expected_label": EXPECTED_PREDICTED_LABEL,
            "actual_label": actual_label,
            "label_match": label_match,
            "inference_cycles": actual_cycles,
            "inference_latency_us": actual_us,
            "status": "PASS" if (logits_match and argmax_match and label_match) else "FAIL",
        }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[Code2Edge] Saved physical inference report to {output_json}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run physical DS-CNN inference on STM32U585.")
    parser.add_argument("--port", default="COM3", help="Serial port (default: COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--timeout", type=float, default=20.0, help="Timeout in seconds")
    parser.add_argument("--output", default=str(EVIDENCE_OUTPUT), help="Output JSON path")
    args = parser.parse_args()

    try:
        report = run_hardware_inference(
            port=args.port,
            baud_rate=args.baud,
            timeout_sec=args.timeout,
            output_json=Path(args.output),
        )
        status = report.get("validation", {}).get("status", "UNKNOWN")
        print(f"\n=======================================================")
        print(f" Physical DS-CNN Inference Verification: [{status}]")
        print(f"=======================================================")
        sys.exit(0 if status == "PASS" else 1)
    except Exception as e:
        print(f"[Code2Edge] Hardware inference failed: {e}", file=sys.stderr)
        sys.exit(1)
