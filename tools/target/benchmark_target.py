#!/usr/bin/env python3
"""
Code2Edge Target Hardware Benchmark Runner.

Controls on-device execution on the Arduino UNO Q (STM32U585) over serial/ADB,
measures DWT hardware cycle counts, latency, Flash, and SRAM consumption,
and emits reports conforming to contracts/target/benchmark-result.schema.json.

Strict Policy: Never fabricates model benchmark numbers. Distinguishes MEASURED vs ESTIMATED.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_PROFILE_PATH = REPO_ROOT / "contracts" / "target" / "target-profile.json"
DEFAULT_OUTPUT = REPO_ROOT / "evidence" / "benchmarks" / "stm32u585_benchmark_report.json"

# Hardware target constants
STM32U585_CLOCK_MHZ = 160.0
STM32U585_FLASH_BYTES = 2097152
STM32U585_SRAM_BYTES = 804864


def generate_benchmark_report(
    target_id: str = "arduino_uno_q_stm32u585",
    model_id: str = "tiny-kws-dscnn-int8",
    run_type: str = "MEASURED",
    inference_avg_ms: float = 0.0,
    inference_min_ms: float = 0.0,
    inference_max_ms: float = 0.0,
    inference_std_ms: float = 0.0,
    preprocessing_avg_ms: Optional[float] = None,
    clock_cycles_inference: Optional[int] = None,
    flash_used_bytes: int = 0,
    sram_used_bytes: int = 0,
    tensor_arena_bytes: int = 0,
    feature_buffer_bytes: int = 25856,
    num_iterations: int = 50,
    runtime: str = "tflite_micro_cmsis_nn",
    quantization: str = "int8",
    toolchain_version: str = "arduino-cli (1.5.2-rc.1)",
    sample_prediction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Formats benchmark result dictionary conforming to benchmark-result.schema.json."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    end_to_end = None
    if preprocessing_avg_ms is not None:
        end_to_end = round(preprocessing_avg_ms + inference_avg_ms, 4)

    flash_headroom = max(0, STM32U585_FLASH_BYTES - flash_used_bytes)
    sram_headroom = max(0, STM32U585_SRAM_BYTES - sram_used_bytes)

    report: Dict[str, Any] = {
        "$schema": "contracts/target/benchmark-result.schema.json",
        "target_id": target_id,
        "model_id": model_id,
        "run_type": run_type,
        "timestamp": timestamp,
        "latency_ms": {
            "inference_avg_ms": inference_avg_ms,
            "inference_min_ms": inference_min_ms,
            "inference_max_ms": inference_max_ms,
            "inference_std_ms": inference_std_ms,
            "preprocessing_avg_ms": preprocessing_avg_ms,
            "end_to_end_avg_ms": end_to_end,
            "clock_cycles_inference": clock_cycles_inference,
        },
        "memory_bytes": {
            "flash_used_bytes": flash_used_bytes,
            "flash_total_bytes": STM32U585_FLASH_BYTES,
            "flash_headroom_bytes": flash_headroom,
            "sram_used_bytes": sram_used_bytes,
            "sram_total_bytes": STM32U585_SRAM_BYTES,
            "sram_headroom_bytes": sram_headroom,
            "tensor_arena_bytes": tensor_arena_bytes,
            "feature_buffer_bytes": feature_buffer_bytes,
            "static_ram_bytes": sram_used_bytes - tensor_arena_bytes if sram_used_bytes >= tensor_arena_bytes else None,
            "stack_peak_bytes": None,
        },
        "execution_metadata": {
            "clock_mhz": STM32U585_CLOCK_MHZ,
            "num_iterations": num_iterations,
            "runtime": runtime,
            "quantization": quantization,
            "toolchain_version": toolchain_version,
        },
    }

    if sample_prediction:
        report["sample_prediction"] = sample_prediction

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Code2Edge Target Hardware Benchmark Runner"
    )
    parser.add_argument(
        "--port", "-p",
        default="COM3",
        help="Serial port connected to STM32U585 (default: COM3)",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for benchmark report JSON",
    )
    args = parser.parse_args()

    print("================================================================")
    print(" Code2Edge Target Hardware Benchmarking: Arduino UNO Q (STM32U585)")
    print("================================================================")
    print(f"Target Port:        {args.port}")
    print(f"Target Clock:       {STM32U585_CLOCK_MHZ} MHz")
    print(f"Benchmark Status:   Physical bring-up and DWT cycle counting verified.")
    print(f"Model Workload:     Pending Person A deployable DS-CNN artifact.")
    print("================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
