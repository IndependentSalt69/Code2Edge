#!/usr/bin/env python3
"""
Inspect and report the hardware readiness and target profile of the STM32U585 MCU
subsystem on the Arduino UNO Q. Emits JSON conforming to contracts/target/target-profile.schema.json.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_FILE = REPO_ROOT / "contracts" / "target" / "target-profile.json"


def find_arduino_cli() -> str | None:
    # 1. Check PATH
    cli = shutil.which("arduino-cli")
    if cli:
        return cli

    # 2. Check standard Windows Arduino IDE backend location
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        ide_cli = Path(local_app_data) / "Programs" / "Arduino IDE" / "resources" / "app" / "lib" / "backend" / "resources" / "arduino-cli.exe"
        if ide_cli.exists():
            return str(ide_cli)

    return None


def get_installed_cores(arduino_cli_path: str) -> list[str]:
    try:
        res = subprocess.run([arduino_cli_path, "core", "list", "--format", "json"], capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        if isinstance(data, list):
            return [c.get("id", "") for c in data]
    except Exception:
        pass
    return []


def check_target() -> dict:
    cli_path = find_arduino_cli()
    toolchain_status = "NOT_VERIFIED"
    toolchain_version = "Unknown"
    core_installed = False

    if cli_path:
        try:
            res = subprocess.run([cli_path, "version", "--format", "json"], capture_output=True, text=True, check=True)
            v_data = json.loads(res.stdout)
            toolchain_version = v_data.get("VersionString", "1.4.1")
            installed_cores = get_installed_cores(cli_path)
            if "arduino:zephyr" in installed_cores:
                core_installed = True
                toolchain_status = "VERIFIED"
            else:
                toolchain_status = "NOT_VERIFIED"
        except Exception:
            toolchain_status = "UNKNOWN"

    profile = {
        "$schema": "./target-profile.schema.json",
        "target_id": "arduino_uno_q_stm32u585",
        "board_name": "Arduino UNO Q",
        "mcu": {
            "core": "ARM Cortex-M33",
            "part_number": "STM32U585",
            "clock_hz": 160000000,
            "fpu": "Single-precision FP32",
            "dsp_extensions": True,
            "flash_bytes": 2097152,
            "sram_bytes": 804864
        },
        "memory_limits": {
            "flash_limit_bytes": 2097152,
            "sram_limit_bytes": 804864,
            "max_tensor_arena_bytes": 524288,
            "max_feature_buffer_bytes": 65536
        },
        "toolchain": {
            "name": "arduino-cli",
            "cli_path": cli_path,
            "version": toolchain_version,
            "core_platform": "arduino:zephyr (1.0.0)",
            "compiler": "arm-zephyr-eabi-gcc",
            "core_installed": core_installed,
            "status": toolchain_status
        },
        "bridge": {
            "transport": "UART",
            "device_path": "/dev/ttyMSM0",
            "baud_rate": 115200,
            "packet_format": "binary_8byte_framed",
            "status": "NOT_VERIFIED"
        },
        "supported_runtimes": [
            "tflite_micro",
            "cmsis_nn_baremetal"
        ],
        "supported_operators": [
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "AVERAGE_POOL_2D",
            "FULLY_CONNECTED",
            "RELU",
            "SOFTMAX"
        ],
        "restrictions": [
            "Zero dynamic memory allocation in inference loop",
            "Static tensor arena and feature buffers only",
            "All weights resident in Flash (.rodata)",
            "Target is Arduino UNO Q (STM32U585), NOT Arduino UNO R4"
        ]
    }
    return profile


def main():
    profile = check_target()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print("================================================================")
    print(" Code2Edge Target Hardware Profile: Arduino UNO Q (STM32U585)")
    print("================================================================")
    print(f"Target ID:        {profile['target_id']}")
    print(f"MCU:              {profile['mcu']['part_number']} ({profile['mcu']['core']} @ {profile['mcu']['clock_hz'] // 1000000} MHz)")
    print(f"Flash Budget:     {profile['memory_limits']['flash_limit_bytes'] // 1024} KB")
    print(f"SRAM Budget:      {profile['memory_limits']['sram_limit_bytes'] // 1024} KB")
    print(f"CLI Detected:     {profile['toolchain']['name']} ({profile['toolchain']['version']})")
    print(f"Zephyr Core:      {'INSTALLED' if profile['toolchain']['core_installed'] else 'NOT INSTALLED (Available in index)'}")
    print(f"Toolchain Status: {profile['toolchain']['status']}")
    print(f"Bridge Status:    {profile['bridge']['status']}")
    print(f"\nWrote profile to: {OUTPUT_FILE}")
    print("================================================================")


if __name__ == "__main__":
    main()
