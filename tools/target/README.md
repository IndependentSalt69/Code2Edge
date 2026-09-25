# Target Tooling & Benchmarking (`tools/target/`)

This directory houses the host-side tools for target profiling, MPU ↔ MCU communication, flashing, and authoritative benchmarking on the **Arduino UNO Q (STM32U585)**.

## Tools Overview

| Script | Purpose | Output Contract |
|---|---|---|
| `check_target.py` | Detects hardware availability, toolchain versions, memory limits, and platform profile | `contracts/target/target-profile.schema.json` |
| `bridge_test.py` | Sends test arrays over UART to verify MPU ↔ MCU serial reliability | Console log & loopback verification |
| `build_firmware.py` | Invokes `arduino-cli` with `arduino:zephyr:unoq` to build firmware binary | ELF/BIN build output & linker map |
| `parse_map.py` | Parses GCC linker `.map` file to extract exact Flash and SRAM bytes | Memory breakdown dict |
| `benchmark_target.py` | Measures hardware inference latency, Flash, SRAM, and arena footprint | `contracts/target/benchmark-result.schema.json` |
| `run_device_parity.py` | Executes test corpus on MCU and verifies predictions against Python reference | `contracts/target/device-parity-result.schema.json` |

## Strict Data Policy

All benchmark scripts writing to `evidence/benchmarks/` must explicitly declare whether the report is `MEASURED` (from physical silicon execution) or `ESTIMATED` (from static analysis).
