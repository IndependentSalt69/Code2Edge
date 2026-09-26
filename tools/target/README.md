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
| `run_device_parity.py` | Decodes MCU UART parity dumps and verifies on-device tensor equivalence against golden host reference | `contracts/target/device-parity-result.schema.json` |

## Tier-2 On-Device Differential Parity Comparator (`run_device_parity.py`)

The `run_device_parity.py` tool parses physical UART dumps emitted by the STM32U585 benchmark harness (`tests/firmware/benchmark_harness/benchmark_harness.ino`), decodes all 6,464 IEEE-754 float32 hex words, and validates numerical parity against golden host reference tensors.

### Quick Usage

```bash
# Run against the default saved UART dump (evidence/parity/device_parity_yes_uart.txt):
python tools/target/run_device_parity.py

# Specify custom dump, fixture WAV, and output report paths:
python tools/target/run_device_parity.py \
    --input evidence/parity/device_parity_yes_uart.txt \
    --fixture reference/tiny-kws/app/examples/yes.wav \
    --output evidence/parity/device_parity_report.json \
    --reference-mode pytorch

# Run in quiet mode (suitable for CI / test scripts):
python tools/target/run_device_parity.py --quiet
```

### Tolerances & Validation Gates

Conforming to `docs/validation.md` (Stage S3 Normalized Features):
- **Max Absolute Error:** $\le 1.0 \times 10^{-3}$ ($0.001$)
- **Relative L2 Error:** $\le 5.0 \times 10^{-3}$ ($0.005$)
- **Cosine Similarity:** $\ge 0.999990$
- **Exit Code:** `0` on `PASS`, `1` on `FAIL` / divergence.

## Strict Data Policy

All benchmark scripts writing to `evidence/benchmarks/` must explicitly declare whether the report is `MEASURED` (from physical silicon execution) or `ESTIMATED` (from static analysis).
