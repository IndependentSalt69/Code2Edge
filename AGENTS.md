# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Mission & Identity

**Code2Edge** — *Deployment with proof.*  
A Bob-native workflow that traces an ML repository's complete inference pipeline, generates a constrained-device implementation, and proves correctness through stage-wise differential parity before and after real hardware deployment.

- **Primary Workload:** Keyword Spotting using the frozen `tiny-kws` DS-CNN (119k params).
- **Target Hardware:** **Arduino UNO Q — STM32U585 MCU Subsystem** (ARM Cortex-M33 @ 160 MHz, 2 MB Flash, 786 KB SRAM).
- **Hardware Clarification Rule:** Code2Edge targets the **STM32U585** on the **Arduino UNO Q**. It does **NOT** target the Arduino UNO R4 (Renesas RA4M1). Any references to UNO R4 have been corrected.

## Team Tracks & Responsibilities

- **Person A (Host ML & Edge Code Generation):** Owns reference ingestion, host parity harness (`src/inference/`, `src/parity/`), and C code generation for feature extraction and neural network (`src/pipeline/`).
- **Person B (Embedded Firmware & Hardware Benchmarking):** Owns STM32U585 firmware integration (`src/firmware/`), hardware toolchain/build (`tools/target/`), dummy-model proof, on-device parity verification, and authoritative latency/memory benchmarking (`contracts/target/`).

## Directory Layout

```text
Code2Edge/
├── reference/
│   ├── tiny-kws/              # IMMUTABLE upstream reference workload (c097b35)
│   ├── corpus/                # Small/fixed test corpus metadata
│   └── golden/                # Frozen reference intermediate fixtures (S0–S6)
├── src/
│   ├── pipeline/              # Generated edge pipeline implementation (C/C++)
│   ├── inference/             # Host reference inference runner & non-invasive hooks
│   ├── parity/                # Host differential parity comparison engine
│   └── firmware/              # STM32U585 firmware integration & Arduino sketch
├── mcp_server/                # Code2Edge MCP server tools
├── tools/
│   ├── reference/             # Checkpoint fetcher and reference integrity checker
│   └── target/                # Hardware build, flash, bridge test, and benchmark tools
├── tests/
│   ├── pipeline/              # Unit tests for generated C pipeline
│   ├── parity/                # Parity gate test suite
│   ├── mcp/                   # MCP server contract tests
│   ├── firmware/              # Firmware unit and memory tests
│   └── integration/           # End-to-end integration tests
├── contracts/
│   ├── mcp/                   # MCP tool interfaces
│   ├── parity/                # Parity schemas
│   └── target/                # target-profile, benchmark-result, device-parity schemas
├── docs/                      # Architecture, problem, feasibility, hardware, plan docs
├── evidence/
│   ├── parity/                # Host & device parity verification evidence
│   ├── benchmarks/            # Authoritative benchmark tables (predicted vs measured)
│   └── screenshots/           # Hardware and UI capture evidence
├── submission/                # Final project submission bundle
├── bob_sessions/              # Bob session tracking (member-1, member-2)
└── checkpoints/               # Local model weights (gitignored except .gitkeep)
```

## Stack & Toolchain Decisions

- **Host Python:** Python 3.10+ (PyTorch >= 2.6, torchaudio >= 2.6, numpy, soundfile, scikit-learn).
- **Edge MCU Toolchain:** `arduino-cli` with `arduino:zephyr` core (version 1.0.0, board FQBN `arduino:zephyr:unoq`), utilizing `arm-zephyr-eabi-gcc`.
- **Embedded Runtime:** TensorFlow Lite for Microcontrollers (TFLM) with CMSIS-NN kernel acceleration on ARM Cortex-M33.
- **MPU ↔ MCU Bridge:** Internal UART on `/dev/tty*` with 8-byte framed binary packet protocol. Fallback: offline golden fixtures in Flash.

## Key Gotchas & Architectural Rules

1. **DO NOT MODIFY `reference/tiny-kws/`:** This directory is an immutable vendored snapshot. Any overrides, wrappers, or extensions must be authored in `src/`.
2. **Strict Parity Gate Sequence:**
   - **Tier 1 (Host Parity Gate):** Must pass all stage tolerances (S0–S6) before embedded compilation is permitted. Person B must NEVER bypass this gate.
   - **Tier 2 (On-Device Parity Gate):** Validates execution on the physical STM32U585 against golden host tensors.
3. **No Fabricated Benchmarks:** Never present estimated latency or memory as measured results. All reports must explicitly distinguish `ESTIMATED` vs `MEASURED`.
4. **Zero Dynamic Allocation on MCU:** `malloc()` and `free()` are forbidden in inference loops. Tensor arena and feature buffers must be allocated statically in `.bss`.
5. **Checkpoint & Dataset Safety:** Pretrained weights (`checkpoints/best.pt`) and local audio datasets (`SPEECH_COMMANDS_DATA_ROOT`) must never be committed to git.
