# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Status

**Code2Edge is in active development.**
- **Reference Workload:** Vendored in `reference/tiny-kws/` (pinned commit `c097b35ae4b9cd585a544c74db16892ce674b186` of [priyadeepjaiswal9c/tiny-kws](https://github.com/priyadeepjaiswal9c/tiny-kws)).
- **Core Stack:** Python 3.10+ (PyTorch, torchaudio, numpy) for host inference and parity verification; C (ARM CMSIS-DSP, CMSIS-NN) for target edge code generation.
- **Target MCU:** STM32U585 (ARM Cortex-M33) on Arduino UNO R4 / UNO Q.

## Directory Layout

| Path | Purpose |
|------|---------|
| `reference/tiny-kws/` | **IMMUTABLE** vendored reference workload (PyTorch DS-CNN, LogMel, evaluation) |
| `src/inference/` | Host reference inference runner & non-invasive hook capture |
| `src/parity/` | Differential parity harness (stages S0–S6), tolerance checks, and reporting |
| `src/export/` | Edge code generation (C / CMSIS-DSP / CMSIS-NN) |
| `tests/parity/` | Parity gate test suite and intermediate stage validation tests |
| `tools/` | Developer tooling (integrity check, checkpoint fetcher) |
| `deploy/` | MCU deployment & firmware integration |
| `docs/` | Architecture, problem statement, feasibility, and integration plan |
| `checkpoints/` | Local model weights (gitignored except `.gitkeep`; canonical `best.pt`) |
| `data/` | Local dataset mount point (gitignored) |

## Stack & Environment

- **Host Python:** PyTorch >= 2.6, torchaudio >= 2.6, numpy, soundfile, scikit-learn.
- **Edge Runtime:** C99 / C11, ARM CMSIS-DSP, ARM CMSIS-NN.

## Key Gotchas & Rules

- **DO NOT MODIFY `reference/tiny-kws/`:** This directory is an immutable vendored snapshot. Any overrides, wrappers, or extensions must be authored in `src/`.
- **Checkpoint Handling:** Pretrained weights (`checkpoints/best.pt`) are downloaded from Hugging Face Hub (`priyadeepjaiswal9c/tiny-kws`) and must not be checked into git.
- **Dataset Root:** Audio datasets are referenced via the `SPEECH_COMMANDS_DATA_ROOT` environment variable and must never be committed to git.
- **Parity Tolerance:** Any edge implementation must satisfy the stage-wise tolerance table defined in `docs/INTEGRATION_PLAN.md` before deployment.
