# Code2Edge

[![CI](https://github.com/IndependentSalt69/Code2Edge/actions/workflows/ci.yml/badge.svg)](https://github.com/IndependentSalt69/Code2Edge/actions/workflows/ci.yml)

> **Deployment with proof.**

Code2Edge is a Bob-native edge AI deployment and differential parity verification pipeline. It traces an ML model's complete inference pipeline, generates a constrained-device implementation, and proves numerical correctness through stage-wise differential parity — before and after real hardware deployment.

## Target Hardware

| | |
|---|---|
| **Board** | Arduino UNO Q |
| **MCU Subsystem** | STMicroelectronics STM32U585 (ARM Cortex-M33 @ 160 MHz, 2 MB Flash, 786 KB SRAM) |
| **MPU Subsystem** | Qualcomm Dragonwing QRB2210 (Linux, communicates with STM32U585 via internal UART) |

> **Note:** Code2Edge targets the **STM32U585** on the **Arduino UNO Q**. The Arduino UNO R4 (Renesas RA4M1) is **not** the target.

## Reference Workload

- **Model:** DS-CNN (Depthwise Separable CNN), 119,372 params, fp32 / int8 quantized
- **Task:** Keyword Spotting (12-class: yes, no, up, down, left, right, on, off, stop, go + silence + unknown)
- **Upstream:** [`reference/tiny-kws/`](reference/tiny-kws/) — vendored at commit `c097b35` (MIT License, Priyadeep Jaiswal)

## Repository Layout

```
reference/          Immutable upstream workload + frozen golden tensors
src/
  pipeline/         Generated edge C code (feature_extraction, model_data)
  parity/           Host parity harness C++ (preprocess, parity_runner)
  inference/        Host reference inference runner
  firmware/         STM32U585 Arduino firmware (app.ino, model_runner)
mcp_server/         10-tool MCP server, adapters, mocks
workflow/           Bob workflow state machine (gates, approvals)
tools/
  reference/        Checkpoint fetcher, C code generator
  target/           Hardware benchmark, check, parity, map-parse tools
  host_compat/      Host C stdlib shims for DLL builds
tests/
  pipeline/         Preprocessing parity unit tests
  parity/           Device parity tests
  mcp/              MCP server integration tests
  firmware/         Benchmark harness + bringup sketches
  target/           Target tool unit tests
contracts/
  mcp/              JSON schemas for all 10 MCP tools
  target/           Hardware profile + benchmark result schemas
docs/               Architecture, hardware, audit, and integration docs
evidence/
  parity/           Host & device parity reports
  benchmarks/       Authoritative hardware benchmark results
  model/            Model artifact validation
  runs/             Per-run workflow execution records
bob_sessions/       Bob session logs (member-1, member-2, person-c)
submission/         Final submission bundle
checkpoints/        Local model weights (gitignored)
```

## Key Results

| Metric | Value |
|---|---|
| Host parity gate | 500 samples, 2500/2500 checks PASS (tolerance ≤ 10⁻⁴) |
| Mean inference latency | **5,026.2 ms** (DWT cycles @ 160 MHz, 50-run benchmark) |
| Flash usage | 310.8 KB / 2 MB (39.0% of virtual partition) |
| SRAM usage | 242.2 KB / 786 KB (92.0% of virtual partition) |
| Test suite | 46/46 pytest tests passing |

## Getting Started

```bash
git clone git@github.com:IndependentSalt69/Code2Edge.git
cd Code2Edge
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ --ignore=tests/mcp --ignore=tests/target -q
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full setup, branch conventions, and how to add a new MCP tool.

## Architecture

1. **Two-Tier Differential Parity:**
   - *Tier 1 (Host Gate):* C pipeline vs PyTorch reference across S0–S6; must pass before embedded compilation.
   - *Tier 2 (On-Device Gate):* STM32U585 execution vs frozen golden tensors; 100% classification agreement required.
2. **Authoritative Benchmarking:** Hardware cycles via Cortex-M33 `DWT->CYCCNT`; memory from linker `.map` files.
3. **Zero Dynamic Allocation:** Static tensor arenas and feature buffers only — no `malloc`/`free` in inference loops.

## Attribution

- **tiny-kws** by Priyadeep Jaiswal — [GitHub](https://github.com/priyadeepjaiswal9c/tiny-kws), MIT License.
- **Speech Commands Dataset** — Pete Warden, 2018, CC-BY-4.0.