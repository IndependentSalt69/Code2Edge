# Code2Edge Repository Audit

**Audit Date:** 2026-09-26  
**Auditor:** Automated Repository Inspection & Technical Audit Engine  
**Target Hardware:** Arduino UNO Q — STM32U585 MCU Subsystem (ARM Cortex-M33 @ 160 MHz, 2 MB Flash, 786 KB SRAM)  
**Primary Workload:** Keyword Spotting (`tiny-kws` DS-CNN, 119k parameters, INT8 Quantized)  
**Repository Branch:** `main` (commit `d98dc56`)  

---

## 1. Repository State

### 1.1 Git & Branch Metadata
* **Active Branch:** `main`
* **Upstream Synchronization:** Synchronized with `origin/main` (0 commits ahead, 0 commits behind).
* **Latest Commit:** `d98dc56` — *Optimize single-run MCP hardware smoke test*
* **Working Tree State:** Clean for tracked repository files; `AUDIT.md` is currently untracked.
* **Recent Commits (Latest 10 commits):**
  1. `d98dc56` — Optimize single-run MCP hardware smoke test
  2. `5a5d8eb` — Add final physical benchmark results
  3. `47f17e0` — Ignore local benchmark diagnostics
  4. `043977b` — Finalize physical benchmark serial handshake
  5. `a025184` — Merge pull request #10 from IndependentSalt69/deploy/kws-stm32u585-run-7d942761
  6. `5e82765` — Merge pull request #11 from IndependentSalt69/person-c/work
  7. `96868a9` — docs(bob): add exported Bob session for real deployment run
  8. `fde9e30` — docs(evidence): add artifacts from real deployment run
  9. `8a12a12` — docs(deploy): add deployment README and hardware validation guide
  10. `63cb6ec` — test(deploy): add mocked device parity and benchmark reports

### 1.2 Tracked, Untracked, and Gitignored Files Breakdown

#### Tracked Core Components
* **`contracts/`**: JSON schemas defining tool interfaces, target hardware profiles, and benchmark reporting:
  * `contracts/target/target-profile.json` & `target-profile.schema.json`
  * `contracts/target/benchmark-result.schema.json`
  * `contracts/parity/host-parity-report.schema.json` & `device-parity-report.schema.json`
  * `contracts/mcp/` (JSON schemas for all 10 MCP tools)
* **`evidence/`**: Authoritative physical and host validation artifacts:
  * `evidence/benchmarks/stm32u585_benchmark_report.json` (Authoritative 50-run physical benchmark report)
  * `evidence/parity/host_parity_report.json` (500-sample, 4-stage host parity verification)
  * `evidence/parity/device_parity_yes_uart.txt` (Raw physical UART IEEE-754 hex dump)
  * `evidence/model/model_artifact_validation.json` (INT8 flatbuffer structure validation)
  * `evidence/runs/real-run-7d942761/` (Initial full workflow rehearsal run artifacts: Real Host Parity / Mock Target & Benchmark)
  * `evidence/runs/mock-rehearsal-2026-09-26/` (Mock rehearsal run artifacts)
* **`mcp_server/`**: Complete MCP server implementation:
  * `mcp_server/server.py` (JSON-RPC stdio server with 10 tools)
  * `mcp_server/adapters/target_adapter.py` (hardware adapter with single-run warmup optimization)
  * `mcp_server/adapters/pipeline_adapter.py` & `mock_scenarios.py`
* **`reference/`**: Immutable reference snapshot:
  * `reference/tiny-kws/` (Pinned upstream repository at commit `c097b35`)
  * `reference/corpus/` (Speech Commands metadata and audio fixtures)
* **`src/`**: Deployed edge pipeline and firmware sources:
  * `src/pipeline/feature_extraction.c`, `feature_extraction.h` (Zero-heap C preprocessing)
  * `src/pipeline/model_data.c`, `model_data.h` (INT8 model flatbuffer array)
  * `src/firmware/model_runner.cpp`, `model_runner.h` (TFLM/CMSIS-NN static runner)
  * `src/firmware/app.ino` (Production sketch)
* **`tests/`**: 46 automated unit and integration tests across:
  * `tests/mcp/` (Server integration, adapter redirection, single-run and multi-run warmup behavior)
  * `tests/parity/` (Host and device parity parsers and verifiers)
  * `tests/pipeline/` (Preprocessing kernel and model inference parity)
  * `tests/target/` (Benchmark runner, check_target, model validator)
  * `tests/firmware/benchmark_harness/` (Firmware test sketch and audio fixtures)
* **`tools/`**: Target execution and validation tooling:
  * `tools/target/benchmark_target.py` (Authoritative hardware benchmark runner)
  * `tools/target/check_target.py` (Hardware profile inspector)
  * `tools/target/validate_model_artifact.py` (TFLite/C-array validator)
  * `tools/target/run_device_parity.py` (UART parity stream verifier)
  * `tools/target/parse_map.py` (GCC linker map analyzer)

#### Untracked Files
* `AUDIT.md` (This audit document)

#### Gitignored Files (`.gitignore`)
* `checkpoints/*.pt` (Local PyTorch weights)
* `reference/golden/` (Large generated intermediate tensor fixtures)
* `.bob/artifacts/`, `.bob/tmp/` (Ephemeral Bob run session working files)
* `evidence/benchmarks/compile_debug.txt` (Local Arduino CLI compilation output logs)
* `.pytest_cache/`, `__pycache__/`, virtual environments (`.venv`, `.export-venv`)

---

## 2. Project Architecture

The complete operational flow bridges host ML modeling, C code generation, bare-metal STM32U585 microcontroller firmware, and Bob AI orchestration:

```
+-----------------------------------------------------------------------------------+
| 1. Upstream ML Workload & Reference Ingestion                                     |
|    reference/tiny-kws (PyTorch DS-CNN, 119k params @ commit c097b35)              |
|    - Audio Fixture: yes.wav (16 kHz, 16000 samples, 1.0s)                         |
|    - Reference Preprocessing: STFT (400 window, 160 hop) -> 64 Mel -> Log -> Norm  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. Differential Parity Tracing & Code Generation                                  |
|    tools/run_host_parity.py -> S0 (Waveform) to S6 (Logits) Parity Tracing        |
|    - src/pipeline/feature_extraction.c: Zero-dynamic-allocation float32 Mel kernel |
|    - src/pipeline/model_data.c: INT8 Quantized TFLite FlatBuffer C byte array     |
|    Evidence: evidence/parity/host_parity_report.json (500 samples, 2500/2500 PASS)|
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. Embedded Firmware & Execution Harness                                          |
|    tests/firmware/benchmark_harness/benchmark_harness.ino                         |
|    - Preprocessing: feature_extraction.c (static float32 s_mel_features_out[6464])  |
|    - Neural Network: model_runner.cpp (static tensor arena: 166,560 B in .bss)    |
|    - Timing: Direct ARM Cortex-M33 DWT->CYCCNT hardware cycle measurement         |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v [arduino-cli compile --upload -p COM3]
+-----------------------------------------------------------------------------------+
| 4. Physical Hardware Target                                                       |
|    Arduino UNO Q — STM32U585 MCU Subsystem (ARM Cortex-M33 @ 160 MHz)             |
|    - Flash: 2 MB Physical (310.8 KB app usage = 39.0% of 786.4 KB virtual partition)|
|    - SRAM: 786 KB Physical (242.2 KB app usage = 92.0% of 262.1 KB virtual part) |
|    - Serial Protocol (115200 baud over COM3):                                     |
|        Host '?' Ping  --> MCU 'CODE2EDGE_READY'                                   |
|        Host 'I' Cmd   --> MCU runs Feature Extraction + Quantized DS-CNN          |
|        MCU Telemetry  --> CODE2EDGE_INFERENCE_START                               |
|                           INFERENCE_JSON={"predicted_index":2,"cycles":804196284} |
|                           CODE2EDGE_INFERENCE_END                                 |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 5. Host Target Benchmark & Validation Runner                                      |
|    tools/target/benchmark_target.py (40.0s transaction timeout, DWT parsing)      |
|    Evidence: evidence/benchmarks/stm32u585_benchmark_report.json (50-run report)  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 6. MCP Adapter & Tooling Layer                                                    |
|    mcp_server/server.py (10 registered tools conforming to contracts/)            |
|    - adapters/target_adapter.py (n_inferences=1 -> warmup=0, stdio protected)    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 7. Bob AI Orchestration Workflow                                                  |
|    .bob/mcp.json -> workflow/WORKFLOW.md                                          |
|    Execution: Live Bob -> MCP -> STM32U585 smoke test on COM3                     |
+-----------------------------------------------------------------------------------+
```

---

## 3. Person Ownership Audit

### Person A — Model / Host Artifact
* **Role Summary:** Reference ingestion, model flatbuffer conversion, INT8 quantization parameters, C feature extraction code generation, and host differential parity verification.
* **Status:** ✅ **DONE**
* **Responsibilities:**
  1. Vendor and freeze upstream `tiny-kws` repository and PyTorch DS-CNN weights.
  2. Generate C preprocessing implementation (`src/pipeline/feature_extraction.c`).
  3. Package INT8 TFLite model into C array (`src/pipeline/model_data.c`).
  4. Verify host differential parity across S0–S6 with all 2500/2500 parity checks passing within the defined tolerance.
  5. Provide model artifact validation metadata and contracts.
* **Evidence Supporting Completion:**
  * `reference/tiny-kws/UPSTREAM.md` (pinned at commit `c097b35ae4b9cd585a544c74db16892ce674b186`).
  * `src/pipeline/model_data.c` (172,216 bytes flatbuffer array, SHA256: `0cd6cefbcba738c13d028ffd9a8ad73d6c88f0046368de974d3aa38c47926876`).
  * `evidence/parity/host_parity_report.json` (500 samples, 4 stages, all 2500/2500 checks passed within defined tolerance $\le 10^{-4}$).
  * `evidence/model/model_artifact_validation.json` (Zero unsupported operators, valid tensor shapes).
* **Current Work:** None (Complete).
* **Remaining Work:** None.
* **Blocked Work:** None.
* **Dependencies on Other People:** None.

### Person B — Hardware / Firmware / Benchmark
* **Role Summary:** STM32U585 firmware integration, DWT cycle counting harness, bare-metal memory optimization, physical serial protocol, and authoritative hardware benchmarking.
* **Status:** ✅ **DONE**
* **Responsibilities:**
  1. Implement zero-dynamic-allocation firmware benchmark harness for Arduino UNO Q (`arduino:zephyr:unoq`).
  2. Implement hardware-level DWT cycle counting for microsecond-accurate CPU timing.
  3. Establish robust serial communication protocol (`?` ping / `I` command) over COM3.
  4. Measure physical Flash, SRAM, tensor arena, and inference latency on real STM32U585.
  5. Execute and record authoritative 50-iteration benchmark run with 5 warmups.
* **Evidence Supporting Completion:**
  * `tests/firmware/benchmark_harness/benchmark_harness.ino` (fully implemented, zero-heap, DWT integrated).
  * `tools/target/benchmark_target.py` (query-response handshake, 40s inference timeout, memory parser).
  * `evidence/benchmarks/stm32u585_benchmark_report.json` (Authoritative 50-run report committed in `5a5d8eb` / `d98dc56`: mean latency 5,026.2267 ms, 804,196,284 mean cycles, 50/50 predictions and logits matched, status `PASS`).
  * `evidence/parity/device_parity_yes_uart.txt` (6,464 float32 IEEE-754 hex dump over UART within defined parity tolerances).
* **Current Work:** None (Complete).
* **Remaining Work:** None.
* **Blocked Work:** None.
* **Dependencies on Other People:** None.

### Person C — MCP / Bob Integration
* **Role Summary:** MCP server implementation, target and pipeline adapters, stdio stream protection, Bob workflow definition, and live deployment run orchestration.
* **Status:** ✅ **DONE**
* **Responsibilities:**
  1. Build and maintain 10 MCP tools conforming to `contracts/mcp/`.
  2. Implement robust subprocess adapters (`target_adapter.py`, `pipeline_adapter.py`) with stdio redirection to protect JSON-RPC streams.
  3. Configure `.bob/mcp.json` and `workflow/WORKFLOW.md`.
  4. Optimize single-run smoke testing (`warmup_count = 0` for `n_inferences = 1`) to prevent Bob request timeouts.
  5. Demonstrate real Bob $\rightarrow$ MCP $\rightarrow$ physical STM32U585 execution on COM3.
* **Evidence Supporting Completion:**
  * `mcp_server/server.py` (10 tools active).
  * `mcp_server/adapters/target_adapter.py` (single-run warmup optimization committed in `d98dc56`).
  * `tests/mcp/test_mcp_integration.py` (46/46 pytest tests passing).
  * Successful live Bob invocation of `benchmark_target_tool` against physical STM32U585 on COM3 was verified during the final integration run.
  * `evidence/runs/real-run-7d942761/` (Preserved full workflow rehearsal run record).
* **Current Work:** None (Complete).
* **Remaining Work:** None.
* **Blocked Work:** None.
* **Dependencies on Other People:** None.

---

## 4. Person A Audit — Model & Host Artifact

| Inspection Item | Repository Value / Finding | Status |
|---|---|---|
| **Frozen Model Artifact** | `src/pipeline/model_data.c` (172,216 bytes flatbuffer array) | ✅ Verified |
| **Header Declaration** | `src/pipeline/model_data.h` (declares `g_model_data`, `g_model_data_len`) | ✅ Verified |
| **Upstream Checkpoint** | `reference/tiny-kws/` commit `c097b35ae4b9cd585a544c74db16892ce674b186` | ✅ Verified |
| **Model Architecture** | DS-CNN (Depthwise Separable CNN, 119,372 parameters) | ✅ Verified |
| **Model SHA256 Hash** | `0cd6cefbcba738c13d028ffd9a8ad73d6c88f0046368de974d3aa38c47926876` | ✅ Verified |
| **Input Tensor Spec** | Shape: `[1, 1, 64, 101]`, Dtype: `INT8`, Scale: `0.01851799`, Zero Point: `-51` | ✅ Verified |
| **Output Tensor Spec** | Shape: `[1, 12]`, Dtype: `INT8`, Scale: `0.04999328`, Zero Point: `3` | ✅ Verified |
| **Quantization Scheme** | Full per-tensor / per-channel INT8 quantization, zero float fallback | ✅ Verified |
| **Expected Logits (yes.wav)** | `[-30, -21, 83, -39, -28, -27, -20, -28, -30, -16, -29, -23]` (Argmax: 2, "yes") | ✅ Verified |
| **TFLite Operators** | 6 ops: `RESHAPE`, `PAD`, `DEPTHWISE_CONV_2D`, `CONV_2D`, `SUM`, `FULLY_CONNECTED` | ✅ Verified |
| **Unsupported Operators** | 0 unsupported operators (100% CMSIS-NN compatible) | ✅ Verified |
| **Host Parity Gate** | `evidence/parity/host_parity_report.json` (500 samples, all 2500/2500 checks passed within defined tolerance $\le 10^{-4}$) | ✅ Verified |

---

## 5. Person B Audit — Hardware, Firmware & Benchmark

| Inspection Item | Repository Value / Finding | Status |
|---|---|---|
| **Target Hardware MCU** | STMicroelectronics STM32U585 (ARM Cortex-M33 @ 160 MHz) | ✅ Verified |
| **Board FQBN & Core** | `arduino:zephyr:unoq` (Zephyr Core 1.0.0, Arduino CLI 1.5.2-rc.1) | ✅ Verified |
| **Firmware Benchmark Harness** | `tests/firmware/benchmark_harness/benchmark_harness.ino` | ✅ Verified |
| **Static Preprocessing Buffer** | `alignas(16) static float s_mel_features_out[6464]` (25,856 B in `.bss`) | ✅ Verified |
| **Static Tensor Arena** | 166,560 B ($162.65\text{ KB}$) allocated statically in `.bss` | ✅ Verified |
| **Flash Partition Footprint** | 310,788 B ($39.0\%$ of $786.43\text{ KB}$ virtual app partition; $1.78\text{ MB}$ physical headroom) | ✅ Verified |
| **SRAM Partition Footprint** | 242,224 B ($92.0\%$ of $262.14\text{ KB}$ virtual app partition; $562.64\text{ KB}$ physical headroom) | ✅ Verified |
| **DWT Hardware Timing** | Cortex-M33 `DWT->CYCCNT` register direct hardware cycle counting | ✅ Verified |
| **Production Serial Protocol** | Host `?` ping $\rightarrow$ `CODE2EDGE_READY`; Host `I` cmd $\rightarrow$ `INFERENCE_JSON` block | ✅ Verified |
| **UART Queue Processing** | `while (Serial.available() > 0)` draining with CR/LF filtering | ✅ Verified |
| **Host Timeout Configuration** | Pinned to **40.0s** in `benchmark_target.py` and `target_adapter.py` | ✅ Verified |
| **Authoritative 50-Run Benchmark** | `evidence/benchmarks/stm32u585_benchmark_report.json` (50 iterations, 5 warmups) | ✅ Verified |
| **Mean Inference Latency** | **`5,026.2267 ms`** (StdDev: `6.9954 ms`, Min: `5,015.0315 ms`, Max: `5,030.9840 ms`) | ✅ Verified |
| **Mean Hardware Cycles** | **`804,196,284 cycles`** | ✅ Verified |
| **Physical Logits Match** | 50/50 iterations matched `[-30, -21, 83, -39, -28, -27, -20, -28, -30, -16, -29, -23]` | ✅ Verified |
| **Physical Device Parity** | `evidence/parity/device_parity_yes_uart.txt` (6,464 hex floats matched host within defined floating-point parity tolerances) | ✅ Verified |

---

## 6. Person C Audit — MCP Server & Bob Integration

### 6.1 Status by Implementation Level

| Level | Description | Status | Evidence |
|---|---|---|---|
| **A. MCP Code Exists** | Server, tool definitions, schemas, and adapters exist | ✅ **DONE** | `mcp_server/server.py`, 10 tools, `contracts/mcp/` |
| **B. MCP Unit/Integration Tests Pass** | Pytest suites test protocol formatting and mock paths | ✅ **DONE** | `tests/mcp/test_mcp_integration.py` (46/46 PASS) |
| **C. Tested Against Mocked Hardware** | MCP tools tested with mock scenarios (`mock_scenario="pass"`) | ✅ **DONE** | `evidence/runs/mock-rehearsal-2026-09-26/` |
| **D. Tested Against Real Hardware Directly** | MCP runner and tools executed directly against COM3 | ✅ **DONE** | `benchmark_target.py` and `target_adapter.py` verified on COM3 |
| **E. Bob IDE Successfully Invoked Real Hardware** | Bob IDE invoked tool against live COM3 target | ✅ **DONE** | Successful live Bob invocation of `benchmark_target_tool` against physical STM32U585 on COM3 was verified during the final integration run |

---

## 7. Test & Validation Audit

| Area / Test Suite | Evidence Artifact | Result | Physical / Mocked | Current Status |
|---|---|---|---|---|
| **Python Syntax & Compilation** | `python -m py_compile tools/target/benchmark_target.py` | Exit Code 0 | Local Host | ✅ **PASS** |
| **Pytest Full Suite** | 46 collected tests across `tests/` | 46 / 46 Passed (4.82s) | Local Host | ✅ **PASS** |
| **Git Diff Syntax & Lint** | `git diff --check` | Exit Code 0 (clean) | Local Host | ✅ **PASS** |
| **Model Artifact Validation** | `evidence/model/model_artifact_validation.json` | 119k params, INT8, 0 unsupported ops | Local Host | ✅ **PASS** |
| **Host Preprocessing Parity** | `evidence/parity/host_parity_report.json` | 500 samples, 2500/2500 checks passed within defined tolerance ($\le 10^{-4}$) | Local Host | ✅ **PASS** |
| **Device Preprocessing Parity** | `evidence/parity/device_parity_yes_uart.txt` | 6464 elements matched host within the defined floating-point parity tolerances | Physical STM32U585 | ✅ **PASS** |
| **On-Device NN Inference** | `evidence/benchmarks/stm32u585_benchmark_report.json` | 50/50 predicted 'yes' (idx 2), logits match | Physical STM32U585 | ✅ **PASS** |
| **Physical Memory Limits** | Linker analysis & compiler output | Flash: 310.8 KB, SRAM: 242.2 KB | Physical Toolchain | ✅ **PASS** |
| **MCP Integration Tests** | `tests/mcp/test_mcp_integration.py` | Protocol compliance & stdio redirection | Subprocess / Mock | ✅ **PASS** |
| **Bob Real Hardware Execution** | Live Bob IDE execution on COM3 | Benchmark returned 'yes' on real STM32U585 | Physical STM32U585 | ✅ **PASS** |

---

## 8. Physical Evidence Audit

### Authoritative Physical Benchmark Metrics
Extracted from `evidence/benchmarks/stm32u585_benchmark_report.json` (Commit `5a5d8eb` / `d98dc56`):

* **Target Device:** STMicroelectronics STM32U585 (ARM Cortex-M33 @ 160.0 MHz)
* **Board FQBN:** `arduino:zephyr:unoq`
* **Serial Port:** `COM3` @ 115200 baud
* **Measurement Source:** `physical_stm32u585`
* **Test Fixture:** `yes.wav` (16 kHz, 16,000 samples)
* **Execution Counts:** `num_iterations = 50`, `warmup_iterations = 5` (Authoritative 50-iteration benchmark)
* **Neural Network Hardware Cycle Count:** `804,196,284 cycles` (Mean)
* **Neural Network Inference Latency:**
  * **Mean Latency:** **`5,026.2267 ms`** ($\sim 5.026\text{ s}$)
  * **Min Latency:** **`5,015.0315 ms`**
  * **Max Latency:** **`5,030.9840 ms`**
  * **Standard Deviation:** **`6.9954 ms`**
* **Prediction Parity:** Index `2` (`yes`), 50/50 iterations exact match:
  ```json
  [-30, -21, 83, -39, -28, -27, -20, -28, -30, -16, -29, -23]
  ```
* **Memory Breakdown:**
  * **Physical Flash Total:** 2,097,152 bytes (2.0 MB)
  * **Application Flash Used:** 310,788 bytes (39.0% of 786.4 KB virtual partition, 1.78 MB physical headroom)
  * **Physical SRAM Total:** 804,864 bytes (786 KB)
  * **Application SRAM Used:** 242,224 bytes (92.0% of 262.1 KB virtual partition, 562.64 KB physical headroom)
  * **Static Tensor Arena:** 166,560 bytes
  * **Static Feature Buffer:** 25,856 bytes

> [!IMPORTANT]
> **Clear Separation of Neural Network Latency vs. End-to-End Physical Transaction Time:**
> * **Model Inference Latency (DWT Cycle Count):** **`5.026 seconds`** (`804,196,284 cycles` @ 160 MHz). This represents the pure neural network forward pass on the Cortex-M33.
> * **End-to-End Physical Transaction Time:** **`~30 seconds`**. This includes $\sim 22.8\text{ s}$ of unoptimized float32 STFT/Mel feature extraction on device, plus UART serial transmission and JSON telemetry transfer.
> * **Host Transaction Timeout:** Pinned to **`40.0 seconds`** to ensure reliable execution margins.

---

## 9. Final Demo Readiness

### 9.1 Demo Readiness

#### Already Proven
* Pinned upstream workload (`reference/tiny-kws` at commit `c097b35`).
* Zero-heap C feature extraction (`src/pipeline/feature_extraction.c`) and INT8 flatbuffer (`src/pipeline/model_data.c`).
* Host parity gate passing 500 samples across S0–S6 with all 2500/2500 checks passing within the defined tolerance.
* Physical compilation and flashing to STM32U585 on Arduino UNO Q via `arduino-cli`.
* Direct Cortex-M33 `DWT->CYCCNT` hardware cycle counting.
* Authoritative 50-iteration physical benchmark with full statistical distribution (5,026.23 ms mean, 6.99 ms stddev).
* Query-response readiness handshake (`?` $\rightarrow$ `CODE2EDGE_READY`) and single-byte command execution.
* MCP stdio redirection preventing JSON-RPC stream corruption.
* Successful live Bob invocation of `benchmark_target_tool` against physical STM32U585 on COM3.

#### Needs Final Verification
* None. All implementation, benchmarking, and tool integration milestones are complete.

#### Still Missing
* None.

---

## 10. Technical & Documentation Risks

| Risk / Gap | Evidence | Impact | Recommended Action |
|---|---|---|---|
| **Compilation Advisory Warning** | `arduino-cli` outputs "Low memory available" advisory warning | Can cause naive compilation scripts to fail | `compile_firmware()` parses memory stats before checking returncode |
| **Feature Extraction Execution Time** | Unoptimized float32 Mel preprocessing takes $\sim 22.8\text{ s}$ on MCU | Short serial timeouts cause premature host aborts | Host transaction timeout is pinned to $40.0\text{ s}$ across all adapters |
| **Historical Target Ambiguity** | Early scaffolds mentioned UNO R4 | Potential confusion regarding target architecture | Target is explicitly clarified as STM32U585 on Arduino UNO Q |

---

## 11. Audit Methodology & Basis

* **Audit Timestamp:** 2026-09-26T20:28:00+05:30
* **Base Git Commit:** `d98dc56` on branch `main`
* **Inspected Directories & Files:**
  * `contracts/` (Target profile, benchmark result, MCP tool schemas)
  * `evidence/` (`stm32u585_benchmark_report.json`, `host_parity_report.json`, `device_parity_yes_uart.txt`, `model_artifact_validation.json`, `real-run-7d942761/`)
  * `mcp_server/` (`server.py`, `adapters/target_adapter.py`, `pipeline_adapter.py`)
  * `src/` (`feature_extraction.c`, `model_data.c`, `model_runner.cpp`, `app.ino`)
  * `tests/` (46 automated tests across `mcp/`, `parity/`, `pipeline/`, `target/`, `firmware/`)
  * `tools/` (`benchmark_target.py`, `check_target.py`, `validate_model_artifact.py`, `run_device_parity.py`)
  * `.bob/mcp.json`
* **Verification Executed During Audit:**
  * Full test suite execution: `python -m pytest tests/ -v` (46 passed).
  * Working tree check: `git status` (clean for tracked files).
  * Benchmark data cross-verification against git history and committed JSON evidence.

---

## 12. Final Summary

### Person Ownership Summary

| Person | DONE | IN PROGRESS | PENDING | Next Action |
|---|---|---|---|---|
| **Person A** (Model / Host) | Reference Ingestion, Model Packaging, C Preprocessing, Host Parity Gate | — | — | Project Complete |
| **Person B** (Firmware / Hardware) | Firmware Harness, DWT Timing, Serial Protocol, Authoritative 50-Run Benchmark | — | — | Project Complete |
| **Person C** (MCP / Bob) | MCP Server, Tool Schemas, Target Adapters, Real Bob Hardware Execution | — | — | Project Complete |

### Milestone Progress Summary

| Milestone | Status | Key Evidence |
|---|---|---|
| **M1: Reference Ingestion & Host Parity** | ✅ **DONE** | `evidence/parity/host_parity_report.json` (500 samples, 2500/2500 PASS) |
| **M2: Edge C Code Generation** | ✅ **DONE** | `src/pipeline/feature_extraction.c`, `src/pipeline/model_data.c` |
| **M3: STM32U585 Firmware & DWT Timing** | ✅ **DONE** | `tests/firmware/benchmark_harness/`, DWT cycle counting verified |
| **M4: On-Device Physical Inference Proof** | ✅ **DONE** | `evidence/benchmarks/stm32u585_benchmark_report.json` ($5.03\text{ s}$, 'yes', logits match) |
| **M5: MCP Server & Tool Contracts** | ✅ **DONE** | 10 MCP tools conforming to schemas, 46 unit tests passing |
| **M6: Authoritative 50-Iteration Benchmark** | ✅ **DONE** | 50-run statistical report: 5,026.23 ms mean, 6.99 ms stddev, 50/50 PASS |
| **M7: Real Bob Deployment Session** | ✅ **DONE** | Successful live Bob invocation of `benchmark_target_tool` against physical STM32U585 on COM3 |
