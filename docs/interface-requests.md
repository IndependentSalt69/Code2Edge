# Person A ↔ Person B Interface Contract & Requests

**Document Status:** Formalized Interface Agreement  
**Participants:** Person A (Host Pipeline, Parity Engine & Code Generation) & Person B (Hardware, Firmware & Benchmarking)

---

## 1. Overview

To guarantee seamless integration between host differential parity and microcontroller deployment, this document defines the exact artifacts, schemas, memory budgets, and sign-off criteria that flow between Person A and Person B.

---

## 2. Interface Exchange Matrix

| ID | Item | Producer | Consumer | Target Path / Schema | Description | Status |
|---|---|---|---|---|---|---|
| **IF-01** | Target Profile | Person B | Person A | `contracts/target/target-profile.schema.json` | Hardware limits (786 KB SRAM, 2 MB Flash), supported ops, no-heap rule | **DEFINED** |
| **IF-02** | Input Contract | Person A | Person B | `contracts/target/input-tensor.json` | Audio input shape `(16000,)`, int16/float32, sampling rate 16 kHz | **DONE / VERIFIED** |
| **IF-03** | Model Artifact | Person A | Person B | `src/pipeline/model_data.h` | C array of quantized int8 TFLite model or CMSIS-NN weight tensors | **DONE / VERIFIED** |
| **IF-04** | Preprocessing C Code | Person A | Person B | `src/pipeline/feature_extraction.h / .c` | Self-contained C implementation of STFT -> Mel -> Log -> Normalize | **DONE / VERIFIED** |
| **IF-05** | Golden Reference Tensors | Person A | Person B | `reference/golden/` | Serialized intermediate tensors (S0–S6) for the smoke-test WAV clips | **DONE / VERIFIED** |
| **IF-06** | Host Parity Sign-off | Person A | Person B | `evidence/parity/host_parity_report.json` | Automated proof that generated C matches PyTorch within tolerance | **DONE / VERIFIED** |
| **IF-07** | Target Benchmark Tool | Person B | Project | `contracts/target/benchmark-result.schema.json` | Execution metrics: latency (ms), Flash (KB), SRAM (KB), arena (KB) | **DEFINED** |
| **IF-08** | Device Parity Report | Person B | Project | `contracts/target/device-parity-result.schema.json` | Verification that on-target inference matches golden tensors | **DEFINED** |

---

## 3. Detailed Specifications

### 3.1 Preprocessing Requirements (Person A → Person B)
- **Header Guard & Signatures:** Clean C99 function signatures:
  ```c
  void feature_extraction_init(void);
  void feature_extraction_run(const int16_t *audio_pcm_16k, int8_t *mel_features_out);
  ```
- **Memory Constraint:** Zero dynamic allocation. All FFT twiddle factors, Hamming windows, and Mel filter matrices must reside in `const` Flash memory (`.rodata`).
- **Scratch Buffer:** Any temporary buffer required for FFT or Mel calculation must be statically declared or provided as a caller-allocated buffer not exceeding 32 KB.

### 3.2 Model Artifact Requirements (Person A → Person B)
- **Format:** C header file containing:
  ```c
  extern const unsigned char g_model_data[];
  extern const int g_model_data_len;
  ```
- **Alignment:** Aligned to a 16-byte boundary (`alignas(16)` or `__attribute__((aligned(16)))`).
- **Quantization:** Standard asymmetric int8 per-channel quantization. Input tensor: `(1, 1, 64, 101)`, int8. Output tensor: `(1, 12)`, int8 or float32.

### 3.3 Golden Test Fixtures (Person A → Person B)
- At least 4 distinct audio test clips from `reference/tiny-kws/app/examples/` (`yes.wav`, `no.wav`, `go.wav`, `stop.wav`).
- For each clip, reference intermediate tensors must be dumped for differential testing:
  - S0: Raw PCM waveform (16000 samples)
  - S1: Mel power spectrogram `(1, 64, 101)`
  - S2: Log-Mel spectrogram `(1, 1, 64, 101)`
  - S3: Normalized features `(1, 1, 64, 101)`
  - S5: Final logits `(1, 12)`
  - S6: Predicted class index (integer)

### 3.4 Hardware Constraints Imposed by Person B (Person B → Person A)
1. **Flash Ceiling:** Total firmware image (Application + Preprocessing + Weights + TFLM) must not exceed 1,500 KB (leaving 548 KB safety margin on STM32U585).
2. **SRAM Ceiling:** Static BSS + Tensor Arena + Feature Scratch must not exceed 400 KB (leaving 386 KB safety margin on STM32U585).
3. **Execution Time Target:** Real-time budget is 1,000 ms; target inference latency is ≤ 100 ms at 160 MHz.

---

## 4. Person C → Team Requests

Person C uses this section to request changes in read-only territory. Do NOT make direct edits to those paths — resolve here and flag in chat.

| Date | From | To | Request | Status |
|------|------|----|---------|--------|
| 2026-09-26 | C | Team | **Folder layout resolved:** `AGENTS.md` initially said code goes in `src/`, team context assumed `pipeline/` and `firmware/` as top-level. Confirmed by `main` commits: A = `reference/` + `src/pipeline/`; B = `src/firmware/` + `tools/target/`; shared = `tests/`, `deploy/` (deploy branch). ✅ Resolved — matches proposed layout. | **Resolved** |
| 2026-09-26 | C | Owner of `.gitignore` | **`.bob` is gitignored:** The `.gitignore` has a bare `.bob` rule, so new shared Bob config files (`.bob/mcp.json`, `.bob/custom_modes.yaml`) will not be tracked. C will use `git add -f` for those two paths only. Preferred fix: change rule to `.bob/*` with explicit exceptions, or remove the `.bob` ignore entirely. | **Open** |
| 2026-09-26 | C | A | **No exported quantized model:** `profile_model_tool` (real mode) needs `total_macs`, `total_params` and a per-layer breakdown, but no `.tflite`/`.onnx`/int8 export exists under `reference/tiny-kws/`. C derived these analytically from `src/model.py`'s `DSCNN(width=160, n_blocks=4)` definition instead (cross-checked: matches `assets/metrics.json`'s `n_parameters=119372` exactly). This is a stand-in, not a substitute — once IF-03's model artifact (or any `.tflite` export) lands, wire `profile_model_tool` to read it directly instead. | **Open — workaround in place** |
| 2026-09-26 | C | Team | **No design-time prediction source for latency/flash:** `benchmark_target_tool`'s `predicted_vs_measured` table needs a "predicted" latency and total firmware flash size to compare against the physical measurement. Neither exists anywhere in the repo yet (the previous adapter code hardcoded 5000.0 ms / 304.0 KB — removed, see `mcp_server/adapters/target_adapter.py`'s `run_benchmark_target`, no real source to replace them with). SRAM already has a real prediction (`check_target`'s arena+feature-buffer sizing). If anyone builds a latency-from-MACs estimator or tracks predicted total flash image size, wire it in here instead of leaving `predicted: "not estimated"`. | **Open** |
| 2026-09-26 | C | A | **Host parity harness isn't parameterized by path:** `run_parity_test_tool(gate="host")` is now wired to `evidence/parity/host_parity_report.json`, but that report comes from `tools/run_host_parity.py`, which hardcodes `compile_c_dll()` to always build `src/pipeline/feature_extraction.c` — there's no argument to point it at a different C source (e.g. a copy sitting in a `deploy/` worktree during a real Bob run). Until it takes a source-path parameter, a real deployment run's host parity gate can only verify identity with Person A's already-committed pipeline, not independently re-verify a generated/copied version of it. Also: `end_to_end.prediction_agreement`/`accuracy_impl` are null in this tool's output because the harness only measures preprocessing-stage parity, not corpus-wide model prediction agreement — flagging in case a full model-inference sweep across the 500-sample corpus is worth adding. | **Open** |
| 2026-09-26 | C | A | **Real pipeline has 4 stages, not 7:** `contracts/inspect_pipeline.schema.json`'s stage-name enum assumes `resample`, `pre_emphasis`, `framing`, `fft`, `mel`, `log`, `normalize` are all separable. The real reference (`reference/tiny-kws/src/common.py:LogMel`) has no resample (input already 16 kHz), no pre-emphasis (not implemented), and computes STFT + mel filterbank inside one fused `torchaudio.transforms.MelSpectrogram` call rather than separate framing/fft functions. `inspect_pipeline_tool` (real mode) now emits only `fft`, `mel`, `log`, `normalize` (4 stages) with a warning, rather than nulling the missing three (the schema's enum doesn't allow a null stage name). Flagging in case this affects how A's generated C++ / the parity harness expect stages to be reported. | **Open — informational, no fix needed unless A's harness assumes 7 stages** |
