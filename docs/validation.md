# Code2Edge Validation Strategy: Two-Tier Differential Parity

**Core Principle:** "Deployment with proof."  
Unlike traditional edge deployment pipelines that treat hardware execution as a black box and evaluate only final top-1 accuracy, Code2Edge enforces **two-tier, stage-wise differential parity**.

---

## 1. Two-Tier Validation Architecture

```text
Host Research Model (PyTorch)
       │
       ▼
 [ Tier 1: Host Parity Gate ] ────(FAIL)────> Bob Repair (Host C Logic)
       │ (PASS)
       ▼
 On-Device Compilation & Flash (STM32U585)
       │
       ▼
 [ Tier 2: Device Parity Gate ] ───(FAIL)────> Bob Repair (MCU Memory / Quantization)
       │ (PASS)
       ▼
 Authoritative Benchmark & Verification Proof
```

### Tier 1: Host-Side Differential Parity (Pre-Hardware Gate)
- **Engine:** `src/parity/compare.py` & `src/parity/gate.py`.
- **Purpose:** Proves that the generated C code executes with numerical equivalence to the PyTorch reference before any embedded compiler or physical silicon is touched.
- **Rule:** Hardware compilation is strictly blocked if Tier 1 fails.

### Tier 2: On-Device Differential Parity (Silicon Verification Gate)
- **Engine:** `tools/target/run_device_parity.py`.
- **Purpose:** Verifies that physical execution on the ARM Cortex-M33 (with real FPU/DSP rounding, int8 MAC overflow handling, and memory alignment) matches the golden host tensors.
- **Rule:** Deployment sign-off requires 100% classification agreement on the golden smoke test corpus.

---

## 2. Stage-Wise Parity Tolerance Matrix

| Stage | Name | Output Shape | Data Type | Host Parity Tolerance | Device Parity Tolerance | Divergence Root Cause Clues |
|---|---|---|---|---|---|---|
| **S0** | Raw Audio PCM | `(16000,)` | int16 / float32 | Binary Exact (0 error) | Binary Exact (0 error) | Resampling, endianness, padding |
| **S1** | Mel Power Spectrogram | `(1, 64, 101)` | float32 | Rel Error $\le 1\times 10^{-4}$ | Rel Error $\le 1\times 10^{-3}$ | Window function, FFT scaling, Mel matrix |
| **S2** | Log-Mel Features | `(1, 1, 64, 101)` | float32 | Rel Error $\le 1\times 10^{-4}$ | Rel Error $\le 1\times 10^{-3}$ | Epsilon addition (`1e-6`), natural vs log10 |
| **S3** | Normalized Features | `(1, 1, 64, 101)` | float32 / int8 | Rel Error $\le 1\times 10^{-4}$ | Rel Error $\le 5\times 10^{-3}$ | Mean/std offset, int8 quantization scale |
| **S4a** | Stem Conv2D Output | `(1, 160, 32, 51)` | float32 / int8 | Rel Error $\le 1\times 10^{-3}$ | Cosine Sim $\ge 0.998$ | Conv stride/pad, fused BN folding |
| **S4b** | DS-Block 0 Output | `(1, 160, 16, 26)` | float32 / int8 | Rel Error $\le 1\times 10^{-3}$ | Cosine Sim $\ge 0.995$ | Depthwise conv per-channel multiplier |
| **S4c** | DS-Block 3 Output | `(1, 160, 16, 26)` | float32 / int8 | Rel Error $\le 1\times 10^{-3}$ | Cosine Sim $\ge 0.990$ | Accumulator saturation in deep layers |
| **S4d** | Global Avg Pool (GAP) | `(1, 160)` | float32 / int8 | Rel Error $\le 1\times 10^{-3}$ | Cosine Sim $\ge 0.990$ | Spatial dimension reduction rounding |
| **S5** | Logits | `(1, 12)` | float32 | Abs Error $\le 0.05$ | Abs Error $\le 0.10$ | Fully connected weights / bias shift |
| **S6** | Predicted Class Index | scalar | int32 | **Exact Match Required** | **Exact Match Required** | Argmax logic |

---

## 3. First-Divergent-Stage Diagnostics

When a differential parity check fails, the report isolates the **earliest stage** in the data flow that crossed the tolerance threshold:
```text
======================================================================
PARITY GATE FAILURE: FIRST DIVERGENT STAGE DETECTED
======================================================================
Test Fixture:           reference/tiny-kws/app/examples/yes.wav
Divergent Stage:        S2 (Log-Mel Features)
Observed Max Abs Error: 0.0412 (Tolerance: 0.0001)
Observed Rel Error:     0.0125 (Tolerance: 0.0001)
Preceding Stage S1:     PASS (Rel Error: 0.00004)

Diagnostic Guidance:
  Check log epsilon addition (1e-6) and verify whether natural log
  or base-10 log is being used in CMSIS-DSP feature generation.
======================================================================
```
Downstream failures are automatically recognized as cascade effects of S2, avoiding futile debugging of neural network weights when the defect lies in audio feature scaling.
