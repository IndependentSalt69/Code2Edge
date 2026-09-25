# Architecture

## System Overview

Code2Edge is structured into three primary operational tiers:
1. **Reference Layer (`reference/tiny-kws/`):** Immutable, frozen upstream research workload implementing the gold-standard Python/PyTorch inference pipeline, weights, and feature extraction.
2. **Verification & Parity Tier (`src/parity/`, `src/inference/`, `contracts/target/`):** Two-tier stage-wise differential instrumentation that captures and compares tensors across both host and edge silicon runtimes.
3. **Edge Pipeline & Firmware Tier (`src/pipeline/`, `src/firmware/`, `tools/target/`):** Generated C code for audio feature extraction (CMSIS-DSP) and quantized DS-CNN execution (TFLM / CMSIS-NN) targeting the STM32U585 MCU.

```text
+-----------------------------------------------------------------------------------+
|                                  Code2Edge                                        |
+-----------------------------------------------------------------------------------+
|  [ Reference Workload ] (reference/tiny-kws/)                                     |
|    - Pinned Commit: c097b35ae4b9cd585a544c74db16892ce674b186                      |
|    - Pretrained Checkpoint: checkpoints/best.pt (119k params, DSCNN)              |
|    - Exact Pipeline: 16kHz Audio -> LogMel (64x101) -> Normalization -> DSCNN     |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|  [ Host Inference & Stage Capture ] (src/inference/, src/parity/)                 |
|    - S0: Raw PCM Audio (16,000 samples, float32)                                  |
|    - S1: Mel Power Spectrogram (1, 64, 101)                                       |
|    - S2: Log-Mel Spectrogram (1, 1, 64, 101)                                      |
|    - S3: Normalized Log-Mel (1, 1, 64, 101)                                       |
|    - S4a-d: DS-CNN Internal Tensors (Stem, DS-Blocks 0-3, GAP)                    |
|    - S5: Classification Logits (1, 12)                                            |
|    - S6: Final Class Prediction (int32)                                           |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|  [ TIER 1: Host Differential Parity Gate ] (src/parity/gate.py)                   |
|    - Mandatory Pre-Hardware Gate: Blocks MCU build if any stage exceeds tolerance |
|    - Tolerance Matrix (S0: exact, S1-S3: <=1e-4, S4a-d: <=1e-3, S5: <=0.05)       |
+-----------------------------------------------------------------------------------+
                                         │ (PASS)
                                         ▼
+-----------------------------------------------------------------------------------+
|  [ Target Firmware & MCU Execution ] (src/firmware/, tools/target/)               |
|    - Build System: arduino-cli (arduino:zephyr:unoq)                              |
|    - Target: STM32U585 (ARM Cortex-M33 @ 160 MHz) on Arduino UNO Q               |
|    - Hardware Note: Arduino UNO R4 (Renesas RA4M1) is NOT the target.            |
|    - Runtime: TFLite Micro + CMSIS-NN kernels, static tensor arena                |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|  [ TIER 2: On-Device Differential Parity Gate ] (tools/target/run_device_parity.py|
|    - Compares physical silicon execution against host golden tensors              |
|    - Verifies 100% classification agreement on test fixtures                      |
+-----------------------------------------------------------------------------------+
                                         │ (PASS)
                                         ▼
+-----------------------------------------------------------------------------------+
|  [ Authoritative Benchmarking ] (tools/target/benchmark_target.py)                |
|    - Hardware cycle measurement via DWT_CYCCNT -> Latency (ms)                    |
|    - Flash & SRAM extraction from GCC linker map file                             |
|    - Output strictly distinguishes ESTIMATED vs MEASURED numbers                  |
+-----------------------------------------------------------------------------------+
```

## Components

### 1. Reference Workload (`reference/tiny-kws/`)
- Contains the pinned upstream implementation: feature extraction parameters (`common.py`), network architecture (`model.py`), evaluation scripts (`evaluate.py`), and frozen metrics (`assets/metrics.json`).
- Treated as strictly read-only and governed by upstream MIT license.

### 2. Host Inference Runner (`src/inference/`)
- High-level Python API providing clean execution of the reference pipeline.
- Registers non-invasive PyTorch forward hooks to capture intermediate activations without modifying upstream source files.

### 3. Differential Parity Engine (`src/parity/`)
- Captures reference stage tensors into standardized `.npy` fixtures under `reference/golden/`.
- Compares host reference outputs against edge-generated outputs across all stages (S0 through S6).
- Implements `ParityGate` to enforce numerical tolerance limits before MCU deployment is sanctioned.

### 4. Edge Pipeline Generator (`src/pipeline/`)
- Generates self-contained C code for the complete pipeline.
- Maps feature extraction to CMSIS-DSP primitives (`arm_rfft_fast_f32`).
- Maps quantized DS-CNN operators to TFLite Micro and CMSIS-NN kernels.

### 5. Firmware & Hardware Target Tooling (`src/firmware/`, `tools/target/`)
- `src/firmware/`: Arduino sketch / Zephyr RTOS C++ application for the STM32U585 on Arduino UNO Q.
- `tools/target/`: Target profiling (`check_target`), build orchestration, bridge verification, and authoritative benchmarking (`benchmark_target`).

### 6. Reference Tooling (`tools/reference/`)
- `tools/reference/fetch_checkpoint.py`: Validates and downloads `best.pt` from Hugging Face Hub.
- `tools/reference/check_reference_integrity.py`: Verifies SHA-256 hashes against `reference/tiny-kws/UPSTREAM.md`.

## Data Flow

1. **Audio Input:** 1-second 16 kHz mono WAV loaded into memory.
2. **Feature Generation:** CMSIS-DSP or torchaudio calculates 400-point FFT with 160-hop stride across 64 Mel bins, producing a `(64, 101)` power matrix.
3. **Log Compression & Normalization:** Values are converted via `log(mel + 1e-6)` and normalized using dataset global mean and standard deviation.
4. **Convolutional Processing:**
   - Stem Conv2D (1 -> 160 channels, stride 2)
   - 4 Depthwise Separable Blocks (Depthwise Conv 3x3 + Pointwise Conv 1x1 + BN + ReLU)
   - Global Average Pooling (reducing spatial dimension to `1x1x160`)
5. **Classification:** Fully connected layer projects 160 features into 12 class logits; argmax selects the keyword.
