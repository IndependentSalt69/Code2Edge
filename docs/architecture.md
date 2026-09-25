# Architecture

## System Overview

Code2Edge is structured into three primary operational layers:
1. **Reference Layer (`reference/tiny-kws/`):** Immutable, frozen upstream research workload implementing the gold-standard Python/PyTorch inference pipeline, weights, and feature extraction.
2. **Verification & Parity Layer (`src/parity/`, `src/inference/`):** Stage-wise differential instrumentation that captures and compares tensors across both host and edge runtimes.
3. **Edge Generation & Deployment Layer (`src/export/`, `deploy/`):** C code generator producing bare-metal C compatible with ARM Cortex-M33 (CMSIS-DSP and CMSIS-NN).

```
+-----------------------------------------------------------------------------------+
|                                  Code2Edge                                        |
+-----------------------------------------------------------------------------------+
|  [ Reference Workload ] (reference/tiny-kws/)                                     |
|    - Pinned Commit: c097b35ae4b9cd585a544c74db16892ce674b186                      |
|    - Pretrained Checkpoint: best.pt (119k params, DSCNN)                          |
|    - Exact Pipeline: 16kHz Audio -> LogMel (64x101) -> Normalization -> DSCNN     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  [ Host Inference & Stage Capture ] (src/inference/, src/parity/)                 |
|    - S0: Raw PCM Audio (16,000 samples, float32)                                  |
|    - S1: Mel Power Spectrogram (1, 64, 101)                                      |
|    - S2: Log-Mel Spectrogram (1, 1, 64, 101)                                      |
|    - S3: Normalized Log-Mel (1, 1, 64, 101)                                       |
|    - S4a-d: DS-CNN Internal Tensors (Stem, DS-Blocks 0-3, GAP)                    |
|    - S5: Classification Logits (1, 12)                                            |
|    - S6: Final Class Prediction (int32)                                           |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  [ Differential Parity Gate ] (src/parity/gate.py)                                |
|    - Tolerance Matrix (S0: exact, S1-S3: <=1e-4, S4a-d: <=1e-3, S5: <=0.01)       |
|    - First-Divergent-Stage Diagnostics                                            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|  [ Edge Export & Firmware Target ] (src/export/, deploy/)                         |
|    - CMSIS-DSP Feature Extraction (arm_rfft_fast_f32, Mel filterbank)             |
|    - CMSIS-NN Quantized DS-CNN Inference (arm_depthwise_conv_s8, arm_convolve_s8) |
|    - Target: STM32U585 (ARM Cortex-M33) on Arduino UNO R4 / UNO Q                 |
+-----------------------------------------------------------------------------------+
```

## Components

### 1. Reference Workload (`reference/tiny-kws/`)
- Contains the pinned upstream implementation: feature extraction parameters (`common.py`), network architecture (`model.py`), evaluation scripts (`evaluate.py`), and frozen metrics (`assets/metrics.json`).
- Treated as strictly read-only and governed by upstream MIT license.

### 2. Inference Runner (`src/inference/`)
- High-level Python API providing clean execution of the reference pipeline.
- Registers non-invasive PyTorch forward hooks to capture intermediate activations without modifying upstream source files.

### 3. Differential Parity Engine (`src/parity/`)
- Captures reference stage tensors into standardized `.npy` fixtures.
- Compares host reference outputs against edge-generated outputs across all stages (S0 through S6).
- Implements `ParityGate` to enforce numerical tolerance limits before MCU deployment is sanctioned.

### 4. Edge Code Generator (`src/export/`)
- Generates self-contained C code for the complete pipeline.
- Maps feature extraction to CMSIS-DSP primitives.
- Maps convolutions, batch normalization (folded), activations, and linear layers to CMSIS-NN kernels.

### 5. Deployment & Tooling (`deploy/`, `tools/`)
- `tools/fetch_checkpoint.py`: Validates and downloads `best.pt` from Hugging Face Hub.
- `tools/check_reference_integrity.py`: Verifies SHA-256 hashes against `reference/tiny-kws/UPSTREAM.md`.

## Data Flow

1. **Audio Input:** 1-second 16 kHz mono WAV loaded into memory.
2. **Feature Generation:** CMSIS-DSP or torchaudio calculates 400-point FFT with 160-hop stride across 64 Mel bins, producing a `(64, 101)` power matrix.
3. **Log Compression & Normalization:** Values are converted via `log(mel + 1e-6)` and normalized using dataset global mean and standard deviation.
4. **Convolutional Processing:**
   - Stem Conv2D (1 -> 160 channels, stride 2)
   - 4 Depthwise Separable Blocks (Depthwise Conv 3x3 + Pointwise Conv 1x1 + BN + ReLU)
   - Global Average Pooling (reducing spatial dimension to `1x1x160`)
5. **Classification:** Fully connected layer projects 160 features into 12 class logits; argmax selects the keyword.
