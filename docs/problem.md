# Problem Statement

## Overview

Deploying modern deep learning models to ultra-low-power microcontrollers (MCUs) presents steep engineering challenges:
1. **Toolchain Disconnect:** Transitioning from high-level Python/PyTorch research code to low-level embedded C/C++ (CMSIS-DSP, TFLite Micro, CMSIS-NN) frequently introduces subtle mathematical divergences, endianness issues, dynamic range clipping, and numerical drift.
2. **Lack of Stage-Wise Verifiability:** Most deployment pipelines treat the on-device model as a black box, comparing only final top-1 predictions. When classification accuracy degrades on target hardware, engineers lack granular visibility to identify which stage diverged (e.g., STFT windowing, Mel filterbank geometry, logarithmic scaling, normalization offsets, quantization scaling, or depthwise convolutions).
3. **Hardware & Memory Constraints:** The target MCU (**STM32U585** on the **Arduino UNO Q**) has a fixed hardware envelope: 786 KB SRAM and 2 MB Flash. It requires zero-dynamic-allocation guarantees and quantized weights/activations without sacrificing accuracy.

> **Target Platform Clarification:** Code2Edge explicitly targets the **STM32U585 MCU** of the **Arduino UNO Q**. The Arduino UNO R4 (Renesas RA4M1) is **NOT** the target hardware.

## Objectives

- **Automated Two-Tier Differential Parity:** Instrument the full audio inference pipeline across host PyTorch, host generated C, and edge silicon at every intermediate stage (Raw Audio -> Mel Spectrogram -> Log Compression -> Normalization -> Stem -> DS-Blocks -> Global Avg Pool -> Logits -> Class Prediction) to pinpoint the earliest stage of numerical divergence.
- **Reference Workload Grounding:** Anchor all benchmarking to a proven, frozen keyword spotting (KWS) workload ([`tiny-kws`](file:///D:/Projects/Code2Edge/reference/tiny-kws)) achieving >96.5% accuracy on Google Speech Commands V2.
- **Target Edge Generation:** Provide a C code-generation and deployment pathway targeting the ARM Cortex-M33 (STM32U585) utilizing CMSIS-DSP for audio feature extraction and CMSIS-NN / TFLM for quantized tensor execution.
- **Strict Parity Gating:** Establish an automated CI/test gate that blocks deployment if host intermediate stage tolerances are exceeded, followed by on-device silicon parity verification.

## Target Audience & Use Cases

- **Embedded Systems & Firmware Engineers:** Requiring reproducible, mathematically verifiable C libraries for tinyML edge audio.
- **Machine Learning Engineers:** Transitioning speech and audio models from PyTorch prototypes to production edge silicon with proof of mathematical equivalence.
- **Edge AI Researchers:** Benchmarking quantization schemes, CMSIS-NN kernel optimizations, and memory footprints with verifiable predicted-vs-measured metrics.
