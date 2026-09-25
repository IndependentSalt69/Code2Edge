# Problem Statement

## Overview

Deploying modern deep learning models to ultra-low-power microcontrollers (MCUs) presents steep engineering challenges:
1. **Toolchain Disconnect:** Transitioning from high-level Python/PyTorch research code to low-level bare-metal C (CMSIS-DSP, CMSIS-NN) often introduces subtle mathematical divergences, endianness issues, dynamic range clipping, and numerical drift.
2. **Lack of Stage-Wise Verifiability:** Most deployment pipelines treat the on-device model as a black box, comparing only final outputs. When classification accuracy degrades on target hardware, engineers lack granular visibility to identify which stage diverged (e.g., STFT windowing, Mel filterbanks, logarithmic scaling, batch normalization folding, quantization, or depthwise convolutions).
3. **Hardware & Memory Constraints:** The target MCU (STM32U585) has restricted SRAM (786 KB) and Flash (2 MB), requiring strict zero-dynamic-allocation guarantees and quantized weights/activations without sacrificing accuracy.

## Objectives

- **Automated Differential Parity:** Instrument the full audio inference pipeline across host PyTorch and edge C runtimes at every intermediate stage (Raw Audio -> Mel Spectrogram -> Log Compression -> Normalization -> Stem -> DS-Blocks -> Global Avg Pool -> Logits -> Class Prediction) to pinpoint exact numerical divergence.
- **Reference Workload Grounding:** Anchor all benchmarking to a proven, frozen keyword spotting (KWS) workload ([`tiny-kws`](file:///D:/Projects/Code2Edge/reference/tiny-kws)) achieving >96.5% accuracy on Google Speech Commands V2.
- **Target Edge Generation:** Provide a C code-generation and deployment pathway targeting the ARM Cortex-M33 (STM32U585) utilizing CMSIS-DSP for audio feature extraction and CMSIS-NN for quantized tensor execution.
- **Strict Parity Gating:** Establish an automated CI/test gate that blocks deployment if intermediate stage tolerances are exceeded.

## Target Audience & Use Cases

- **Embedded Systems & Firmware Engineers:** Requiring reproducible, mathematically verifiable C libraries for tinyML edge audio.
- **Machine Learning Engineers:** Transitioning speech and audio models from PyTorch prototypes to production edge silicon with confidence.
- **Edge AI Researchers:** Benchmarking quantization schemes, CMSIS-NN kernel optimizations, and memory footprints.
