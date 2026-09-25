# Feasibility Study

## Technical Feasibility

1. **Algorithm Parity:**
   - Mel filterbank calculation is mathematically well-defined and standard in both PyTorch (`torchaudio.transforms.MelSpectrogram`) and CMSIS-DSP (`arm_rfft_fast_f32` with precomputed triangular Mel filter matrix).
   - DS-CNN operations (2D depthwise convolution, 2D pointwise convolution, ReLU, and global average pooling) map 1:1 to optimized CMSIS-NN kernels (`arm_depthwise_conv_s8`, `arm_convolve_s8`, `arm_avgpool_s8`, `arm_fully_connected_s8`).
2. **Numerical Stability:**
   - Normalization using scalar mean/std can be absorbed directly into the input quantization scale and offset.
   - Batch normalization layers in the trained PyTorch model can be mathematically folded into preceding convolution weights and biases during export, eliminating runtime BN computation entirely.

## Resource Requirements

### Target Hardware: STM32U585
- **Core:** ARM Cortex-M33 @ up to 160 MHz
- **Flash:** 2 MB (2048 KB)
- **SRAM:** 786 KB
- **Hardware Acceleration:** FPU, DSP SIMD instructions

### Workload Resource Profile (tiny-kws DS-CNN)
| Metric | Float32 Baseline | Int8 Quantized | Hardware Available | Feasibility Margin |
|---|---|---|---|---|
| **Parameters** | 119,372 | 119,372 | - | - |
| **Model Size (Flash)** | ~478 KB | ~120 KB | 2,048 KB | **~17x headroom** |
| **Activation RAM** | ~330 KB | ~85 KB | 786 KB | **~9x headroom** |
| **Feature Buffer** | 64 x 101 x 4 B = 25.8 KB | 6.5 KB | 786 KB | Negligible |
| **Inference Latency** | ~40-60 ms (est.) | ~10-25 ms (est.) | 1000 ms real-time window | Real-time capable |

## Risks & Mitigations

1. **Risk: Mel Filterbank Discrepancy between torchaudio and CMSIS-DSP**
   - *Mitigation:* Stage S1 and S2 parity gates ensure Mel spectrogram output is validated against torchaudio reference within 1e-4 relative tolerance before running on-device. Precomputed filterbank tables are exported directly from Python to C headers.
2. **Risk: Quantization Drift in DS-CNN**
   - *Mitigation:* Code2Edge isolates intermediate activations per stage (S4a-S4d). If int8 quantization causes error accumulation in deeper DS-blocks, per-channel quantization and bias correction can be applied.
3. **Risk: Arduino UNO Board Variant Ambiguity**
   - *Mitigation:* While Arduino UNO R4 features the Renesas RA4M1 (Cortex-M4), the primary microcontroller target for high-performance TinyML in this project is the STM32U585 (Cortex-M33). Code generation targets standard ARM CMSIS libraries, ensuring cross-compatibility across all Cortex-M targets.
