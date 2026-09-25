# Feasibility Study

## Technical Feasibility

1. **Algorithm Parity:**
   - Mel filterbank calculation is mathematically well-defined and standard in both PyTorch (`torchaudio.transforms.MelSpectrogram`) and CMSIS-DSP (`arm_rfft_fast_f32` with precomputed triangular Mel filter matrix).
   - DS-CNN operations (2D depthwise convolution, 2D pointwise convolution, ReLU, and global average pooling) map 1:1 to optimized CMSIS-NN kernels (`arm_depthwise_conv_s8`, `arm_convolve_s8`, `arm_avgpool_s8`, `arm_fully_connected_s8`) and standard TFLite Micro kernels.
2. **Numerical Stability:**
   - Normalization using scalar mean/std can be absorbed directly into the input quantization scale and offset.
   - Batch normalization layers in the trained PyTorch model can be mathematically folded into preceding convolution weights and biases during export, eliminating runtime BN computation entirely.

## Resource Requirements & Memory Headroom

### Target Hardware: STMicroelectronics STM32U585 on Arduino UNO Q
- **Core:** ARM Cortex-M33 @ up to 160 MHz
- **Flash:** 2,048 KB (2.0 MB)
- **SRAM:** 786 KB
- **Hardware Acceleration:** Single-precision FPU, Armv8-M DSP instructions
- **Hardware Target Clarification:** Code2Edge targets the **STM32U585 MCU subsystem** on the **Arduino UNO Q**. The **Arduino UNO R4** (Renesas RA4M1 Cortex-M4) is **NOT** the target hardware.

### Workload Resource Profile: tiny-kws DS-CNN (Estimated vs Measured)
> [!NOTE]
> Values marked **ESTIMATED** are analytical projections based on model parameters and architecture. Values marked **MEASURED** are pending execution of `tools/target/benchmark_target.py` on physical silicon.

| Metric | Float32 Baseline (Est.) | Int8 Quantized (Est.) | Physical Measured | Hardware Limit | Feasibility Headroom |
|---|---|---|---|---|---|
| **Parameters** | 119,372 | 119,372 | *Pending run* | — | — |
| **Model Size (Flash)** | ~478 KB | ~120 KB | *Pending run* | 2,048 KB | **~17x headroom** |
| **Activation RAM / Arena** | ~330 KB | ~45–85 KB | *Pending run* | 786 KB | **~9x headroom** |
| **Feature Buffer (SRAM)** | 25.8 KB (64x101x4B) | 6.5 KB (int8) | *Pending run* | 786 KB | Negligible (~1%) |
| **Inference Latency** | ~40–60 ms (est.) | ~10–25 ms (est.) | *Pending run* | 1000 ms real-time window | Real-time capable |

## Risks & Mitigations

1. **Risk: Mel Filterbank Discrepancy between torchaudio and CMSIS-DSP**
   - *Mitigation:* Stage S1 and S2 parity gates ensure Mel spectrogram output is validated against torchaudio reference within $1\times 10^{-4}$ relative tolerance on host before running on-device. Precomputed filterbank tables are exported directly from Python to C headers.
2. **Risk: Quantization Drift in DS-CNN**
   - *Mitigation:* Code2Edge isolates intermediate activations per stage (S4a–S4d). If int8 quantization causes error accumulation in deeper DS-blocks, per-channel quantization and bias correction are applied.
3. **Risk: Target Board Clarification**
   - *Resolution:* Target is strictly confirmed as the **STM32U585 MCU** on the **Arduino UNO Q**. Arduino UNO R4 references have been eradicated from the deployment target definition.
4. **Risk: Toolchain & Flashing Pipeline**
   - *Mitigation:* Person B executes a Day-1 dummy-model build, flash, run, and readback proof to eliminate all build and firmware communication uncertainties before Person A delivers the full generated pipeline.
