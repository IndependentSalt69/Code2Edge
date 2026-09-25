# Code2Edge

> **Tagline:** Deployment with proof.

Code2Edge is a Bob-native edge AI compilation, deployment, and differential parity verification pipeline designed to take neural network models from Python/PyTorch prototypes to ultra-low-power microcontrollers, proving mathematical correctness through stage-wise differential parity before and after real hardware deployment.

## Target Hardware

- **Target Board:** **Arduino UNO Q**
- **Target MCU Subsystem:** **STMicroelectronics STM32U585** (ARM Cortex-M33 @ up to 160 MHz, 2 MB Flash, 786 KB SRAM, single-precision FPU, TrustZone, and Armv8-M DSP extensions)
- **Host MPU Subsystem:** Qualcomm Dragonwing QRB2210 (Quad Cortex-A53, Linux) communicating with the STM32U585 via internal UART.
- **Hardware Target Clarification:** Code2Edge targets the **STM32U585 MCU** on the **Arduino UNO Q**. The **Arduino UNO R4** (Renesas RA4M1 Cortex-M4) is **NOT** the target hardware.

## Reference Workload

Code2Edge uses a pinned, frozen reference workload for keyword spotting (KWS):
- **Model:** Depthwise Separable CNN (DS-CNN, 119,372 parameters, fp32 baseline / int8 quantized)
- **Workload Repository:** [tiny-kws](https://github.com/priyadeepjaiswal9c/tiny-kws) by Priyadeep Jaiswal (licensed under the MIT License)
- **Pinned Commit:** `c097b35ae4b9cd585a544c74db16892ce674b186`
- **Vendored Location:** [`reference/tiny-kws/`](reference/tiny-kws/) (see [`reference/tiny-kws/UPSTREAM.md`](reference/tiny-kws/UPSTREAM.md))

## Architecture & Verification Highlights

1. **Two-Tier Differential Parity:**
   - **Tier 1 (Host Parity Gate):** Validates generated C preprocessing and inference against PyTorch reference across 10 discrete stages (S0–S6) with tight numerical tolerances ($\le 10^{-4}$ to $10^{-3}$). Must pass before any embedded compilation occurs.
   - **Tier 2 (On-Device Parity Gate):** Validates physical execution on the STM32U585 Cortex-M33 against golden host reference tensors, ensuring 100% classification agreement.
2. **Authoritative Benchmarking:**
   - Hardware cycle counts via Cortex-M33 DWT cycle counter (`DWT->CYCCNT`).
   - Exact memory usage extracted from linker `.map` files (distinguishing estimated vs measured).
3. **Zero Dynamic Allocation:**
   - Static tensor arenas and ping-pong feature buffers only.

## Attribution & Acknowledgments

- **tiny-kws**: Created by Priyadeep Jaiswal ([GitHub](https://github.com/priyadeepjaiswal9c/tiny-kws)), used under the MIT License.
- **Speech Commands Dataset**: Pete Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition", 2018 (CC-BY-4.0).