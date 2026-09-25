# Code2Edge

Code2Edge is an edge AI compilation, deployment, and differential parity verification pipeline designed to take neural network models from Python/PyTorch prototypes to ultra-low-power microcontrollers (targeting the STM32U585 on Arduino UNO R4 / UNO Q).

## Reference Workload

Code2Edge uses a pinned, frozen reference workload for keyword spotting (KWS):
- **Model:** Depthwise Separable CNN (DS-CNN, 119k parameters, fp32 / int8)
- **Workload Repository:** [tiny-kws](https://github.com/priyadeepjaiswal9c/tiny-kws) by Priyadeep Jaiswal (licensed under the MIT License)
- **Pinned Commit:** `c097b35ae4b9cd585a544c74db16892ce674b186`
- **Vendored Location:** `reference/tiny-kws/` (see [`reference/tiny-kws/UPSTREAM.md`](reference/tiny-kws/UPSTREAM.md))

## Target Hardware

- **MCU:** STM32U585 (ARM Cortex-M33 with FPU, DSP extensions, and TrustZone)
- **Boards:** Arduino UNO R4 / UNO Q
- **Compute Libraries:** CMSIS-DSP (feature extraction: Mel spectrogram) and CMSIS-NN (neural network inference)

## Attribution & Acknowledgments

- **tiny-kws**: Created by Priyadeep Jaiswal ([GitHub](https://github.com/priyadeepjaiswal9c/tiny-kws)), used under the MIT License.
- **Speech Commands Dataset**: Pete Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition", 2018 (CC-BY-4.0).