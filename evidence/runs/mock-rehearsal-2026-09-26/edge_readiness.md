# Edge Readiness Report

Run: `run-30027352`  Data: **mock rehearsal**

## Fit
- fits: **True**
- SRAM budget: 786 KB, used: 135.7 KB, headroom: 650.3 KB
- Flash budget: 2048 KB, model: 32.4 KB
- Arena: 128 KB, feature buffer: 7.7 KB
- Unsupported ops: none
- Warnings: ['AVERAGE_POOL_2D with kernel > 3x3 may be slow without CMSIS-NN acceleration']

## Model
- DS-CNN, 24922 params, 5740544 MACs, int8

## Pipeline constants (7 stages, domain=kws)
- sample_rate: 16000
- window_length: 400
- hop_length: 320
- n_fft: 512
- n_mels: 40
- n_frames: 49
- pre_emphasis_coeff: 0.97
- norm_mean: -4.2
- norm_std: 2.1

## Risks
- Mock data only (rehearsal) — real parity numbers come from Person A/B adapters once wired.
- AVERAGE_POOL_2D kernel size may need CMSIS-NN acceleration on-device.
