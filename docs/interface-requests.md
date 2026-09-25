# Person A ↔ Person B Interface Contract & Requests

**Document Status:** Formalized Interface Agreement  
**Participants:** Person A (Host Pipeline, Parity Engine & Code Generation) & Person B (Hardware, Firmware & Benchmarking)

---

## 1. Overview

To guarantee seamless integration between host differential parity and microcontroller deployment, this document defines the exact artifacts, schemas, memory budgets, and sign-off criteria that flow between Person A and Person B.

---

## 2. Interface Exchange Matrix

| ID | Item | Producer | Consumer | Target Path / Schema | Description | Status |
|---|---|---|---|---|---|---|
| **IF-01** | Target Profile | Person B | Person A | `contracts/target/target-profile.schema.json` | Hardware limits (786 KB SRAM, 2 MB Flash), supported ops, no-heap rule | **DEFINED** |
| **IF-02** | Input Contract | Person A | Person B | `contracts/target/input-tensor.json` | Audio input shape `(16000,)`, int16/float32, sampling rate 16 kHz | **PENDING (Person A)** |
| **IF-03** | Model Artifact | Person A | Person B | `src/pipeline/model_data.h` | C array of quantized int8 TFLite model or CMSIS-NN weight tensors | **PENDING (Person A)** |
| **IF-04** | Preprocessing C Code | Person A | Person B | `src/pipeline/feature_extraction.h / .c` | Self-contained C implementation of STFT -> Mel -> Log -> Normalize | **PENDING (Person A)** |
| **IF-05** | Golden Reference Tensors | Person A | Person B | `reference/golden/` | Serialized intermediate tensors (S0–S6) for the smoke-test WAV clips | **PENDING (Person A)** |
| **IF-06** | Host Parity Sign-off | Person A | Person B | `evidence/parity/host_parity_report.json` | Automated proof that generated C matches PyTorch within tolerance | **PENDING (Person A)** |
| **IF-07** | Target Benchmark Tool | Person B | Project | `contracts/target/benchmark-result.schema.json` | Execution metrics: latency (ms), Flash (KB), SRAM (KB), arena (KB) | **DEFINED** |
| **IF-08** | Device Parity Report | Person B | Project | `contracts/target/device-parity-result.schema.json` | Verification that on-target inference matches golden tensors | **DEFINED** |

---

## 3. Detailed Specifications

### 3.1 Preprocessing Requirements (Person A → Person B)
- **Header Guard & Signatures:** Clean C99 function signatures:
  ```c
  void feature_extraction_init(void);
  void feature_extraction_run(const int16_t *audio_pcm_16k, int8_t *mel_features_out);
  ```
- **Memory Constraint:** Zero dynamic allocation. All FFT twiddle factors, Hamming windows, and Mel filter matrices must reside in `const` Flash memory (`.rodata`).
- **Scratch Buffer:** Any temporary buffer required for FFT or Mel calculation must be statically declared or provided as a caller-allocated buffer not exceeding 32 KB.

### 3.2 Model Artifact Requirements (Person A → Person B)
- **Format:** C header file containing:
  ```c
  extern const unsigned char g_model_data[];
  extern const int g_model_data_len;
  ```
- **Alignment:** Aligned to a 16-byte boundary (`alignas(16)` or `__attribute__((aligned(16)))`).
- **Quantization:** Standard asymmetric int8 per-channel quantization. Input tensor: `(1, 1, 64, 101)`, int8. Output tensor: `(1, 12)`, int8 or float32.

### 3.3 Golden Test Fixtures (Person A → Person B)
- At least 4 distinct audio test clips from `reference/tiny-kws/app/examples/` (`yes.wav`, `no.wav`, `go.wav`, `stop.wav`).
- For each clip, reference intermediate tensors must be dumped for differential testing:
  - S0: Raw PCM waveform (16000 samples)
  - S1: Mel power spectrogram `(1, 64, 101)`
  - S2: Log-Mel spectrogram `(1, 1, 64, 101)`
  - S3: Normalized features `(1, 1, 64, 101)`
  - S5: Final logits `(1, 12)`
  - S6: Predicted class index (integer)

### 3.4 Hardware Constraints Imposed by Person B (Person B → Person A)
1. **Flash Ceiling:** Total firmware image (Application + Preprocessing + Weights + TFLM) must not exceed 1,500 KB (leaving 548 KB safety margin on STM32U585).
2. **SRAM Ceiling:** Static BSS + Tensor Arena + Feature Scratch must not exceed 400 KB (leaving 386 KB safety margin on STM32U585).
3. **Execution Time Target:** Real-time budget is 1,000 ms; target inference latency is $\le 100\text{ ms}$ at 160 MHz.
