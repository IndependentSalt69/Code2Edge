# Code2Edge: Preprocessing Integration & Hardware Benchmark Log

**Date:** 2026-09-26  
**Target:** Arduino UNO Q — STMicroelectronics STM32U585 MCU Subsystem (ARM Cortex-M33 @ 160 MHz)  
**Core / FQBN:** `arduino:zephyr` (1.0.0) / `arduino:zephyr:unoq`  
**Toolchain:** `arm-zephyr-eabi-gcc` 1.0.1  
**Workload:** Keyword Spotting Audio Preprocessing (`mel_spectrogram`)  

---

## 1. Executive Summary & Verification Status

This log records the integration of Person A's feature extraction pipeline into the Person B on-device benchmark infrastructure on the physical **Arduino UNO Q (STM32U585)**.

### Master Verification Status Table
| Subsystem / Gate | Status | Evidence Location | Notes |
| :--- | :--- | :--- | :--- |
| **Hardware Bringup (UNO Q / STM32U585)** | **VERIFIED** | [`contracts/target/target-profile.json`](contracts/target/target-profile.json) | Board detected on COM3, toolchain operational |
| **RouterBridge RPC Smoke Test** | **VERIFIED** | [`tests/firmware/router_bridge_smoke/`](tests/firmware/router_bridge_smoke/) | `code2edge_ping() -> 42` over `/dev/ttyHS1` |
| **DWT Benchmark Timing Infrastructure** | **VERIFIED** | [`evidence/benchmarks/benchmark_harness_physical_run.json`](evidence/benchmarks/benchmark_harness_physical_run.json) | 50 iterations, 990 cycles ($6.1875\text{ }\mu\text{s}$, $\sigma=0.00$) |
| **Target Preprocessing Compilation** | **VERIFIED** | [`tests/firmware/benchmark_harness/`](tests/firmware/benchmark_harness/) | $70.40\text{ KB}$ Flash, $36.52\text{ KB}$ SRAM |
| **Host Differential Parity (Stage-Wise)** | **SMOKE PASSED / CORPUS PENDING** | [`evidence/parity/host_parity_report.json`](evidence/parity/host_parity_report.json) | S0–S3 match `sample_0001` ground truth; full 500-sample corpus sweep pending |
| **On-Device Differential Parity** | **NOT VERIFIED / PENDING** | [`contracts/target/device-parity-result.schema.json`](contracts/target/device-parity-result.schema.json) | Awaiting physical UART tensor extraction vs reference |
| **DS-CNN Model Inference** | **NOT VERIFIED / PENDING** | [`contracts/target/target-profile.json`](contracts/target/target-profile.json) | Awaiting TFLite Micro / CMSIS-NN integration |

---

## 2. Frozen Reference & Contract Audit

The preprocessing pipeline implements the exact mathematical sequence frozen in upstream reference [`reference/tiny-kws/src/common.py`](reference/tiny-kws/src/common.py):
$$\text{Raw PCM (16 kHz, 1.0 s)} \xrightarrow{\text{STFT (Hann)}} |X(f)|^2 \xrightarrow{64\text{ Mel HTK}} \text{Mel Energy} \xrightarrow{\ln(x + 10^{-6})} \text{Log-Mel} \xrightarrow{\text{Normalize}} (1, 1, 64, 101)\text{ Features}$$

| Parameter | Specification | Source in Repository |
| :--- | :--- | :--- |
| **Input Audio** | 16,000 samples ($1.0\text{ s}$), 16-bit mono PCM @ 16 kHz | [`contracts/target/input-tensor.json`](contracts/target/input-tensor.json) |
| **STFT Window** | Periodic Hann window ($N=400$, $25\text{ ms}$) | [`reference/tiny-kws/src/common.py`](reference/tiny-kws/src/common.py) #L36 |
| **STFT Hop Length** | $160\text{ samples}$ ($10\text{ ms}$ step size) | [`reference/tiny-kws/src/common.py`](reference/tiny-kws/src/common.py) #L37 |
| **STFT Padding** | Reflection padding ($200\text{ samples}$ on left, $200\text{ samples}$ on right) | [`reference/tiny-kws/src/common.py`](reference/tiny-kws/src/common.py) #L40 |
| **Number of Frames** | 101 frames ($t = 0 \dots 100$) | [`reference/tiny-kws/src/common.py`](reference/tiny-kws/src/common.py) #L40 |
| **FFT Frequency Bins** | 201 positive frequency bins ($0\text{ Hz} \dots 8000\text{ Hz}$) | Real FFT $N/2 + 1$ |
| **Mel Filterbank** | 64 triangular filters ($20.0\text{ Hz} \dots 7600.0\text{ Hz}$), HTK scale | [`reference/tiny-kws/src/common.py`](reference/tiny-kws/src/common.py) #L38–39 |
| **Log Compression** | Natural log with additive $\epsilon = 10^{-6}$: $\ln(\text{energy} + 10^{-6})$ | [`reference/tiny-kws/src/common.py`](reference/tiny-kws/src/common.py) #L41, #L67 |
| **Normalization** | Global mean: $-6.902360439300537$, Global std: $4.81721305847168$ | [`reference/normalization.json`](reference/normalization.json) |
| **Output Tensor** | Shape: `(1, 1, 64, 101)` (6,464 `float32` elements, row-major `[mel * 101 + frame]`) | [`reference/tiny-kws/src/model.py`](reference/tiny-kws/src/model.py) |
| **Test Audio Fixture** | [`reference/tiny-kws/app/examples/yes.wav`](reference/tiny-kws/app/examples/yes.wav) (16 kHz, 16-bit mono, 16,000 samples) | [`tests/firmware/benchmark_harness/fixtures/audio_fixture_yes.h`](tests/firmware/benchmark_harness/fixtures/audio_fixture_yes.h) |

---

## 3. Preprocessing C Implementation Architecture

### 3.1 Flash-Resident Constant Tables (`.rodata`)
- **Hann Window:** `const float g_hann_window[400]` ($1,600\text{ bytes}$)
- **400-Point Twiddle Tables:** `const float g_cos_400[400]` ($1,600\text{ bytes}$) & `const float g_sin_400[400]` ($1,600\text{ bytes}$)
- **Sparse Mel Filterbank:** 370 non-zero weights + 64 band descriptors ($1,736\text{ bytes}$)
- **Total Flash Footprint:** $\mathbf{\approx 6.5\text{ KB}}$

### 3.2 Reflection Padding & Real DFT Routine
```c
static inline int16_t get_padded_sample(const int16_t *audio, int idx) {
    if (idx < 0) {
        idx = -idx; // Reflect around index 0
    } else if (idx >= AUDIO_CLIP_SAMPLES) {
        idx = 2 * (AUDIO_CLIP_SAMPLES - 1) - idx; // Reflect around 15999
    }
    return audio[idx];
}
```
For each frame $t \in [0, 100]$:
1. Samples $n \in [0, 399]$ windowed by `g_hann_window[n]`.
2. Discrete Fourier Transform power spectrum:
   $$\text{Re}(X[k]) = \sum_{n=0}^{399} x[n] \cos\left(\frac{2\pi (k \cdot n \pmod{400})}{400}\right)$$
   $$\text{Im}(X[k]) = -\sum_{n=0}^{399} x[n] \sin\left(\frac{2\pi (k \cdot n \pmod{400})}{400}\right)$$
   $$\text{Power}[k] = \text{Re}(X[k])^2 + \text{Im}(X[k])^2$$
3. Sparse Mel dot product $\rightarrow \ln(x + 10^{-6}) \rightarrow \frac{x - \mu}{\sigma}$.
4. Stored in row-major layout `mel_features_out[mel_idx * 101 + frame_idx]`.

---

## 4. Benchmark Harness Integration (`benchmark_harness.ino`)

The benchmark harness executes two distinct workloads:
1. **`deterministic_kernel` (50 iterations):** Validates DWT cycle counting infrastructure.
2. **`mel_spectrogram` (10 iterations, 2 warmups):** Measures real STFT + Mel feature extraction on `g_audio_fixture_yes`.

### Interactive Serial Commands (115200 Baud)
- `'B'` or `'b'` / `'R'` or `'r'` : **Run All Benchmarks** — Runs both deterministic validation and `mel_spectrogram` benchmarks.
- `'M'` or `'m'` / `'P'` or `'p'` : **Run Preprocessing Benchmark Only** — Runs `mel_spectrogram` benchmark.
- `'D'` or `'d'` : **Host/Device Parity Dump Hook** — Streams the feature checksum (`0xD7F1D853`), tensor shape (`[1,1,64,101]`), and boundary float values over UART for automated differential parity gates.
- `'K'` or `'k'` : **Run Infrastructure Benchmark Only** — Runs `deterministic_kernel` validation only.

---

## 5. Compilation & Memory Analysis

### Compiler Output Summary (`arduino-cli compile --fqbn arduino:zephyr:unoq tests/firmware/benchmark_harness`)
```text
Sketch uses 128904 bytes (16%) of program storage space. Maximum is 786432 bytes.
Global variables use 65140 bytes (24%) of dynamic memory, leaving 197004 bytes for local variables. Maximum is 262144 bytes.
```

### Linker Map Section Measurements (`tools/target/parse_map.py`)
| Linker Section | Type | Size (Bytes) | Size (KB) | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `.text` | Code Flash | 27,620 | 26.97 KB | Application code + Preprocessing math |
| `.rodata` | Constants Flash | 968 | 0.95 KB | Global string constants |
| `.llext.rodata.noreloc` | Zephyr LLEXT | 41,796 | 40.82 KB | Audio PCM fixture + Mel filterbank tables |
| `.data` | Initialized SRAM | 12 | 0.01 KB | Initialized static variables |
| `.bss` | Zero-Init SRAM | 36,508 | 35.65 KB | DWT cycle records + Output feature buffer (6,464 floats) |
| **Total App Flash** | **`.text + .rodata + .data + .llext`** | **70,396** | **68.75 KB** | **3.36% of 2 MB Flash (2,026,756 B Headroom)** |
| **Total App SRAM** | **`.data + .bss`** | **36,520** | **35.66 KB** | **4.54% of 786 KB SRAM (768,344 B Headroom)** |

---

## 6. Stage-Wise Host Parity Verification Results

Executed [`tools/run_host_parity.py`](tools/run_host_parity.py) comparing the C pipeline against frozen ground truth artifacts in [`reference/artifacts/sample_0001/`](reference/artifacts/sample_0001/):

| Stage | Name | Ground Truth Artifact | Max Abs Diff | Mean Abs Diff | Cosine Sim | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S0** | Raw Audio Waveform | `post_input.npy` `(16000,)` | $0.000000$ | $0.000000$ | $1.00000000$ | **PASS** |
| **S1** | Power Spectrum STFT | `post_power_spectrum.npy` `(1, 201, 101)` | $3.173828 \times 10^{-3}$ | $1.039262 \times 10^{-5}$ | $1.00000000$ | **PASS** |
| **S1_mel** | Mel Filterbank Energy | `post_mel.npy` `(1, 64, 101)` | $5.126953 \times 10^{-3}$ | $1.020852 \times 10^{-4}$ | $1.00000000$ | **PASS** |
| **S2** | Log-Mel Features | `post_log.npy` `(1, 1, 64, 101)` | $2.631187 \times 10^{-3}$ | $1.680750 \times 10^{-4}$ | $1.00000000$ | **PASS** |
| **S3** | Normalized Log-Mel | `post_normalize.npy` `(1, 1, 64, 101)` | $5.459785 \times 10^{-4}$ | $3.489649 \times 10^{-5}$ | $1.00000000$ | **PASS** |

Parity report saved to: [`evidence/parity/host_parity_report.json`](evidence/parity/host_parity_report.json).

---

## 7. Files Created & Modified

1. **[`src/pipeline/feature_extraction.h`](src/pipeline/feature_extraction.h)** — Canonical Person A C header declaring preprocessing parameters, constants, and API.
2. **[`src/pipeline/feature_extraction.c`](src/pipeline/feature_extraction.c)** — Canonical Person A self-contained C implementation of STFT, Mel filterbank, log-compression, and global normalization.
3. **[`tests/firmware/benchmark_harness/fixtures/audio_fixture_yes.h`](tests/firmware/benchmark_harness/fixtures/audio_fixture_yes.h)** — Aligned 16,000-sample `int16_t` PCM test fixture extracted from `yes.wav`.
4. **[`tests/firmware/benchmark_harness/feature_extraction.h`](tests/firmware/benchmark_harness/feature_extraction.h)** — Local sketch include for self-contained Arduino CLI compilation.
5. **[`tests/firmware/benchmark_harness/feature_extraction.c`](tests/firmware/benchmark_harness/feature_extraction.c)** — Local sketch source for self-contained Arduino CLI compilation.
6. **[`tests/firmware/benchmark_harness/benchmark_harness.ino`](tests/firmware/benchmark_harness/benchmark_harness.ino)** — Embedded benchmark sketch with `mel_spectrogram` workload and B/R, M/P, D, K interactive commands.
7. **[`tools/reference/generate_preprocessing_c.py`](tools/reference/generate_preprocessing_c.py)** — Automated table and test fixture generator from frozen reference.
8. **[`tools/run_host_parity.py`](tools/run_host_parity.py)** — Stage-wise host parity comparison tool against frozen PyTorch reference artifacts.
9. **[`tests/pipeline/test_preprocessing_parity.py`](tests/pipeline/test_preprocessing_parity.py)** — Pytest test suite asserting stage-wise parity against ground truth.
10. **[`evidence/parity/host_parity_report.json`](evidence/parity/host_parity_report.json)** — Machine-readable host parity report.
11. **[`contracts/target/target-profile.json`](contracts/target/target-profile.json)** — Hardware profile with verified physical toolchain & RouterBridge metadata.
12. **[`PREPROCESSING_INTEGRATION_LOG.md`](PREPROCESSING_INTEGRATION_LOG.md)** — This comprehensive log.

---

## 8. Next Hardware Execution Step

Flash the compiled preprocessing benchmark harness to the STM32U585:

```bash
# 1. Upload sketch to Arduino UNO Q over COM3
arduino-cli upload -p COM3 --fqbn arduino:zephyr:unoq tests/firmware/benchmark_harness

# 2. Monitor serial console at 115200 baud
arduino-cli monitor -p COM3 --config baudrate=115200
```
