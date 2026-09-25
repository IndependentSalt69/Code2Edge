# CODE2EDGE REFERENCE AUDIT

**Repository:** `D:\tinyml-compression-benchmark`
**Audit Date:** 2026-09-26
**Auditor:** Code2Edge Technical Audit (automated)
**Target MCU:** STM32U585 (see §12 note on Arduino UNO Q ambiguity)
**Workload:** Keyword Spotting / DS-CNN

> **CRITICAL FINDING:** The tinyml-compression-benchmark repository is an **early-stage research scaffold**. Only Phase 1 (dataset exploration and 12-class protocol definition) is complete. **No model, no training code, no preprocessing pipeline, no inference code, no checkpoints, and no evaluation results exist.** The repository cannot serve as a Code2Edge reference workload in its current state.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Dataset](#2-dataset)
3. [Complete Inference Pipeline](#3-complete-inference-pipeline)
4. [Model](#4-model)
5. [Existing Results](#5-existing-results)
6. [Inference Script](#6-inference-script)
7. [Preprocessing Parity Risks](#7-preprocessing-parity-risks)
8. [Export / Conversion Pipeline](#8-export--conversion-pipeline)
9. [Code2Edge Reference Candidate](#9-code2edge-reference-candidate)
10. [Reference Tensor Dump Design](#10-reference-tensor-dump-design)
11. [Host Parity Design](#11-host-parity-design)
12. [Hardware Readiness Analysis](#12-hardware-readiness-analysis)
13. [Code2Edge MCP Tool Requirements](#13-code2edge-mcp-tool-requirements)
14. [Risks / Gaps](#14-risks--gaps)
15. [Final Executive Summary](#15-final-executive-summary)

---

## 1. Repository Structure

### 1.1 Verified Source File Inventory

The repository contains **30 git commits** and has the following non-empty, substantive files:

| File | Size | Purpose | Contains Code? | Relevant for Code2Edge? |
|------|------|---------|----------------|-------------------------|
| `src/data/dataset.py` | 2,377 B | PyTorch `Dataset` wrapper around `torchaudio.datasets.SPEECHCOMMANDS` | Yes | Yes — raw waveform loading |
| `src/data/explore_dataset.py` | 4,710 B | Raw 35-class dataset characterization & split analysis | Yes | Informational only |
| `src/data/analyze_12class_protocol.py` | 5,509 B | 12-class protocol feasibility: target/unknown/silence class counts | Yes | Yes — defines label set |
| `src/data/diagnose_12class.py` | 1,714 B | Background noise window counting diagnostics | Yes | Informational only |
| `src/data/diagnose_unknown.py` | 1,692 B | Unknown-class diagnostics | Yes | Informational only |
| `experiments/001_raw_dataset_characterization.md` | 4,741 B | Experiment 001 report: raw dataset statistics | Documentation | Yes — verified numbers |
| `experiments/experiments/002_12_class_protocol.md` | 7,003 B | Experiment 002 report: 12-class KWS protocol definition | Documentation | Yes — protocol spec |
| `papers/reading_notes/hello_edge.md` | 1,134 B | Literature notes on "Hello Edge" paper | Notes | No |
| `requirements.txt` | 1,834 B | Frozen pip dependencies (PyTorch 2.11 + CUDA 12.8) | Config | Yes — dependency baseline |
| `README.md` | 8,355 B | Project overview with roadmap | Documentation | Yes — roadmap context |
| `mindmap.md` | 967 B | ASCII research scope diagram | Documentation | No |
| `plots/dataset/raw_class_distribution.png` | 164,082 B | Bar chart of 35-class training distribution | Artifact | No |

### 1.2 Empty / Placeholder Files (0 bytes)

These files exist but contain **zero bytes**:

| File | Intended Purpose | Status |
|------|------------------|--------|
| `src/data/preprocessing.py` | MFCC / feature extraction | **NOT IMPLEMENTED** |
| `src/data/transforms.py` | Data augmentation transforms | **NOT IMPLEMENTED** |
| `src/data/visualize.py` | Visualization utilities | **NOT IMPLEMENTED** |
| `configs/baseline.yaml` | Baseline DS-CNN config | **NOT IMPLEMENTED** |
| `configs/default.yml` | Default config | **NOT IMPLEMENTED** |
| `configs/ptq.yaml` | Post-Training Quantization config | **NOT IMPLEMENTED** |
| `configs/qat.yaml` | Quantization-Aware Training config | **NOT IMPLEMENTED** |
| `configs/pruning.yaml` | Pruning config | **NOT IMPLEMENTED** |
| `configs/distillation.yaml` | Knowledge Distillation config | **NOT IMPLEMENTED** |
| `environment.yml` | Conda environment | **NOT IMPLEMENTED** |
| `experiments/001_dataset_exploration.md` | Duplicate/empty experiment doc | Empty |
| `experiments/002_preprocessing.md` | Preprocessing experiment doc | Empty |
| `experiments/003_baseline_training.md` | Baseline training experiment doc | Empty |
| `papers/decision_log.md` | Decision log | Empty |

### 1.3 Empty Directories (Planned but Unpopulated)

These directories exist but contain **no files**:

| Directory | Intended Purpose |
|-----------|------------------|
| `src/models/` | DS-CNN and compressed model definitions |
| `src/training/` | Trainer routines |
| `src/evaluation/` | Benchmark evaluation metrics |
| `src/quantization/` | PTQ & QAT modules |
| `src/pruning/` | Pruning methods |
| `src/distillation/` | Knowledge distillation pipelines |
| `src/deployment/` | TFLite / C++ export & profiling |
| `src/utils/` | Utility functions |
| `tests/` | Test files (directory exists but empty) |
| `notebooks/` | Jupyter notebooks (empty) |
| `logs/` | Training logs (only `.gitkeep`) |
| `results/` | Results JSON (only `.gitkeep`) |

### 1.4 README Roadmap vs. Reality

The README documents a 5-phase roadmap. **Only Phase 1 is complete:**

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Dataset Characterization & Protocol Definition | **Completed** |
| Phase 2 | Feature Extraction Pipeline (MFCC, augmentation) | **Not started** — `preprocessing.py` & `transforms.py` are 0 bytes |
| Phase 3 | DS-CNN Baseline Model | **Not started** — `src/models/` is empty |
| Phase 4 | Compression Experiments (PTQ, QAT, Pruning, Distillation) | **Not started** |
| Phase 5 | Evaluation & Edge Profiling | **Not started** |

**Source:** `README.md`, "Benchmark Roadmap" section.

---

## 2. Dataset

### 2.1 VERIFIED Facts

| Property | Value | Source |
|----------|-------|--------|
| Dataset | Google Speech Commands V2 | `src/data/dataset.py` L12: `from torchaudio.datasets import SPEECHCOMMANDS`; on disk at `datasets/SpeechCommands/speech_commands_v0.02/` |
| Dataset version | V2 (`speech_commands_v0.02`) | Directory name on disk; tar.gz at `datasets/speech_commands_v0.02.tar.gz` |
| Dataset root | `datasets/` (relative to project root) | `src/data/dataset.py` L15: `DATASET_ROOT = Path("datasets")` |
| Loading mechanism | `torchaudio.datasets.SPEECHCOMMANDS` | `src/data/dataset.py` L30 |
| Split mechanism | Official `validation_list.txt` / `testing_list.txt` | `src/data/explore_dataset.py` L55–63 |
| Total spoken-word WAV files | 105,829 | `experiments/001_raw_dataset_characterization.md` §3 |
| Training split | 84,843 spoken-word samples | Same source |
| Validation split | 9,981 spoken-word samples | Same source |
| Testing split | 11,005 spoken-word samples | Same source |
| Raw vocabulary | 35 word classes | Same source |
| Number of benchmark classes | 12 | `experiments/experiments/002_12_class_protocol.md` §2 |
| Target keywords | `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go` | `src/data/analyze_12class_protocol.py` L31–42 |
| Unknown class | Aggregated from 25 non-target word categories | Same file, `count_target_and_unknown()`, L102–111 |
| Silence class | 1-sec windows from `_background_noise_/*.wav`, 50% overlap | Same file, L45–47: `WINDOW_LENGTH = SAMPLE_RATE`, `HOP_LENGTH = SAMPLE_RATE // 2` |
| Validation background file | `running_tap.wav` only | Same file, L49: `VALIDATION_BACKGROUND_FILE = "running_tap.wav"` |
| Training silence windows | 668 | `experiments/experiments/002_12_class_protocol.md` §7 |
| Validation silence windows | 121 | Same source |
| Assumed sample rate | 16,000 Hz | `src/data/analyze_12class_protocol.py` L44: `SAMPLE_RATE = 16_000` |
| Audio duration | 1 second (16,000 samples at 16 kHz) | Implicit from `WINDOW_LENGTH = SAMPLE_RATE` |
| Channels | Mono (1 channel) | Speech Commands V2 standard; `diagnose_12class.py` prints `channels=1` |

### 2.2 Code Path: WAV → Dataset Object → Tensor

**VERIFIED path** (as implemented in `src/data/dataset.py`):

```
WAV file on disk (datasets/SpeechCommands/speech_commands_v0.02/<word>/<file>.wav)
  ↓
torchaudio.datasets.SPEECHCOMMANDS.__getitem__(index)
  ↓
Returns: (waveform: Tensor, sample_rate: int, label: str, speaker_id: str, utterance_number: int)
  ↓
SpeechCommandsDataset.__getitem__() wraps into dict:
  {"waveform": Tensor[1, N], "sample_rate": 16000, "label": str, ...}
```

- `waveform` dtype: `torch.float32` (torchaudio default)
- `waveform` shape: `[1, N]` where `N` varies per sample (most are 16,000 but some shorter)
- Value range: `[-1.0, 1.0]` (torchaudio WAV loading default)

### 2.3 UNKNOWN / NOT IMPLEMENTED

| Property | Status |
|----------|--------|
| Padding/truncation to fixed length | **NOT IMPLEMENTED** |
| Augmentation | **NOT IMPLEMENTED** — `transforms.py` is 0 bytes |
| MFCC / Mel spectrogram extraction | **NOT IMPLEMENTED** — `preprocessing.py` is 0 bytes |
| Normalization (mean/std) | **NOT IMPLEMENTED** |
| 12-class label mapping in dataset | **NOT IMPLEMENTED** — `dataset.py` returns raw word labels |
| Silence sample generation in dataset | **NOT IMPLEMENTED** — only analyzed, not integrated |
| Unknown class sampling strategy | **NOT FINALIZED** — Experiment 002 §12 open question |
| Deterministic seed | **NOT IMPLEMENTED** |
| Test set composition for 12-class | **NOT FINALIZED** — Experiment 002 §7 |

---

## 3. Complete Inference Pipeline

> **NO INFERENCE PIPELINE EXISTS.**
>
> There is no preprocessing code, no model, no inference script, and no evaluation code in this repository.

### What Would Be Needed

Based on the project's stated goals (DS-CNN for 12-class KWS):

```
RAW WAV (16 kHz, mono, ~1 second)
  ↓
[PADDING/TRUNCATION to 16,000 samples] — NOT IMPLEMENTED
  ↓
[MFCC or Log-Mel feature extraction] — NOT IMPLEMENTED
  ↓
[NORMALIZATION] — NOT IMPLEMENTED
  ↓
MODEL INPUT TENSOR — NOT IMPLEMENTED
  ↓
DS-CNN — NOT IMPLEMENTED
  ↓
MODEL OUTPUT (logits for 12 classes) — NOT IMPLEMENTED
  ↓
ARGMAX → predicted class index — NOT IMPLEMENTED
  ↓
LABEL LOOKUP → predicted keyword — NOT IMPLEMENTED
```

**Every stage above is NOT IMPLEMENTED.**

---

## 4. Model

> **NO MODEL EXISTS.**

### 4.1 Verified Facts

| Property | Status |
|----------|--------|
| Model class/file | **DOES NOT EXIST** — `src/models/` is empty |
| DS-CNN implementation | **NOT IMPLEMENTED** |
| Architecture definition | **NOT IMPLEMENTED** |
| Input tensor shape | **UNKNOWN** |
| Output tensor shape | **UNKNOWN** (presumably `[batch, 12]`) |
| Parameter count | **UNKNOWN** |
| Activation functions | **UNKNOWN** |
| Checkpoint file | **DOES NOT EXIST** — no `.pt`, `.pth`, `.onnx`, `.tflite`, `.h5`, `.pb`, `.ckpt`, or `.bin` found |
| Checkpoint format | **N/A** |
| Weight loading code | **DOES NOT EXIST** |
| BatchNorm | **UNKNOWN** |
| Dropout | **UNKNOWN** |
| Model variant | **N/A** — no model exists |

### 4.2 Model Variants

**No model variants exist.** All compression directories and config files are empty.

### 4.3 Code2Edge Reference Candidate

**CANNOT RECOMMEND A MODEL.** No trained model, definition, or checkpoint exist.

---

## 5. Existing Results

> **NO RESULTS EXIST.**

| Metric | Status | Source |
|--------|--------|--------|
| Validation accuracy | **NOT AVAILABLE** | `results/` contains only `.gitkeep` |
| Test accuracy | **NOT AVAILABLE** | Same |
| Loss | **NOT AVAILABLE** | Same |
| Precision / Recall / F1 | **NOT AVAILABLE** | Same |
| Confusion matrix | **NOT AVAILABLE** | Same |
| Model size | **NOT AVAILABLE** | No model exists |
| Parameter count | **NOT AVAILABLE** | Same |
| FLOPs | **NOT AVAILABLE** | Same |
| Latency | **NOT AVAILABLE** | Same |
| Compression results | **NOT AVAILABLE** | Same |
| Training logs | **NOT AVAILABLE** | `logs/` contains only `.gitkeep` |

Only output artifact: `plots/dataset/raw_class_distribution.png` (35-class bar chart from Experiment 001).

---

## 6. Inference Script

> **NO INFERENCE SCRIPT EXISTS.**

No `inference.py`, `evaluate.py`, or any evaluation code anywhere in the repository.

### Training/Inference Preprocessing Parity

**Cannot be assessed.** Neither training nor inference preprocessing exist.

---

## 7. Preprocessing Parity Risks

> Since no preprocessing pipeline is implemented, this section documents **anticipated** risks based on the project's stated goals (MFCC-based DS-CNN) and standard KWS practice.

| # | Stage | Anticipated Python Behavior | Source of Risk | Why MCU Parity Could Fail | How to Validate |
|---|-------|----------------------------|----------------|--------------------------|-----------------|
| 1 | Audio loading | torchaudio loads WAV as `float32 [-1, 1]` | `src/data/dataset.py` L30 | MCU reads raw PCM int16; scaling must match (÷32768 vs ÷32767) | Compare stage_00 tensors bit-for-bit |
| 2 | Padding/truncation | **NOT YET DEFINED** | `preprocessing.py` empty | Zero-pad side (left/right/center) could differ | Needs spec first |
| 3 | Window function | **NOT YET DEFINED** | Expected: Hann for STFT | Periodic vs symmetric Hann convention | Compare window coefficients exactly |
| 4 | FFT | **NOT YET DEFINED** | Expected: `torch.stft` or `MelSpectrogram` | FFT normalization, `onesided`, complex handling | Compare frame-by-frame |
| 5 | Mel filterbank | **NOT YET DEFINED** | Expected: torchaudio Mel filterbank | HTK vs Slaney scale, filter count, freq range, triangle norm | Compare filterbank matrix |
| 6 | Log compression | **NOT YET DEFINED** | Expected: `log(x + eps)` | epsilon value, log base (ln vs log10), clamping | Compare post-log values |
| 7 | DCT (MFCC) | **NOT YET DEFINED** | If MFCC: DCT-II | `ortho` normalization, num coefficients | Compare DCT output |
| 8 | Feature normalization | **NOT YET DEFINED** | Per-channel, per-utterance, or global | Statistics must be frozen identically | Publish exact mean/std |
| 9 | Tensor layout | **NOT YET DEFINED** | PyTorch: `[B, C, F, T]` | MCU may expect `[T, F]` or `[F, T]` | Document exact layout |
| 10 | Quantization scale/zero-point | **NOT YET DEFINED** | If INT8 model | Must embed identically in MCU model | Compare quantized I/O |

> **All risks are ANTICIPATED, not verified, because no preprocessing code exists.**

---

## 8. Export / Conversion Pipeline

> **NO EXPORT OR CONVERSION PIPELINE EXISTS.**

| Format | Status |
|--------|--------|
| ONNX export | **NOT IMPLEMENTED** |
| TFLite export | **NOT IMPLEMENTED** — `src/deployment/` is empty |
| TorchScript | **NOT IMPLEMENTED** |
| INT8 quantization | **NOT IMPLEMENTED** — `src/quantization/` is empty |
| FP16 | **NOT IMPLEMENTED** |
| C array header | **NOT IMPLEMENTED** |

---

## 9. Code2Edge Reference Candidate

> **CANNOT DEFINE A REFERENCE CANDIDATE.**
>
> Requirements for a Code2Edge reference:
> 1. A trained model checkpoint — **MISSING**
> 2. A defined preprocessing pipeline — **MISSING**
> 3. An inference entry point — **MISSING**
> 4. Reproducible evaluation results — **MISSING**

### 9.1 What Currently Exists and Can Be Reused

| Asset | File | Reusability |
|-------|------|-------------|
| Dataset loader (raw waveforms) | `src/data/dataset.py` | Usable as-is |
| 12-class label definition | `src/data/analyze_12class_protocol.py` L31–42 | `TARGET_WORDS` frozen |
| Silence windowing parameters | Same file, L44–48 | `SR=16000, HOP=8000, WIN=16000` |
| Split logic | Same file + `explore_dataset.py` | Official split files used correctly |
| Dataset on disk | `datasets/SpeechCommands/speech_commands_v0.02/` | Full V2 present locally |
| Python dependencies | `requirements.txt` | Frozen dependency list |

### 9.2 What Must Be Built Before Code2Edge Can Proceed

1. **Preprocessing pipeline** (MFCC or Log-Mel with all parameters frozen)
2. **DS-CNN model definition** (PyTorch `nn.Module`)
3. **Training script** with reproducible configuration
4. **Trained FP32 baseline checkpoint** with documented accuracy
5. **Inference script** (WAV → prediction)
6. **Evaluation script** (accuracy on test set)
7. **Export pipeline** (PyTorch → ONNX → TFLite or similar)

---

## 10. Reference Tensor Dump Design

> **Design only — NOT to be implemented until §9.2 prerequisites are met.**

Stages for a future `dump_reference.py`:

| Stage ID | Stage Name | Source Function (TBD) | Input | Output | Expected Shape | Dtype | Parity Value |
|----------|-----------|----------------------|-------|--------|----------------|-------|-------------|
| `stage_00` | `audio_load` | `torchaudio.load()` / `SPEECHCOMMANDS.__getitem__()` | WAV path | Raw waveform | `[1, N]` | `float32` | PCM→float conversion |
| `stage_01` | `pad_or_truncate` | TBD | `[1, N]` | Fixed waveform | `[1, 16000]` | `float32` | Pad/truncate logic |
| `stage_02` | `pre_emphasis` | TBD (if used) | `[1, 16000]` | Pre-emph waveform | `[1, 16000]` | `float32` | Coefficient parity |
| `stage_03` | `stft_or_spectrogram` | TBD | Waveform | Spectrogram | TBD | `float32` | FFT convention |
| `stage_04` | `mel_filterbank` | TBD | Spectrogram | Mel spectrogram | TBD | `float32` | Scale/filter/range |
| `stage_05` | `log_compression` | TBD | Mel spec | Log-mel spec | TBD | `float32` | Log base/epsilon |
| `stage_06` | `dct_mfcc` | TBD (if MFCC) | Log-mel | MFCC | TBD | `float32` | DCT normalization |
| `stage_07` | `feature_norm` | TBD | Raw features | Normalized | TBD | `float32` | Mean/std values |
| `stage_08` | `model_input` | TBD | Norm features | Input tensor | TBD (e.g. `[1,1,49,10]`) | `float32` | Tensor layout |
| `stage_09` | `model_output` | `model.forward()` | Input tensor | Logits | `[1, 12]` | `float32` | Model inference |
| `stage_10` | `prediction` | `torch.argmax()` | Logits | Class index | `[1]` | `int64` | Final prediction |

**Recommended format:** NumPy `.npz` files, one per test sample.

---

## 11. Host Parity Design

> **Design only — NOT to be implemented until §9.2 prerequisites are met.**

### 11.1 Differential Test Structure

```
For each test WAV file:
  1. Run Python reference pipeline → save all intermediate tensors
  2. Run C++ MCU-equivalent pipeline → save all intermediate tensors
  3. Compare stage-by-stage:
      For stage_i in [stage_00 .. stage_10]:
          max_abs_diff(python[stage_i], cpp[stage_i])
          mean_abs_diff(python[stage_i], cpp[stage_i])
          RMSE(python[stage_i], cpp[stage_i])
          cosine_similarity (for high-dim features)
          if max_abs_diff > threshold → FIRST DIVERGENT STAGE
```

### 11.2 Metrics Per Stage

| Stage | Metrics | Threshold |
|-------|---------|-----------|
| `stage_00` audio_load | `max_abs_diff` | **REQUIRES EXPERIMENT** |
| `stage_01` pad/truncate | `max_abs_diff` | Should be 0.0 (deterministic) |
| `stage_03` STFT | `max_abs_diff`, `RMSE` | **REQUIRES EXPERIMENT** |
| `stage_04` Mel filterbank | `max_abs_diff`, `cosine_sim` | **REQUIRES EXPERIMENT** |
| `stage_05` Log compression | `max_abs_diff` | **REQUIRES EXPERIMENT** |
| `stage_08` model_input | `max_abs_diff`, `RMSE`, `cosine_sim` | **REQUIRES EXPERIMENT** |
| `stage_09` model_output | `max_abs_diff`, `cosine_sim` | **REQUIRES EXPERIMENT** |
| `stage_10` prediction | `prediction_agreement` | 100% is the goal |

### 11.3 Failure Report Structure

```
PARITY TEST FAILURE REPORT

TEST FILE:        yes/abc123_nohash_0.wav

FIRST DIVERGENT STAGE:  stage_04_mel_filterbank

METRICS AT DIVERGENT STAGE:
  max_abs_diff:      0.0234
  mean_abs_diff:     0.0089
  RMSE:              0.0112
  cosine_similarity: 0.9987

PREVIOUS STAGE (stage_03_stft):
  max_abs_diff:      0.0001   ← PASS

DOWNSTREAM STAGES:
  stage_05 .. stage_10:  FAIL (likely downstream consequence)

ROOT CAUSE HYPOTHESIS:
  Check Mel scale formula, filter count, triangle normalization.
```

### 11.4 What Bob Should Receive on Failure

1. Exact stage name/index of first divergence
2. Numerical metrics at that stage
3. Python reference tensor
4. C++ output tensor
5. Element-wise difference tensor
6. All upstream stages confirmed PASS
7. List of parameters that could cause divergence

---

## 12. Hardware Readiness Analysis

> **TARGET MCU RESOLUTION:** The target is confirmed as the **STM32U585 MCU subsystem** on the **Arduino UNO Q** (heterogeneous dual-core architecture pairing Qualcomm Dragonwing QRB2210 MPU with STM32U585 Cortex-M33 MCU). The Arduino UNO R4 (Renesas RA4M1) is **NOT** the target.

### 12.1 STM32U585 Key Specifications

| Property | Value |
|----------|-------|
| Core | Arm Cortex-M33 |
| Max clock | 160 MHz |
| Flash | 2 MB |
| SRAM | 786 KB |
| DSP instructions | Yes |
| FPU | Single-precision (FP32) |

### 12.2 Requirements — Cannot Be Assessed

| Requirement | Status | Reason |
|-------------|--------|--------|
| Model Flash footprint | **TODO** | No model exists |
| Tensor arena size | **TODO** | No model exists |
| SRAM requirements | **TODO** | Depends on model + preprocessing |
| Operator support (TFLM) | **TODO** | DS-CNN ops typically supported |
| Runtime requirements | **TODO** | TFLite Micro or CMSIS-NN |
| Preprocessing buffer | **TODO** | FFT scratch ~2–8 KB typical |
| Model I/O memory | **TODO** | Depends on features |
| Inference latency | **TODO** | Depends on model + clock |

### 12.3 Feasibility (Typical DS-CNN KWS Reference)

MLPerf Tiny KWS benchmark: ~26K params → ~100 KB Flash (INT8), ~30 KB arena, ~50ms at 80 MHz Cortex-M4.

STM32U585 (2 MB Flash, 786 KB SRAM) should be **more than sufficient** — but unconfirmed until model exists.

---

## 13. Code2Edge MCP Tool Requirements

### 13.1 Minimum Viable MCP Tools

| Tool | Purpose | Inputs | Outputs | Deterministic? | Prerequisite |
|------|---------|--------|---------|----------------|--------------|
| `profile_model()` | Model size, params, FLOPs, memory | Checkpoint path, input shape | JSON metrics | Yes | **Needs model** |
| `inspect_pipeline()` | Trace preprocessing, dump intermediates | WAV path, config | Stage list with shapes/dtypes | Yes | **Needs preprocessing** |
| `run_host_parity()` | Compare Python ref vs C++ stage-by-stage | Ref dump, C++ output | Per-stage metrics | Yes | **Needs both impl** |
| `inspect_target()` | Query MCU specs | Board ID | JSON constraints | Yes | Board DB needed |
| `generate_edge_code()` | Generate C/C++ inference | Model, config, board | C/C++ source | No (AI/Bob) | **Needs model+pipeline** |

### 13.2 Design Principle

- `profile_model`, `inspect_pipeline`, `run_host_parity`, `inspect_target` → **Deterministic** (measurable facts)
- `generate_edge_code` → **AI-driven** (Bob orchestrates)

### 13.3 Current Status

**None of these tools can be built yet** — missing model and preprocessing pipeline.

---

## 14. Risks / Gaps

| # | Risk / Unknown | Evidence | Severity | How to Resolve |
|---|---------------|----------|----------|----------------|
| 1 | **No model exists** | `src/models/` empty; no checkpoints found | 🔴 CRITICAL | Implement DS-CNN, train baseline |
| 2 | **No preprocessing exists** | `preprocessing.py` is 0 bytes | 🔴 CRITICAL | Implement MFCC/Log-Mel extraction |
| 3 | **No inference code exists** | No inference script anywhere | 🔴 CRITICAL | Implement inference entry point |
| 4 | **No training code exists** | `src/training/` empty | 🔴 CRITICAL | Implement training loop |
| 5 | **No evaluation results** | `results/` and `logs/` only `.gitkeep` | 🔴 CRITICAL | Train and evaluate baseline |
| 6 | **12-class dataset not finalized** | Experiment 002 §12: 6 open questions | 🟡 HIGH | Finalize unknown/silence sampling |
| 7 | **No export pipeline** | `src/deployment/` empty | 🟡 HIGH | Implement ONNX/TFLite export |
| 8 | **No config files populated** | All 6 YAML configs are 0 bytes | 🟡 HIGH | Define and freeze hyperparameters |
| 9 | **No tests** | `tests/` empty | 🟡 MEDIUM | Add preprocessing/model unit tests |
| 10 | **CUDA-only deps** | `requirements.txt`: `torch==2.11.0+cu128` | 🟡 MEDIUM | Add CPU fallback |
| 11 | **Target MCU ambiguity** | "STM32U585 on Arduino UNO Q" contradictory | 🟡 MEDIUM | Confirm exact board |
| 12 | **No deterministic seed** | Experiment 002 §10 requires it but none set | 🟡 MEDIUM | Implement seeded sampling |
| 13 | **No augmentation defined** | `transforms.py` is 0 bytes | 🟢 LOW | Training-only; define for reproducibility |
| 14 | **environment.yml empty** | 0 bytes | 🟢 LOW | `requirements.txt` covers deps |
| 15 | **Stale README references** | README lists nonexistent files | 🟢 LOW | Update README |

---

## 15. Final Executive Summary

### A. VERIFIED REFERENCE PIPELINE

**NO VERIFIED PIPELINE EXISTS.**

The only verified code path:

```
WAV file on disk
  ↓
torchaudio.datasets.SPEECHCOMMANDS (via src/data/dataset.py)
  ↓
Dict{"waveform": Tensor[1, N], "sample_rate": 16000, "label": str, ...}
```

Raw audio loader only. **No feature extraction, no model, no inference.**

### B. RECOMMENDED CODE2EDGE REFERENCE

**CANNOT RECOMMEND.** No model or checkpoint exists.

Future recommendation (when available):
- **Model:** FP32 DS-CNN baseline (to be created in `src/models/`)
- **Checkpoint:** First validated FP32 checkpoint achieving >90% on 12-class test
- **Preprocessing:** To be implemented in `src/data/preprocessing.py`

### C. REQUIRED REFERENCE ARTIFACTS (Before Code2Edge Can Begin)

| # | Artifact | Status |
|---|----------|--------|
| 1 | Finalized 12-class Dataset class (with unknown/silence) | ❌ Missing |
| 2 | Preprocessing pipeline with frozen MFCC/Log-Mel params | ❌ Missing |
| 3 | DS-CNN model definition (`nn.Module`) | ❌ Missing |
| 4 | Trained FP32 checkpoint (`.pt` / `.pth`) | ❌ Missing |
| 5 | Baseline evaluation results (accuracy, loss, confusion matrix) | ❌ Missing |
| 6 | Inference script (WAV → prediction) | ❌ Missing |
| 7 | Export script (PyTorch → ONNX → TFLite) | ❌ Missing |
| 8 | Frozen preprocessing constants (mean, std, filterbank) | ❌ Missing |
| 9 | Reference tensor dump for ≥10 test samples | ❌ Missing |

### D. BIGGEST TECHNICAL RISKS (Ranked)

1. 🔴 **No model, no preprocessing, no inference pipeline** — Code2Edge has nothing to trace
2. 🔴 **12-class dataset protocol not finalized** — 6 open questions remain
3. 🟡 **No frozen preprocessing constants** — MFCC params critical for MCU parity
4. 🟡 **Target hardware ambiguity** — STM32U585 vs Arduino UNO contradictory
5. 🟡 **No export/conversion pipeline** — no path to edge-deployable format

### E. NEXT 5 ACTIONS (Audit-Based, No Code Modification)

1. **Confirm target hardware**: STM32U585 (B-U585I-IOT02A) or Renesas RA4M1 (Arduino UNO R4)?

2. **Strategic decision**: Build missing pipeline in tinyml-compression-benchmark, or build fresh in Code2Edge using the benchmark's dataset protocol as spec?

3. **Freeze 12-class protocol**: Resolve 6 open questions from Experiment 002 §12 (unknown/silence counts, test composition, balance strategy).

4. **Define preprocessing parameters**: Document exact MFCC/Log-Mel parameters before writing code (window, hop, FFT size, Mel filters, freq range, num coefficients, normalization).

5. **Create minimal DS-CNN reference**: Implement model, train, evaluate, freeze checkpoint — prerequisite for everything Code2Edge needs.

---

*End of audit. All findings based on direct inspection of `D:\tinyml-compression-benchmark` as of 2026-09-26. No source files were modified.*
