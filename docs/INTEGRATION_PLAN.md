# Code2Edge Integration Plan — tiny-kws Reference Workload

**Reference workload:** `tiny-kws` ([priyadeepjaiswal9c/tiny-kws](https://github.com/priyadeepjaiswal9c/tiny-kws))  
**Target MCU:** STM32U585 on Arduino UNO Q (Clarification: Arduino UNO R4 is NOT the target)  
**Status:** In Progress (Phase 1 vendoring complete)

---

## 1. Upstream Pin — Exact Commit

The reference workload is frozen to a single, reproducible upstream state:

```text
SHA:     c097b35ae4b9cd585a544c74db16892ce674b186
Date:    2026-08-13T08:28:34Z
Message: docs: standardize contributor attribution
Repo:    https://github.com/priyadeepjaiswal9c/tiny-kws
```

Recorded in `reference/tiny-kws/UPSTREAM.md` and in `.pin`.

---

## 2. Vendored Files from Reference Repository

| Upstream path | Purpose for Code2Edge | Required? |
|---|---|---|
| `src/common.py` | Single source of truth: LABELS, feature params, `LogMel`, `normalize()` | **Yes — critical** |
| `src/model.py` | DSCNN architecture; used for checkpoint loading and host parity | **Yes — critical** |
| `src/evaluate.py` | Official test-set evaluation; parity gate baseline | **Yes** |
| `src/dataset.py` | Data loading logic; needed for parity dataset enumeration | **Yes** |
| `src/prepare_data.py` | Documents exact split/cache construction | Yes (reference) |
| `src/train.py` | Documents exact training config embedded in checkpoint | Yes (reference) |
| `requirements.txt` | Upstream dependency versions | **Yes** |
| `LICENSE` | MIT license — preserved verbatim | **Yes — legal** |
| `README.md` | Attribution and feature/architecture documentation | **Yes** |
| `assets/metrics.json` | Frozen gold-standard evaluation numbers for parity tolerance | **Yes** |
| `app/app.py` | Real-world live inference path; resampling & window logic | Yes (reference) |
| `app/MODEL_CARD.md` | Authoritative architecture description with layer config | Yes (reference) |
| `app/examples/*.wav` | Parity test fixtures | **Yes** |

---

## 3. Placement Inside Code2Edge

```text
Code2Edge/
├── reference/
│   └── tiny-kws/               ← vendored upstream snapshot
│       ├── UPSTREAM.md         ← pinned SHA, date, source URL, file hashes
│       ├── .pin                ← machine-readable pin
│       ├── LICENSE             ← verbatim upstream MIT license
│       ├── README.md           ← verbatim upstream README
│       ├── requirements.txt    ← verbatim upstream deps
│       ├── assets/
│       │   └── metrics.json    ← gold-standard numbers
│       ├── app/
│       │   ├── app.py
│       │   ├── MODEL_CARD.md
│       │   └── examples/       ← smoke-test WAV fixtures
│       └── src/
│           ├── common.py       ← IMMUTABLE: feature params, LogMel, LABELS
│           ├── model.py        ← IMMUTABLE: DSCNN architecture
│           ├── evaluate.py     ← IMMUTABLE: gold evaluation script
│           ├── dataset.py      ← IMMUTABLE
│           ├── prepare_data.py ← IMMUTABLE
│           └── train.py        ← IMMUTABLE
│
├── src/                        ← Code2Edge source (SEPARATE from reference)
│   ├── inference/              ← Code2Edge inference runner (wraps reference)
│   ├── parity/                 ← Stage-wise parity instrumentation
│   ├── export/                 ← Edge code generation pipeline
│   └── tools/                  ← Utilities
│
├── tests/
│   ├── parity/                 ← Parity gate test suite
│   └── fixtures/               ← Test fixtures
│
├── data/                       ← Gitignored local dataset mount point
└── checkpoints/                ← Gitignored pretrained checkpoints
```

---

## 4. Exact Python Inference Pipeline

From `common.py` and `model.py`:
- **Audio:** 16,000 Hz, 16,000 samples (1.0 sec), float32 mono.
- **MelSpectrogram:** N_FFT=400, HOP_LENGTH=160, N_MELS=64, F_MIN=20.0 Hz, F_MAX=7600.0 Hz, power=2.0. Output shape: `(1, 64, 101)`.
- **Log compression:** `log(mel + 1e-6)` -> `(1, 1, 64, 101)`.
- **Normalization:** `(feats - mean) / std` using global dataset statistics in `ckpt["stats"]`.
- **DSCNN Model:** Stem (1->160 ch, stride 2) -> 4 DS-Blocks (160 ch) -> AdaptiveAvgPool2d(1,1) -> Linear(160, 12).
- **Labels (12):** `0: silence`, `1: unknown`, `2: yes`, `3: no`, `4: up`, `5: down`, `6: left`, `7: right`, `8: on`, `9: off`, `10: stop`, `11: go`.

---

## 5. Differential Parity Stages

| # | Stage Name | Tensor Shape | Dtype | Parity Tolerance |
|---|---|---|---|---|
| S0 | Raw waveform | `(16000,)` | float32 | Exact |
| S1 | Mel power spectrogram | `(1, 64, 101)` | float32 | Relative error <= 1e-4 |
| S2 | Log-mel feature map | `(1, 1, 64, 101)` | float32 | Relative error <= 1e-4 |
| S3 | Normalized log-mel | `(1, 1, 64, 101)` | float32 | Relative error <= 1e-4 |
| S4a | Stem output | `(1, 160, 32, 51)` | float32 | Relative error <= 1e-3 |
| S4b | DS-Block 0 output | `(1, 160, 16, 26)` | float32 | Relative error <= 1e-3 |
| S4c | DS-Block 3 output | `(1, 160, 16, 26)` | float32 | Relative error <= 1e-3 |
| S4d | GAP embedding | `(1, 160)` | float32 | Relative error <= 1e-3 |
| S5 | Logits | `(1, 12)` | float32 | Absolute error <= 0.01; argmax must match |
| S6 | Predicted class index | scalar int | int32 | **Exact match required** |
