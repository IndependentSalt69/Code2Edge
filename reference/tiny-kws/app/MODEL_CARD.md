---
license: mit
language: en
tags:
  - audio-classification
  - keyword-spotting
  - tinyml
  - pytorch
datasets:
  - speech_commands
metrics:
  - accuracy
  - f1
---

# tiny-kws — DS-CNN keyword spotter (12-class Speech Commands v2)

A 119,372-parameter (~0.48 MB fp32) depthwise-separable CNN for spoken
command recognition, trained from scratch in PyTorch. Input: 1-second 16 kHz
audio → 64×101 log-mel spectrogram. Output: one of 12 classes — the keywords
*yes, no, up, down, left, right, on, off, stop, go*, plus *unknown* and
*silence*.

- **Architecture**: DS-CNN (Zhang et al. 2017, arXiv:1711.07128): 10×4
  conv stem (stride 2) → 4 depthwise-separable blocks (160 ch, one stride-2)
  → global average pooling → dropout 0.2 → linear.
- **Dataset**: Google Speech Commands v0.02 (Warden 2018, arXiv:1804.03209,
  CC-BY-4.0): 105,829 one-second utterances, 35 words. Official
  validation/testing lists (speaker-disjoint); "unknown" = seeded 10% sample
  of the 25 non-keyword words; "silence" = background-noise crops.
- **Training**: 30 epochs on a free Colab T4 (GPU), AdamW lr 3e-3
  (cosine-annealed), batch 128, label smoothing 0.1, fp32. Best validation
  accuracy 96.15% at epoch 30. Augmentation: ±100 ms time-shift +
  background-noise mixing (p=0.8, vol U(0,0.1)).
- **Features**: log-mel, 64 mels, 25 ms window / 10 ms hop, normalized by
  train-set global mean/std (stored inside the checkpoint).

## Evaluation — official Speech Commands v2 test set (4,890 clips)

<!-- METRICS_TABLE: produced by evaluate.py, never hand-written -->
| metric | value |
|---|---|
| accuracy | 96.65% |
| macro-F1 | 96.64% |
| CPU latency (batch=1, 1 thread, Apple M2) | 1.90 ms mean / 2.08 ms p95 |

Per-class F1 ranges from 0.921 ("unknown", the hardest class) to 0.998
("silence"); all 10 keywords score ≥0.94. Full per-class table and the
confusion matrix: see `metrics.json` and `confusion_matrix.png` in this repo.
Evaluating this checkpoint on the Colab T4 and on an Apple M2 produced
bit-for-bit identical metrics (reproducible across devices).

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# model.py + common.py from https://github.com/priyadeepjaiswal9c/tiny-kws
from model import DSCNN
from common import LogMel, normalize

ckpt = torch.load(hf_hub_download("priyadeepjaiswal9c/tiny-kws", "best.pt"),
                  map_location="cpu", weights_only=True)
model = DSCNN(**ckpt["model_config"]); model.load_state_dict(ckpt["model_state"]); model.eval()

wav = torch.zeros(16000)          # your 1 s, 16 kHz, mono float32 waveform
feats = normalize(LogMel()(wav), ckpt["stats"])
probs = model(feats).softmax(1)[0]
print(dict(zip(ckpt["labels"], probs.tolist())))
```

## Intended use & limitations

Demo/educational model for isolated 1-second command words in quiet-to-mild
noise. Not a streaming/wake-word system (no sliding-window detection), not
robust to far-field audio or heavy noise, English only, and trained on
crowdsourced speech that skews toward certain accents — expect degraded
accuracy outside that distribution.

Live demo: https://huggingface.co/spaces/priyadeepjaiswal9c/tiny-kws · Code: https://github.com/priyadeepjaiswal9c/tiny-kws
