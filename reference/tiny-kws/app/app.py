"""tiny-kws live demo — Gradio app for Hugging Face Spaces (or local).

The pre-processing here mirrors training EXACTLY (same LogMel module, same
normalization stats pulled from the checkpoint), plus two steps that only
matter for live microphones:
  1. Browser mics record 48 kHz and often stereo -> average to mono and
     resample to 16 kHz. Without this, features are computed on the wrong
     timescale and accuracy silently collapses.
  2. Recordings are never exactly 1 s -> select the highest-energy 1-second
     window so we crop to the spoken word, not to leading silence.
"""

import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.functional as AF

# import model/feature code: same dir (Space layout) or ../src (repo layout)
_here = Path(__file__).resolve().parent
for cand in (_here, _here.parent / "src"):
    if (cand / "model.py").exists():
        sys.path.insert(0, str(cand))
        break
from common import CLIP_SAMPLES, LogMel, SAMPLE_RATE, normalize  # noqa: E402
from model import DSCNN  # noqa: E402

MODEL_REPO = os.environ.get("TINYKWS_MODEL_REPO", "")  # e.g. "user/tiny-kws"


def find_checkpoint() -> Path:
    local = [_here / "best.pt", _here.parent / "checkpoints" / "best.pt"]
    for p in local:
        if p.exists():
            return p
    if MODEL_REPO:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(MODEL_REPO, "best.pt"))
    raise FileNotFoundError(
        "no best.pt found locally and TINYKWS_MODEL_REPO is not set")


CKPT = torch.load(find_checkpoint(), map_location="cpu", weights_only=True)
LABELS = CKPT["labels"]
STATS = CKPT["stats"]
MODEL = DSCNN(**CKPT["model_config"])
MODEL.load_state_dict(CKPT["model_state"])
MODEL.eval()
LOGMEL = LogMel()
torch.set_num_threads(2)  # Spaces free tier has 2 vCPUs


def to_mono_16k(sr: int, wav: np.ndarray) -> torch.Tensor:
    """Browser audio -> float32 mono 16 kHz tensor."""
    wav = np.asarray(wav)
    # scale BEFORE channel-averaging: mean() promotes int16 to float64 and
    # would silently skip the integer scaling for stereo input
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    else:
        wav = wav.astype(np.float32)
    if wav.ndim == 2:                       # (n, channels) -> mono
        wav = wav.mean(axis=1)
    t = torch.from_numpy(wav)
    if sr != SAMPLE_RATE:                   # polyphase resample (anti-aliased)
        t = AF.resample(t, orig_freq=sr, new_freq=SAMPLE_RATE)
    return t


def best_1s_window(wav: torch.Tensor) -> torch.Tensor:
    """Pad to >=1 s, then return the highest-energy 1-second slice."""
    if len(wav) <= CLIP_SAMPLES:
        return torch.nn.functional.pad(wav, (0, CLIP_SAMPLES - len(wav)))
    energy = (wav ** 2).cumsum(0)
    window = energy[CLIP_SAMPLES:] - energy[:-CLIP_SAMPLES]
    start = int(window.argmax())
    return wav[start:start + CLIP_SAMPLES]


def spectrogram_image(feats: torch.Tensor) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.imshow(feats[0, 0].numpy(), origin="lower", aspect="auto",
              cmap="magma", extent=[0, 1, 0, 64])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel band")
    ax.set_title("what the model sees: log-mel spectrogram (64 x 101)")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


@torch.no_grad()
def classify(audio):
    if audio is None:
        return {"(record or upload a clip first)": 1.0}, None
    sr, wav = audio
    wav = to_mono_16k(sr, wav)
    wav = best_1s_window(wav)
    feats = LOGMEL(wav)                      # (1, 1, 64, 101), unnormalized
    img = spectrogram_image(feats)
    probs = MODEL(normalize(feats, STATS)).softmax(dim=1)[0]
    return {l: float(p) for l, p in zip(LABELS, probs)}, img


examples = sorted(str(p) for p in (_here / "examples").glob("*.wav"))

demo = gr.Interface(
    fn=classify,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy",
                    label="say one of: yes / no / up / down / left / right / "
                          "on / off / stop / go"),
    outputs=[gr.Label(num_top_classes=3, label="prediction"),
             gr.Image(label="log-mel spectrogram")],
    examples=examples if examples else None,
    title="tiny-kws — edge keyword spotting",
    description=(
        "A 119K-parameter (~0.5 MB) depthwise-separable CNN trained on "
        "Google Speech Commands v2, running on free CPU. Record ~1 second "
        "of audio with one of the 10 keywords — or anything else to see "
        "'unknown'. Mic audio is resampled 48 kHz -> 16 kHz and cropped to "
        "its highest-energy 1-second window before classification. "
        "[GitHub](https://github.com/priyadeepjaiswal9c/tiny-kws)"
    ),
)

if __name__ == "__main__":
    demo.launch()
