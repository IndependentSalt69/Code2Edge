"""Shared constants and the audio feature frontend for tiny-kws.

Everything that must be identical between training, evaluation and the live
demo lives here: the label order, the feature parameters, and the log-mel
transform itself. If train and inference disagree on any of these, accuracy
silently collapses, so they are defined exactly once.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

# ---------------------------------------------------------------------------
# Task definition: the standard 12-class Speech Commands benchmark
# ---------------------------------------------------------------------------
KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = ["silence", "unknown"] + KEYWORDS  # index 0 = silence, 1 = unknown
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

# The 25 non-keyword words in Speech Commands v2; clips from these become "unknown".
UNKNOWN_WORDS = [
    "backward", "bed", "bird", "cat", "dog", "eight", "five", "follow",
    "forward", "four", "happy", "house", "learn", "marvin", "nine", "one",
    "seven", "sheila", "six", "three", "tree", "two",
    "visual", "wow", "zero",
]

# ---------------------------------------------------------------------------
# Audio / feature parameters
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000          # Hz; dataset native rate
CLIP_SAMPLES = 16_000         # exactly 1 second per clip
N_FFT = 400                   # 25 ms analysis window
HOP_LENGTH = 160              # 10 ms hop -> 101 frames per 1 s clip
N_MELS = 64
F_MIN, F_MAX = 20.0, 7600.0   # stay below Nyquist (8 kHz)
N_FRAMES = CLIP_SAMPLES // HOP_LENGTH + 1  # 101 (torch STFT uses center padding)
LOG_EPS = 1e-6


class LogMel(nn.Module):
    """Waveform (B, 16000) -> log-mel spectrogram (B, 1, 64, 101).

    Kept as an nn.Module so it can be moved to CUDA for fast feature
    extraction on Colab; on Apple Silicon it should stay on CPU because
    torchaudio's spectrogram ops have gaps on MPS.
    """

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        feats = torch.log(self.mel(wav) + LOG_EPS)
        return feats.unsqueeze(1)  # add channel dim


def normalize(feats: torch.Tensor, stats: dict) -> torch.Tensor:
    """Apply train-set global mean/std normalization."""
    return (feats - stats["mean"]) / stats["std"]


def load_stats(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
