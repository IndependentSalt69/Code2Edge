"""PyTorch datasets over the precomputed caches written by prepare_data.py.

Two different cache formats, on purpose:

* TRAIN  -> raw int16 waveforms in one memory-mapped .npy file.
  Augmentation (random time-shift, background-noise mixing) must operate on
  the waveform BEFORE the spectrogram, so we cannot cache spectrograms here.
  A single memmapped file still avoids the cost of opening ~38k tiny WAVs
  every epoch — reads are sequential pages from one file.

* VAL / TEST -> precomputed log-mel tensors (.pt).
  No augmentation is ever applied to evaluation data, so the features are
  deterministic and can be computed exactly once.

The training Dataset returns *waveforms*; the log-mel transform is applied
batch-wise in the training loop (on CUDA when available, on CPU for MPS,
since torchaudio spectrogram ops are not reliable on MPS).
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from common import CLIP_SAMPLES

MAX_SHIFT = 1600          # +-100 ms at 16 kHz
NOISE_PROB = 0.8          # fraction of training samples that get noise mixed in
NOISE_MAX_VOL = 0.1       # noise amplitude scale, as in Warden's TF reference


class TrainWaveformDataset(Dataset):
    def __init__(self, processed_dir: Path, augment: bool = True):
        processed_dir = Path(processed_dir)
        # Keep only the PATH here and open the memmap lazily per process:
        # macOS DataLoader workers are spawned and pickle the dataset, and
        # pickling an np.memmap would materialize the whole bank in RAM.
        self.bank_path = processed_dir / "train_bank.npy"
        self._bank = None
        self.labels = torch.from_numpy(
            np.load(processed_dir / "train_labels.npy")
        ).long()
        # Background noise tracks (float32 1-D tensors of varying length)
        self.noise = [t.float() for t in torch.load(
            processed_dir / "noise.pt", weights_only=True)]
        self.augment = augment

    @property
    def bank(self):
        if self._bank is None:
            self._bank = np.load(self.bank_path, mmap_mode="r")
        return self._bank

    def __getstate__(self):
        return {**self.__dict__, "_bank": None}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wav = torch.from_numpy(
            np.asarray(self.bank[idx], dtype=np.float32) / 32768.0
        )
        if self.augment:
            wav = self._augment(wav)
        return wav, self.labels[idx]

    def _augment(self, wav: torch.Tensor) -> torch.Tensor:
        # 1) random time shift in [-100 ms, +100 ms], zero-padded
        shift = int(torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (1,)))
        if shift != 0:
            wav = torch.roll(wav, shift)
            if shift > 0:
                wav[:shift] = 0.0
            else:
                wav[shift:] = 0.0
        # 2) mix in a random crop of background noise at low volume
        if torch.rand(1).item() < NOISE_PROB:
            track = self.noise[int(torch.randint(len(self.noise), (1,)))]
            start = int(torch.randint(len(track) - CLIP_SAMPLES, (1,)))
            vol = torch.rand(1).item() * NOISE_MAX_VOL
            wav = wav + vol * track[start:start + CLIP_SAMPLES]
        return wav.clamp_(-1.0, 1.0)


class EvalFeaturesDataset(Dataset):
    """Precomputed, UNNORMALIZED log-mel features. Normalization (train-set
    mean/std) is applied here at read time so the stats live in one place."""

    def __init__(self, feats_path: Path, labels_path: Path, stats: dict):
        self.feats = torch.load(feats_path, weights_only=True)
        self.labels = torch.load(labels_path, weights_only=True).long()
        self.mean = float(stats["mean"])
        self.std = float(stats["std"])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.feats[idx] - self.mean) / self.std, self.labels[idx]


def load_manifest(processed_dir: Path) -> dict:
    with open(Path(processed_dir) / "manifest.json") as f:
        return json.load(f)
