"""Build the 12-class Speech Commands benchmark caches from the raw dataset.

Inputs (already downloaded + extracted):
  data/SpeechCommands/speech_commands_v0.02/   main archive (105,829 WAVs, 35 words)
  data/test_set/                               official curated test set (4,890 WAVs)

The 12-class task: 10 keywords + "unknown" + "silence".
  * Splits for train/val follow the OFFICIAL validation_list.txt /
    testing_list.txt — files in either list are NEVER used for training.
    (The lists hash on speaker ID, so no speaker appears in two splits.)
  * "unknown" examples are a seeded random sample of the other 25 words,
    sized to ~10% of each split — the convention from Warden's original
    TensorFlow reference code, which also matches the official test set's
    composition (~8.3% unknown, ~8.3% silence).
  * "silence" examples are seeded random 1-second crops of the six
    _background_noise_ recordings, at a random gain in [0, 1).
  * TEST is the official speech_commands_test_set_v0.02 archive, untouched —
    so accuracy here is directly comparable to published 12-class numbers.

Outputs in data/processed/:
  train_bank.npy    int16 (N, 16000) memmap of training waveforms
  train_labels.npy  int64 (N,)
  noise.pt          list of float32 noise tracks (for train-time mixing)
  val_feats.pt      float32 (N, 1, 64, 101) log-mels (unnormalized)
  val_labels.pt     int64
  test_feats.pt / test_labels.pt   same, from the official test set
  stats.json        global mean/std of train log-mels (for normalization)
  manifest.json     exact per-class counts + provenance
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from common import (CLIP_SAMPLES, KEYWORDS, LABEL_TO_IDX, LOG_EPS, LogMel,
                    SAMPLE_RATE, UNKNOWN_WORDS)

SEED = 42
UNKNOWN_PCT = 0.10  # fraction of each split that is "unknown"
SILENCE_PCT = 0.10  # fraction of each split that is "silence"
STATS_SAMPLE = 4096


def read_wav_fixed(path: Path) -> np.ndarray:
    """Read a WAV as float32 mono, padded/trimmed to exactly 1 s @ 16 kHz."""
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SAMPLE_RATE, f"{path}: unexpected sample rate {sr}"
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < CLIP_SAMPLES:
        wav = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
    return wav[:CLIP_SAMPLES]


def list_split_files(root: Path):
    """Partition the 35-word archive by the official lists."""
    val_set = set((root / "validation_list.txt").read_text().split())
    test_set = set((root / "testing_list.txt").read_text().split())
    train, val = [], []
    words = KEYWORDS + UNKNOWN_WORDS
    for word in words:
        for f in sorted((root / word).glob("*.wav")):
            rel = f"{word}/{f.name}"
            if rel in test_set:
                continue  # official test files: never train on these
            (val if rel in val_set else train).append((word, f))
    return train, val


def build_split(files, noise_tracks, rng):
    """Turn raw (word, path) pairs into a 12-class example list.

    Returns list of (label_idx, source) where source is a Path for real
    clips or a precomputed np.ndarray for silence crops.
    """
    kw = [(LABEL_TO_IDX[w], p) for w, p in files if w in KEYWORDS]
    unk_pool = [p for w, p in files if w not in KEYWORDS]
    total = round(len(kw) / (1 - UNKNOWN_PCT - SILENCE_PCT))
    n_unk = min(round(total * UNKNOWN_PCT), len(unk_pool))
    n_sil = round(total * SILENCE_PCT)
    unk = [(LABEL_TO_IDX["unknown"], p) for p in rng.sample(unk_pool, n_unk)]
    sil = []
    for _ in range(n_sil):
        track = noise_tracks[rng.randrange(len(noise_tracks))]
        start = rng.randrange(len(track) - CLIP_SAMPLES)
        gain = rng.random()  # 0 = true silence, up to full-volume noise
        sil.append((LABEL_TO_IDX["silence"],
                    (track[start:start + CLIP_SAMPLES] * gain).copy()))
    examples = kw + unk + sil
    rng.shuffle(examples)
    return examples


def materialize_wav(source) -> np.ndarray:
    return source if isinstance(source, np.ndarray) else read_wav_fixed(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    root = args.data_dir / "SpeechCommands" / "speech_commands_v0.02"
    test_root = args.data_dir / "test_set"
    out = args.data_dir / "processed"
    out.mkdir(parents=True, exist_ok=True)

    # ---- background noise tracks -----------------------------------------
    noise_tracks = []
    for f in sorted((root / "_background_noise_").glob("*.wav")):
        wav, sr = sf.read(f, dtype="float32")
        assert sr == SAMPLE_RATE
        noise_tracks.append(wav)
    print(f"noise tracks: {len(noise_tracks)} "
          f"({sum(len(t) for t in noise_tracks) / SAMPLE_RATE:.0f} s total)")
    torch.save([torch.from_numpy(t) for t in noise_tracks], out / "noise.pt")

    # ---- official split partition ----------------------------------------
    train_files, val_files = list_split_files(root)
    print(f"raw 35-word files -> train {len(train_files)}, val {len(val_files)}")

    rng = random.Random(SEED)
    train_ex = build_split(train_files, noise_tracks, rng)
    val_ex = build_split(val_files, noise_tracks, random.Random(SEED + 1))

    # ---- train: write int16 waveform bank --------------------------------
    bank = np.lib.format.open_memmap(
        out / "train_bank.npy", mode="w+",
        dtype=np.int16, shape=(len(train_ex), CLIP_SAMPLES))
    train_labels = np.empty(len(train_ex), dtype=np.int64)
    for i, (lab, src) in enumerate(tqdm(train_ex, desc="train bank")):
        wav = materialize_wav(src)
        bank[i] = np.clip(wav * 32768.0, -32768, 32767).astype(np.int16)
        train_labels[i] = lab
    bank.flush()
    np.save(out / "train_labels.npy", train_labels)

    # ---- val + test: precompute log-mel features -------------------------
    logmel = LogMel()  # CPU

    def featurize(examples, desc):
        feats = torch.empty(len(examples), 1, 64, CLIP_SAMPLES // 160 + 1)
        labels = torch.empty(len(examples), dtype=torch.int64)
        for i, (lab, src) in enumerate(tqdm(examples, desc=desc)):
            wav = torch.from_numpy(np.ascontiguousarray(materialize_wav(src)))
            feats[i] = logmel(wav)[0]
            labels[i] = lab
        return feats, labels

    val_feats, val_labels = featurize(val_ex, "val features")
    torch.save(val_feats, out / "val_feats.pt")
    torch.save(val_labels, out / "val_labels.pt")

    # official curated test set: directory name == label
    test_ex = []
    for d in sorted(test_root.iterdir()):
        if not d.is_dir():
            continue
        label = {"_silence_": "silence", "_unknown_": "unknown"}.get(d.name, d.name)
        assert label in LABEL_TO_IDX, f"unexpected test dir {d.name}"
        test_ex += [(LABEL_TO_IDX[label], f) for f in sorted(d.glob("*.wav"))]
    test_feats, test_labels = featurize(test_ex, "test features")
    torch.save(test_feats, out / "test_feats.pt")
    torch.save(test_labels, out / "test_labels.pt")

    # ---- normalization stats from a seeded sample of train ---------------
    srng = np.random.default_rng(SEED)
    idx = np.sort(srng.choice(len(train_ex), replace=False,
                              size=min(STATS_SAMPLE, len(train_ex))))
    chunks = []
    for i in range(0, len(idx), 512):  # chunked to keep peak RAM low
        sample = torch.from_numpy(
            np.asarray(bank[idx[i:i + 512]], dtype=np.float32) / 32768.0)
        chunks.append(logmel(sample))
    sample_feats = torch.cat(chunks)
    stats = {"mean": float(sample_feats.mean()), "std": float(sample_feats.std()),
             "log_eps": LOG_EPS, "stats_sample": int(len(idx))}
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"feature stats: mean={stats['mean']:.4f} std={stats['std']:.4f}")

    # ---- manifest ---------------------------------------------------------
    def counts(labels):
        c = Counter(int(l) for l in labels)
        from common import LABELS
        return {LABELS[k]: v for k, v in sorted(c.items())}

    manifest = {
        "dataset": "Google Speech Commands v0.02 (Warden 2018, arXiv:1804.03209)",
        "splits": {
            "train": {"total": len(train_ex), "per_class": counts(train_labels)},
            "val": {"total": len(val_ex), "per_class": counts(val_labels.numpy())},
            "test": {"total": len(test_ex), "per_class": counts(test_labels.numpy()),
                     "source": "official speech_commands_test_set_v0.02"},
        },
        "unknown_pct": UNKNOWN_PCT, "silence_pct": SILENCE_PCT, "seed": SEED,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
