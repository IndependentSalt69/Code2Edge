from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
KWS_SRC = ROOT / "reference" / "tiny-kws" / "src"

sys.path.insert(0, str(KWS_SRC))

from common import (  # noqa: E402
    CLIP_SAMPLES,
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    LABELS,
    LOG_EPS,
    LogMel,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    load_stats,
    normalize,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)

    return h.hexdigest()


def load_wav_fixed(path: Path) -> torch.Tensor:
    wav, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False,
    )

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {sr} Hz"
        )

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if len(wav) < CLIP_SAMPLES:
        wav = np.pad(
            wav,
            (0, CLIP_SAMPLES - len(wav)),
        )

    wav = wav[:CLIP_SAMPLES]

    return torch.from_numpy(
        np.ascontiguousarray(wav, dtype=np.float32)
    )


def tensor_metadata(x: torch.Tensor) -> dict:
    a = x.detach().cpu().numpy()

    result = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "finite": bool(np.isfinite(a).all()),
    }

    if np.iscomplexobj(a):
        result.update(
            {
                "real_min": float(np.real(a).min()),
                "real_max": float(np.real(a).max()),
                "imag_min": float(np.imag(a).min()),
                "imag_max": float(np.imag(a).max()),
            }
        )

    return result


class ReferenceTracer:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = {
            "stages": {},
            "artifacts": {},
        }

        self.logmel = LogMel()

        self.logmel.mel.spectrogram.register_forward_hook(
            self._capture_power
        )

        self.logmel.mel.mel_scale.register_forward_hook(
            self._capture_mel
        )

        self.logmel.register_forward_hook(
            self._capture_log
        )

    def _save_stage(self, name: str, x: torch.Tensor) -> None:
        array = x.detach().cpu().numpy()

        path = self.out_dir / f"{name}.npy"
        np.save(path, array)

        self.manifest["stages"][name] = {
            "file": path.name,
            **tensor_metadata(x),
        }

    def _capture_power(self, module, inputs, output):
        self._save_stage(
            "post_power_spectrum",
            output,
        )

    def _capture_mel(self, module, inputs, output):
        self._save_stage(
            "post_mel",
            output,
        )

    def _capture_log(self, module, inputs, output):
        # LogMel adds a channel dimension:
        # (B, 1, 64, 101)
        # Save the actual output of the reference module.
        self._save_stage(
            "post_log",
            output,
        )

    def capture(self, wav: torch.Tensor, stats: dict | None):
        self._save_stage(
            "post_input",
            wav,
        )

        with torch.no_grad():
            log_features = self.logmel(wav)

        if stats is not None:
            normalized = normalize(
                log_features,
                stats,
            )

            self._save_stage(
                "post_normalize",
                normalized,
            )

        return log_features

    def save_artifacts(self):
        window = self.logmel.mel.spectrogram.window.detach().cpu().numpy()
        mel_fb = self.logmel.mel.mel_scale.fb.detach().cpu().numpy()

        window_path = self.out_dir / "window.npy"
        mel_path = self.out_dir / "mel_filterbank.npy"

        np.save(window_path, window)
        np.save(mel_path, mel_fb)

        self.manifest["artifacts"]["window"] = {
            "file": window_path.name,
            "shape": list(window.shape),
            "dtype": str(window.dtype),
        }

        self.manifest["artifacts"]["mel_filterbank"] = {
            "file": mel_path.name,
            "shape": list(mel_fb.shape),
            "dtype": str(mel_fb.dtype),
        }


def build_manifest(
    tracer: ReferenceTracer,
    wav_path: Path,
    stats_path: Path | None,
):
    s = tracer.logmel.mel.spectrogram
    m = tracer.logmel.mel.mel_scale

    manifest = {
        "schema_version": 1,

        "reference": {
            "code2edge_root": "repo-root",
            "tiny_kws_source": "reference/tiny-kws",
        },

        "input": {
            "sample_rate": SAMPLE_RATE,
            "clip_samples": CLIP_SAMPLES,
            "channels": 1,
            "dtype": "float32",
            "file": wav_path.name,
            "sha256": sha256_file(wav_path),
        },

        "frontend": {
            "n_fft": N_FFT,
            "win_length": s.win_length,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "f_min": F_MIN,
            "f_max": F_MAX,
            "power": 2.0,
            "center": s.center,
            "pad_mode": s.pad_mode,
            "normalized": s.normalized,
            "onesided": s.onesided,
            "mel_scale": m.mel_scale,
            "mel_norm": m.norm,
            "log_epsilon": LOG_EPS,
            "expected_frames": N_FRAMES,
        },

        "normalization": {
            "stats_file": (
                str(stats_path) if stats_path else None
            ),
        },

        "labels": LABELS,

        "stages": tracer.manifest["stages"],
        "artifacts": tracer.manifest["artifacts"],
    }

    return manifest


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wav",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Optional stats.json generated by prepare_data.py",
    )

    parser.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    wav = load_wav_fixed(args.wav)

    stats = None
    if args.stats is not None:
        stats = load_stats(args.stats)

    tracer = ReferenceTracer(args.out)

    tracer.capture(
        wav,
        stats,
    )

    tracer.save_artifacts()

    manifest = build_manifest(
        tracer,
        args.wav,
        args.stats,
    )

    manifest_path = args.out / "manifest.json"

    with manifest_path.open("w") as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()