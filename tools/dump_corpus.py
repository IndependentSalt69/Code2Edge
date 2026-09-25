from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

# Import the already-validated reference tracer.
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(
    0,
    str(ROOT / "reference" / "tiny-kws" / "src"),
)

from dump_reference import (  # noqa: E402
    ReferenceTracer,
    build_manifest,
    load_wav_fixed,
)
from common import load_stats  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "reference" / "corpus_manifest.json",
    )

    parser.add_argument(
        "--stats",
        type=Path,
        default=(
            ROOT
            / "reference"
            / "tiny-kws"
            / "data"
            / "processed"
            / "stats.json"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reference" / "golden",
    )

    args = parser.parse_args()

    args.manifest = args.manifest.resolve()
    args.stats = args.stats.resolve()
    args.out = args.out.resolve()

    with args.manifest.open() as f:
        corpus = json.load(f)

    samples = corpus["samples"]

    stats = load_stats(args.stats)

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts_dir = args.out / "artifacts"
    artifacts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Dump global artifacts exactly once.
    # ------------------------------------------------------------
    first_sample_dir = args.out / "_reference_artifacts"

    first_sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    first = True

    corpus_summary = {
        "schema_version": 1,
        "source_manifest": str(
            repo_relative(args.manifest)
        ),
        "stats_file": str(
            repo_relative(args.stats)
        ),
        "sample_count": len(samples),
        "samples": [],
    }

    for index, sample in enumerate(samples, start=1):
        sample_id = sample["id"]

        wav_path = ROOT / sample["relative_path"]

        if not wav_path.exists():
            raise FileNotFoundError(wav_path)

        out_dir = args.out / sample_id
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Don't silently reuse an old result.
        for p in out_dir.glob("*.npy"):
            p.unlink()

        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()

        print(
            f"[{index:03d}/{len(samples)}] "
            f"{sample_id} {sample['label']}: "
            f"{sample['relative_path']}"
        )

        wav = load_wav_fixed(wav_path)

        tracer = ReferenceTracer(out_dir)

        tracer.capture(
            wav,
            stats,
        )

        # Saves window.npy and mel_filterbank.npy temporarily
        # in this sample directory.
        tracer.save_artifacts()

        sample_manifest = build_manifest(
            tracer,
            wav_path,
            args.stats,
        )

        # Add corpus identity + normalization values.
        sample_manifest["corpus"] = {
            "id": sample_id,
            "label": sample["label"],
            "relative_path": sample["relative_path"],
            "sha256": sample["sha256"],
        }

        sample_manifest["normalization"]["mean"] = float(
            stats["mean"]
        )
        sample_manifest["normalization"]["std"] = float(
            stats["std"]
        )
        sample_manifest["normalization"]["log_eps"] = float(
            stats["log_eps"]
        )
        sample_manifest["normalization"]["stats_sample"] = int(
            stats["stats_sample"]
        )

        with manifest_path.open("w") as f:
            json.dump(
                sample_manifest,
                f,
                indent=2,
            )

        # Move global artifacts once.
        if first:
            shutil.copy2(
                out_dir / "window.npy",
                artifacts_dir / "window.npy",
            )

            shutil.copy2(
                out_dir / "mel_filterbank.npy",
                artifacts_dir / "mel_filterbank.npy",
            )

            first = False

        # Avoid duplicating the same static arrays 500 times.
        (out_dir / "window.npy").unlink(missing_ok=True)
        (out_dir / "mel_filterbank.npy").unlink(
            missing_ok=True
        )

        # Record compact summary.
        corpus_summary["samples"].append(
            {
                "id": sample_id,
                "label": sample["label"],
                "relative_path": sample["relative_path"],
                "sha256": sample["sha256"],
                "stage_files": {
                    "post_input": "post_input.npy",
                    "post_power_spectrum": (
                        "post_power_spectrum.npy"
                    ),
                    "post_mel": "post_mel.npy",
                    "post_log": "post_log.npy",
                    "post_normalize": (
                        "post_normalize.npy"
                    ),
                },
            }
        )

    # ------------------------------------------------------------
    # Save corpus-level manifest.
    # ------------------------------------------------------------
    corpus_summary["artifacts"] = {
        "window": {
            "file": "artifacts/window.npy",
            "shape": list(
                np.load(
                    artifacts_dir / "window.npy"
                ).shape
            ),
        },
        "mel_filterbank": {
            "file": "artifacts/mel_filterbank.npy",
            "shape": list(
                np.load(
                    artifacts_dir / "mel_filterbank.npy"
                ).shape
            ),
        },
    }

    corpus_summary["normalization"] = {
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "log_eps": float(stats["log_eps"]),
        "stats_sample": int(stats["stats_sample"]),
    }

    output_manifest = args.out / "golden_manifest.json"

    with output_manifest.open("w") as f:
        json.dump(
            corpus_summary,
            f,
            indent=2,
        )

    print()
    print("Golden corpus complete.")
    print("samples:", len(samples))
    print("output:", args.out)
    print("manifest:", output_manifest)


if __name__ == "__main__":
    main()