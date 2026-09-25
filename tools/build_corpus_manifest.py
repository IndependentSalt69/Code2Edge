from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
TINY_KWS = ROOT / "reference" / "tiny-kws"
TEST_ROOT = TINY_KWS / "data" / "test_set"

LABELS = [
    "silence",
    "unknown",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]

TEST_DIR_TO_LABEL = {
    "_silence_": "silence",
    "_unknown_": "unknown",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)

    return h.hexdigest()


def discover_samples() -> dict[str, list[Path]]:
    by_label: dict[str, list[Path]] = {
        label: []
        for label in LABELS
    }

    for directory in sorted(TEST_ROOT.iterdir()):
        if not directory.is_dir():
            continue

        label = TEST_DIR_TO_LABEL.get(
            directory.name,
            directory.name,
        )

        if label not in by_label:
            continue

        by_label[label].extend(
            sorted(directory.glob("*.wav"))
        )

    return by_label


def make_quotas(total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(LABELS))

    quotas = {
        label: base
        for label in LABELS
    }

    for label in LABELS[:remainder]:
        quotas[label] += 1

    return quotas


def inspect_wav(path: Path) -> tuple[int, int]:
    info = sf.info(path)

    return info.samplerate, info.frames


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reference" / "corpus_manifest.json",
    )

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    by_label = discover_samples()

    total_available = sum(
        len(files)
        for files in by_label.values()
    )

    if total_available < args.count:
        raise RuntimeError(
            f"Only {total_available} samples available, "
            f"cannot select {args.count}"
        )

    quotas = make_quotas(args.count)

    rng = random.Random(args.seed)

    selected = []

    for label in LABELS:
        files = list(by_label[label])

        rng.shuffle(files)

        n = quotas[label]

        if len(files) < n:
            raise RuntimeError(
                f"Not enough '{label}' samples: "
                f"need {n}, found {len(files)}"
            )

        for path in files[:n]:
            sample_rate, frames = inspect_wav(path)

            if sample_rate != 16000:
                raise RuntimeError(
                    f"{path}: expected 16000 Hz, "
                    f"got {sample_rate}"
                )

            relative_path = path.relative_to(ROOT)

            selected.append(
                {
                    "id": f"sample_{len(selected) + 1:04d}",
                    "relative_path": relative_path.as_posix(),
                    "label": label,
                    "sample_rate": sample_rate,
                    "num_samples": frames,
                    "sha256": sha256_file(path),
                }
            )

    # Stable ordering independent of dictionary/file traversal.
    selected.sort(
        key=lambda item: item["relative_path"]
    )

    # Reassign IDs after sorting.
    for i, item in enumerate(selected, start=1):
        item["id"] = f"sample_{i:04d}"

    counts = {
        label: 0
        for label in LABELS
    }

    for item in selected:
        counts[item["label"]] += 1

    manifest = {
        "schema_version": 1,
        "dataset": {
            "name": "Google Speech Commands v0.02",
            "split": "official test set",
            "root": "reference/tiny-kws/data/test_set",
        },
        "selection": {
            "count": len(selected),
            "seed": args.seed,
            "strategy": "deterministic_stratified",
            "quotas": quotas,
        },
        "class_counts": counts,
        "samples": selected,
    }

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.out.open("w") as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        json.dumps(
            {
                "count": len(selected),
                "seed": args.seed,
                "class_counts": counts,
                "output": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()