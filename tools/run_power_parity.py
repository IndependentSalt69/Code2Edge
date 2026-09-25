from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CPP_BINARY = Path("/tmp/code2edge_power")

CORPUS_MANIFEST = (
    ROOT / "reference" / "corpus_manifest.json"
)

GOLDEN_ROOT = ROOT / "reference" / "golden"

WORK_ROOT = Path("/tmp/code2edge_power_parity")

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)

    return h.hexdigest()


def compare(reference_path: Path, cpp_path: Path) -> dict:
    reference = np.load(reference_path)

    cpp = np.fromfile(
        cpp_path,
        dtype=np.float32,
    )

    expected_size = 201 * 101

    if cpp.size != expected_size:
        raise RuntimeError(
            f"Expected {expected_size} floats, "
            f"got {cpp.size}"
        )

    cpp = cpp.reshape(201, 101)

    reference = reference.squeeze(0)

    if reference.shape != cpp.shape:
        raise RuntimeError(
            f"Shape mismatch: "
            f"reference={reference.shape}, "
            f"cpp={cpp.shape}"
        )

    ref = reference.astype(np.float64)
    got = cpp.astype(np.float64)

    diff = np.abs(ref - got)

    a = ref.reshape(-1)
    b = got.reshape(-1)

    ref_norm = np.linalg.norm(a)

    relative_l2 = (
        float(np.linalg.norm(a - b) / ref_norm)
        if ref_norm != 0.0
        else 0.0
    )

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    cosine = (
        float(np.dot(a, b) / denominator)
        if denominator != 0.0
        else 1.0
    )

    worst = np.unravel_index(
        np.argmax(diff),
        diff.shape,
    )

    return {
        "reference_shape": list(reference.shape),
        "cpp_shape": list(cpp.shape),
        "finite_reference": bool(
            np.isfinite(reference).all()
        ),
        "finite_cpp": bool(
            np.isfinite(cpp).all()
        ),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "relative_l2": relative_l2,
        "cosine_similarity": cosine,
        "worst_index": [int(worst[0]), int(worst[1])],
        "reference_at_worst": float(reference[worst]),
        "cpp_at_worst": float(cpp[worst]),
    }


def main() -> None:
    if not CPP_BINARY.exists():
        raise SystemExit(
            f"C++ binary not found: {CPP_BINARY}\n"
            "Compile it first."
        )

    with CORPUS_MANIFEST.open() as f:
        manifest = json.load(f)

    selected = manifest["samples"]

    if len(selected) != 500:
        raise RuntimeError(
            f"Expected 500 samples, got {len(selected)}"
        )

    WORK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print(
        f"Running power-spectrum parity "
        f"on {len(selected)} samples..."
    )

    for index, sample in enumerate(
        selected,
        start=1,
    ):
        sample_id = sample["id"]
        label = sample["label"]

        input_npy = (
            GOLDEN_ROOT
            / sample_id
            / "post_input.npy"
        )

        reference_npy = (
            GOLDEN_ROOT
            / sample_id
            / "post_power_spectrum.npy"
        )

        input_bin = (
            WORK_ROOT
            / f"{sample_id}_input.bin"
        )

        output_bin = (
            WORK_ROOT
            / f"{sample_id}_power.bin"
        )

        audio = np.load(
            input_npy
        ).astype(np.float32)

        if audio.shape != (16000,):
            raise RuntimeError(
                f"{sample_id}: unexpected input "
                f"shape {audio.shape}"
            )

        audio.tofile(input_bin)

        if output_bin.exists():
            output_bin.unlink()

        subprocess.run(
            [
                str(CPP_BINARY),
                str(input_bin),
                str(output_bin),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        metrics = compare(
            reference_npy,
            output_bin,
        )

        result = {
            "sample_id": sample_id,
            "label": label,
            "relative_path": sample[
                "relative_path"
            ],
            "input_sha256": sample["sha256"],
            **metrics,
        }

        results.append(result)

        print(
            f"[{index:02d}/{len(selected)}] "
            f"{label:8s} "
            f"max={metrics['max_abs_diff']:.6e} "
            f"mean={metrics['mean_abs_diff']:.6e} "
            f"rel_l2={metrics['relative_l2']:.6e} "
            f"cos={metrics['cosine_similarity']:.12f}"
        )

    max_result = max(
        results,
        key=lambda x: x["max_abs_diff"],
    )

    mean_result = max(
        results,
        key=lambda x: x["mean_abs_diff"],
    )

    worst_cosine = min(
        results,
        key=lambda x: x["cosine_similarity"],
    )

    worst_relative_l2 = max(
        results,
        key=lambda x: x["relative_l2"],
    )

    report = {
        "schema_version": 1,
        "stage": "post_power_spectrum",
        "sample_count": len(results),
        "classes": LABELS,
        "results": results,
        "summary": {
            "worst_max_abs_diff": {
                "sample_id": max_result["sample_id"],
                "label": max_result["label"],
                "value": max_result["max_abs_diff"],
            },
            "worst_mean_abs_diff": {
                "sample_id": mean_result["sample_id"],
                "label": mean_result["label"],
                "value": mean_result["mean_abs_diff"],
            },
            "worst_relative_l2": {
                "sample_id": worst_relative_l2["sample_id"],
                "label": worst_relative_l2["label"],
                "value": worst_relative_l2["relative_l2"],
            },
            "worst_cosine": {
                "sample_id": worst_cosine["sample_id"],
                "label": worst_cosine["label"],
                "value": worst_cosine["cosine_similarity"],
            },
        },
    }

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        report_dir
        / "power_parity_500.json"
    )

    with report_path.open("w") as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print()
    print("Power parity complete.")
    print("Report:", report_path)
    print()
    print(
        "Worst max_abs_diff:",
        max_result["max_abs_diff"],
        "(",
        max_result["sample_id"],
        max_result["label"],
        ")",
    )
    print(
        "Worst cosine:",
        worst_cosine["cosine_similarity"],
        "(",
        worst_cosine["sample_id"],
        worst_cosine["label"],
        ")",
    )


if __name__ == "__main__":
    main()