from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python tools/compare_power.py "
            "<reference.npy> <cpp.bin>"
        )

    reference_path = Path(sys.argv[1])
    cpp_path = Path(sys.argv[2])

    reference = np.load(reference_path)

    cpp = np.fromfile(
        cpp_path,
        dtype=np.float32,
    ).reshape(201, 101)

    # Reference is (1, 201, 101)
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

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    cosine = (
        float(np.dot(a, b) / denominator)
        if denominator != 0.0
        else 1.0
    )

    print("=== POWER SPECTRUM PARITY ===")
    print("reference shape:", reference.shape)
    print("cpp shape:      ", cpp.shape)
    print()
    print("max_abs_diff:   ", float(diff.max()))
    print("mean_abs_diff:  ", float(diff.mean()))
    print("cosine:         ", cosine)
    print()
    print("reference min:  ", float(reference.min()))
    print("reference max:  ", float(reference.max()))
    print("cpp min:        ", float(cpp.min()))
    print("cpp max:        ", float(cpp.max()))


if __name__ == "__main__":
    main()
