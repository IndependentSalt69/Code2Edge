#!/usr/bin/env python3
"""
Verify integrity of the vendored reference workload (reference/tiny-kws/).
Recomputes SHA-256 hashes of all tracked reference files and verifies
them against reference/tiny-kws/UPSTREAM.md.
"""

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = REPO_ROOT / "reference" / "tiny-kws"
UPSTREAM_MD = REFERENCE_DIR / "UPSTREAM.md"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def parse_upstream_hashes() -> dict[str, str]:
    if not UPSTREAM_MD.exists():
        print(f"Error: {UPSTREAM_MD} not found.", file=sys.stderr)
        sys.exit(1)

    expected = {}
    in_hash_section = False

    with open(UPSTREAM_MD, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if "### File integrity" in stripped:
                in_hash_section = True
                continue
            if in_hash_section:
                if stripped.startswith("###") or stripped.startswith("##"):
                    break
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) >= 2:
                    rel_path, expected_hash = parts[0], parts[1]
                    expected[rel_path] = expected_hash.lower()

    return expected


def main() -> int:
    expected_hashes = parse_upstream_hashes()
    if not expected_hashes:
        print("Error: No file hashes found in UPSTREAM.md", file=sys.stderr)
        return 1

    print(f"Checking integrity of {len(expected_hashes)} reference files...")
    all_ok = True

    for rel_path, expected_hash in expected_hashes.items():
        file_path = REFERENCE_DIR / rel_path
        if not file_path.exists():
            print(f"  [MISSING] {rel_path}", file=sys.stderr)
            all_ok = False
            continue

        actual_hash = compute_sha256(file_path)
        if actual_hash != expected_hash:
            print(f"  [MISMATCH] {rel_path}", file=sys.stderr)
            print(f"    Expected: {expected_hash}", file=sys.stderr)
            print(f"    Actual:   {actual_hash}", file=sys.stderr)
            all_ok = False
        else:
            print(f"  [OK]       {rel_path}")

    if all_ok:
        print("\nAll reference files passed integrity verification.")
        return 0
    else:
        print("\nReference integrity check FAILED.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
