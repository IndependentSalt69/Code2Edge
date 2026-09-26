#!/usr/bin/env python3
"""
Download the frozen reference checkpoint (best.pt) from Hugging Face Hub
and save to checkpoints/best.pt.
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Optional

HF_URL = "https://huggingface.co/priyadeepjaiswal9c/tiny-kws/resolve/main/best.pt"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DEST = REPO_ROOT / "checkpoints" / "best.pt"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def download_checkpoint(url: str, dest_path: Path, expected_hash: Optional[str] = None) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoint from:\n  {url}\nto:\n  {dest_path}")

    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                sys.stdout.write(f"\rProgress: {percent:5.1f}% ({downloaded / 1024:.1f} / {total_size / 1024:.1f} KB)")
            else:
                sys.stdout.write(f"\rDownloaded: {downloaded / 1024:.1f} KB")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nError downloading checkpoint: {e}", file=sys.stderr)
        return False

    actual_hash = compute_sha256(dest_path)
    print(f"File SHA-256: {actual_hash}")

    if expected_hash and actual_hash != expected_hash.lower():
        print(f"ERROR: Hash mismatch! Expected {expected_hash}, got {actual_hash}", file=sys.stderr)
        return False

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch frozen tiny-kws checkpoint from Hugging Face.")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Destination path for best.pt")
    parser.add_argument("--expected-hash", type=str, default=None, help="Expected SHA-256 hash")
    parser.add_argument("--force", action="store_true", help="Overwrite existing checkpoint")
    args = parser.parse_args()

    if args.dest.exists() and not args.force:
        print(f"Checkpoint already exists at {args.dest}.")
        print(f"SHA-256: {compute_sha256(args.dest)}")
        print("Use --force to re-download.")
        return 0

    success = download_checkpoint(HF_URL, args.dest, args.expected_hash)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
