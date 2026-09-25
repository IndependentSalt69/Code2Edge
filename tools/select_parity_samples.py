import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "reference" / "corpus_manifest.json") as f:
    manifest = json.load(f)

samples = manifest["samples"]

selected = {}
for sample in samples:
    label = sample["label"]

    if label not in selected:
        selected[label] = sample

for label, sample in selected.items():
    print(
        sample["id"],
        sample["label"],
        sample["relative_path"],
    )