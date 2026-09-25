"""Evaluate a checkpoint on the OFFICIAL Speech Commands v2 test set.

Produces every number we report anywhere (README, model card, application
form) — nothing is ever hand-written:
  * accuracy + macro-F1 + per-class precision/recall/F1
  * confusion matrix image  -> assets/confusion_matrix.png
  * CPU single-clip inference latency (batch=1, 1 thread) — the edge metric
  * parameter count and serialized weight size
All results land in assets/metrics.json and are printed as a markdown table.
"""

import argparse
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader

from common import LABELS, load_stats, pick_device
from dataset import EvalFeaturesDataset
from model import DSCNN, count_parameters


def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = DSCNN(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_all(model, loader, device):
    preds, targets = [], []
    for feats, labels in loader:
        logits = model(feats.to(device))
        preds.append(logits.argmax(1).cpu())
        targets.append(labels)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


@torch.no_grad()
def cpu_latency_ms(model, n_warmup=50, n_runs=200):
    """Single-clip (batch=1) CPU latency with 1 thread — worst-case edge setting."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model_cpu = model.to("cpu").eval()
    x = torch.randn(1, 1, 64, 101)
    for _ in range(n_warmup):
        model_cpu(x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model_cpu(x)
        times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(prev_threads)
    arr = np.array(times)
    return {"mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)), "runs": n_runs, "threads": 1}


def plot_confusion(cm, out_path: Path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix — official Speech Commands v2 test set\n"
                 "(row-normalized; cell text = raw counts)")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/best.pt"))
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--assets-dir", type=Path, default=Path("assets"))
    args = ap.parse_args()

    processed = args.data_dir / "processed"
    stats = load_stats(processed / "stats.json")
    test_ds = EvalFeaturesDataset(processed / "test_feats.pt",
                                  processed / "test_labels.pt", stats)
    loader = DataLoader(test_ds, batch_size=512)
    print(f"official test set: {len(test_ds)} clips")

    model, ckpt = load_model(args.ckpt)
    device = pick_device()
    model.to(device)

    preds, targets = predict_all(model, loader, device)
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    class_ids = list(range(len(LABELS)))  # force 12x12 even if a class is absent
    report = classification_report(targets, preds, labels=class_ids,
                                   target_names=LABELS, digits=4,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(targets, preds, labels=class_ids)

    args.assets_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, args.assets_dir / "confusion_matrix.png")

    n_params = count_parameters(model)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    weight_mb = buf.getbuffer().nbytes / 1e6
    latency = cpu_latency_ms(model)

    metrics = {
        "checkpoint": str(args.ckpt),
        "trained_epochs": ckpt.get("epoch"),
        "test_set": "official speech_commands_test_set_v0.02",
        "n_test": len(test_ds),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {l: report[l] for l in LABELS},
        "confusion_matrix": cm.tolist(),
        "n_parameters": n_params,
        "weight_file_mb": weight_mb,
        "cpu_latency": latency,
    }
    with open(args.assets_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== RESULTS ({args.ckpt}) ===")
    print(f"accuracy : {acc:.4f}")
    print(f"macro-F1 : {macro_f1:.4f}")
    print(f"params   : {n_params:,} | weights {weight_mb:.2f} MB")
    print(f"CPU latency (batch=1, 1 thread): mean {latency['mean_ms']:.2f} ms, "
          f"p95 {latency['p95_ms']:.2f} ms")
    print("\n| class | precision | recall | F1 | support |")
    print("|---|---|---|---|---|")
    for l in LABELS:
        r = report[l]
        print(f"| {l} | {r['precision']:.4f} | {r['recall']:.4f} | "
              f"{r['f1-score']:.4f} | {int(r['support'])} |")


if __name__ == "__main__":
    main()
