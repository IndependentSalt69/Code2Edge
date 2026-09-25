"""Train the DS-CNN keyword spotter.

Usage:
  python src/train.py --epochs 3            # local MPS smoke test
  python src/train.py --epochs 30           # full run (Colab T4)

Device handling:
  * model runs on CUDA > MPS > CPU (auto-detected, fp32 everywhere —
    fp16 autocast is not reliable on MPS)
  * the log-mel frontend runs on CUDA when available (fast, exact), but on
    CPU when the model is on MPS, because torchaudio spectrogram ops have
    gaps on MPS.

The checkpoint bundles everything inference needs (weights, model config,
label order, feature normalization stats), so the demo app can never drift
out of sync with training.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import LABELS, LogMel, load_stats, normalize, pick_device
from dataset import EvalFeaturesDataset, TrainWaveformDataset
from model import DSCNN, count_parameters


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        pred = model(feats).argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    print(f"device: {device}")

    processed = args.data_dir / "processed"
    stats = load_stats(processed / "stats.json")

    train_ds = TrainWaveformDataset(processed, augment=True)
    val_ds = EvalFeaturesDataset(processed / "val_feats.pt",
                                 processed / "val_labels.pt", stats)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, persistent_workers=args.workers > 0,
        pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=512, num_workers=0)
    print(f"train {len(train_ds)} clips | val {len(val_ds)} clips")

    model = DSCNN(n_classes=len(LABELS)).to(device)
    n_params = count_parameters(model)
    print(f"DS-CNN parameters: {n_params:,} (~{n_params * 4 / 1e6:.2f} MB fp32)")

    # log-mel frontend: CUDA if available, else CPU (never MPS)
    mel_device = device if device.type == "cuda" else torch.device("cpu")
    logmel = LogMel().to(mel_device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history, best_val = [], 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = running_correct = running_n = 0
        for wav, labels in tqdm(train_loader, desc=f"epoch {epoch}",
                                disable=None, mininterval=5):
            with torch.no_grad():
                feats = normalize(logmel(wav.to(mel_device)), stats)
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.numel()
            running_correct += (logits.argmax(1) == labels).sum().item()
            running_n += labels.numel()

        val_acc = evaluate(model, val_loader, device)
        train_loss = running_loss / running_n
        train_acc = running_correct / running_n
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_acc": val_acc,
                        "lr": scheduler.get_last_lr()[0], "seconds": dt})
        print(f"epoch {epoch:3d} | loss {train_loss:.4f} | "
              f"train acc {train_acc:.4f} | val acc {val_acc:.4f} | {dt:.0f}s")

        ckpt = {
            "model_state": model.state_dict(),
            "model_config": model.config,
            "labels": LABELS,
            "stats": stats,
            "val_acc": val_acc,
            "epoch": epoch,
            "history": history,
            # str(): the TorchVersion object isn't loadable under weights_only=True
            "torch_version": str(torch.__version__),
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(ckpt, args.out_dir / "best.pt")
            print(f"  -> new best val acc {best_val:.4f}, saved best.pt")

    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done. best val acc: {best_val:.4f}")


if __name__ == "__main__":
    main()
