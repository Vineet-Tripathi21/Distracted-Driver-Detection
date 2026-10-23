"""Train / evaluate with driver-wise validation (single run). For everything in one go use run_all.py.

  python train.py --data_dir data --epochs 5 --val_fold 0 --out_dir runs/fold0
  python train.py --data_dir data --epochs 5 --full --out_dir runs/final
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data import CLASSES, DriverDataset, assign_folds, load_index
from model import ARCHS, build_model, save_checkpoint


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="runs/exp")
    p.add_argument("--arch", default="efficientnet_b0", choices=ARCHS)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--val_fold", type=int, default=0)
    p.add_argument("--full", action="store_true", help="train on all drivers, no validation")
    p.add_argument("--no_pretrained", action="store_true", help="random init (smoke tests only)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="use only N train images (smoke test)")
    p.add_argument("--seed", type=int, default=42)
    return p


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()  # crucial: BatchNorm must use running stats at eval time
    losses, preds, labels = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(criterion(out, y).item() * len(y))
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return sum(losses) / len(labels), preds, labels


def plot_confusion(cm, title, path):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels([f"c{i}" for i in range(10)])
    ax.set_yticklabels([f"c{i}: {c}" for i, c in enumerate(CLASSES)])
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def train_one(a) -> dict:
    """Train one model. Returns a dict of final-epoch metrics (empty for --full)."""
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = assign_folds(load_index(a.data_dir), a.folds, a.seed)
    if a.full:
        train_df, val_df = df, None
    else:
        train_df, val_df = df[df.fold != a.val_fold], df[df.fold == a.val_fold]
    if a.limit:
        train_df = train_df.sample(min(a.limit, len(train_df)), random_state=a.seed)
    print(f"device={device} | train imgs={len(train_df)} ({train_df.subject.nunique()} drivers)"
          + (f" | val imgs={len(val_df)} ({val_df.subject.nunique()} unseen drivers)" if val_df is not None else ""))

    tr_loader = DataLoader(DriverDataset(train_df, a.img_size, True), batch_size=a.batch_size,
                           shuffle=True, num_workers=a.workers, drop_last=True,
                           pin_memory=device == "cuda")
    va_loader = None
    if val_df is not None:
        va_loader = DataLoader(DriverDataset(val_df, a.img_size, False), batch_size=a.batch_size,
                               shuffle=False, num_workers=a.workers)

    model = build_model(a.arch, pretrained=not a.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * len(tr_loader)
    warm = max(1, len(tr_loader))  # 1 epoch warmup then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    eval_criterion = nn.CrossEntropyLoss()

    history = []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run_loss, correct, n = time.time(), 0.0, 0, 0
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        row = {"epoch": ep, "train_loss": run_loss / n, "train_acc": correct / n, "sec": time.time() - t0}
        if va_loader is not None:
            vl, vp, vy = evaluate(model, va_loader, device, eval_criterion)
            row.update(val_loss=vl, val_acc=float((vp == vy).mean()), val_macro_f1=float(f1_score(vy, vp, average="macro")))
        history.append(row)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    # Always report the FINAL epoch (no picking the best epoch on the val set => no optimistic bias)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    save_checkpoint(out / "model.pt", model, a.arch, a.img_size, {"classes": CLASSES, "args": vars(a)})

    result = {}
    if va_loader is not None:
        vl, vp, vy = evaluate(model, va_loader, device, eval_criterion)
        rep = classification_report(vy, vp, labels=list(range(10)), target_names=CLASSES,
                                    output_dict=True, zero_division=0)
        result = {"val_loss": vl, "val_acc": float((vp == vy).mean()),
                  "macro_f1": float(f1_score(vy, vp, average="macro")), "report": rep}
        np.savez(out / "val_preds.npz", preds=vp, labels=vy)
        json.dump(result, open(out / "metrics.json", "w"), indent=2)  # written last => marks the run as complete
        cm = confusion_matrix(vy, vp, labels=list(range(10)))
        plot_confusion(cm, f"Normalised confusion matrix (unseen drivers, acc={np.mean(vp == vy):.3f})",
                       out / "confusion_matrix.png")
        print(f"\nFINAL (unseen drivers): acc={result['val_acc']:.4f} macro-F1={result['macro_f1']:.4f}")
    else:
        json.dump({"full": True}, open(out / "done.json", "w"))
    print(f"saved to {out}/")
    return result


if __name__ == "__main__":
    train_one(get_parser().parse_args())
