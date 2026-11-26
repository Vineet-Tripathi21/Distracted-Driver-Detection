"""Hyper-parameter search and driver-wise cross-validation for the multi-dataset model.

Uses a VALIDATION split carved out of the training drivers (never the test drivers), so the test
numbers in results_multi/REPORT.md stay untouched by tuning.

  python tune_multi.py sweep     # architectures / learning rates / mirror aug -> results_multi/tuning/
  python tune_multi.py cv        # 3-fold driver-wise CV of the chosen config -> results_multi/cv/
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from model import build_model
from multi_data import GROUP_OF, MultiDataset, group_probs, load_multi_index, partial_label_loss

SHARE = {"sf": 0.35, "sfx": 0.15, "as": 0.5}
SWEEP = [  # name, arch, lr, mirror_p
    ("effb0_lr1e-4", "efficientnet_b0", 1e-4, 0.5),
    ("effb0_lr3e-4", "efficientnet_b0", 3e-4, 0.5),
    ("effb0_lr1e-3", "efficientnet_b0", 1e-3, 0.5),
    ("effb0_lr3e-4_nomirror", "efficientnet_b0", 3e-4, 0.0),
    ("mobilenetv3_lr3e-4", "mobilenet_v3_large", 3e-4, 0.5),
    ("mobilenetv3_lr1e-3", "mobilenet_v3_large", 1e-3, 0.5),
    ("resnet50_lr3e-4", "resnet50", 3e-4, 0.5),
]


def subject_folds(df, k, seed=42):
    """Assign each (State Farm / Ain Shams) driver to one of k folds, stratified by source."""
    fold = {}
    rng = np.random.RandomState(seed)
    for src in ["sf", "as"]:
        subs = np.array(sorted(df[df.source == src].subject.unique()))
        rng.shuffle(subs)
        fold.update({s: i % k for i, s in enumerate(subs)})
    return df.subject.map(fold).fillna(-1).astype(int)  # sfx (no driver ids) -> -1 = always train


@torch.no_grad()
def evaluate(model, df, device, workers, img_size=224):
    loader = DataLoader(MultiDataset(df, img_size, False), batch_size=128, num_workers=min(workers, 3))
    model.eval()
    probs, loss, n = [], 0.0, 0
    for x, m in loader:
        x, m = x.to(device), m.to(device)
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            logits = model(x)
        loss += partial_label_loss(logits, m, smoothing=0.0).item() * len(x); n += len(x)
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
    p = np.concatenate(probs)
    fine = df.fine.to_numpy()
    true_g = GROUP_OF[fine]
    return {"val_loss": loss / n, "val_group_acc": float((group_probs(p).argmax(1) == true_g).mean()),
            "val_fine_acc": float((p.argmax(1) == fine).mean()),
            "val_macro_f1": float(f1_score(fine, p.argmax(1), average="macro"))}


def train_one(tr, va, arch, lr, mirror_p, epochs, samples, device, workers, seed=42, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    counts = tr.source.value_counts()
    tot = sum(SHARE[s] for s in counts.index)
    w = tr.source.map(lambda s: SHARE[s] / tot / counts[s]).to_numpy()
    sampler = WeightedRandomSampler(w, samples, replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(MultiDataset(tr, 224, True, mirror_p), batch_size=64, sampler=sampler,
                        num_workers=workers, drop_last=True, pin_memory=True, persistent_workers=True)
    model = build_model(arch, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps, warm = epochs * len(loader), len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    va_small = va.sample(min(len(va), 4000), random_state=0)  # per-epoch curve; full val at the end
    hist = []
    for ep in range(1, epochs + 1):
        model.train(); t0, tl, n = time.time(), 0.0, 0
        for x, m in loader:
            x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(x)
            loss = partial_label_loss(logits, m)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(x); n += len(x)
        r = {"epoch": ep, "train_loss": tl / n, **evaluate(model, va_small if ep < epochs else va, device, workers),
             "sec": round(time.time() - t0)}
        hist.append(r)
        log("   " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    del loader
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "cv"])
    ap.add_argument("--index", default="data_multi/index.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--samples_per_epoch", type=int, default=16000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mirror_p", type=float, default=0.5)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_all = load_multi_index(a.index, sources=["sf", "sfx", "as"], split="train")
    tr_all["fold"] = subject_folds(tr_all, a.folds if a.mode == "cv" else 5)

    out = Path("results_multi") / ("tuning" if a.mode == "sweep" else "cv")
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    results = {}
    if a.mode == "sweep":
        # validation = 1/5 of the training drivers of each dataset (fold 0)
        va, tr = tr_all[tr_all.fold == 0], tr_all[tr_all.fold != 0]
        log(f"[sweep] train {len(tr)} imgs / {tr.subject.nunique()} subjects | "
            f"val {len(va)} imgs / {va.subject.nunique()} drivers")
        if (out / "results.json").is_file():  # resume an interrupted sweep
            results = json.load(open(out / "results.json"))
        for name, arch, lr, mp in SWEEP:
            if name in results:
                continue
            log(f"== {name}: arch={arch} lr={lr} mirror_p={mp}")
            t0 = time.time()
            _, hist = train_one(tr, va, arch, lr, mp, a.epochs, a.samples_per_epoch, device, a.workers, log=log)
            results[name] = {"arch": arch, "lr": lr, "mirror_p": mp, "history": hist,
                             "minutes": round((time.time() - t0) / 60, 1), **{k: v for k, v in hist[-1].items()
                                                                              if k.startswith("val_")}}
            json.dump(results, open(out / "results.json", "w"), indent=2)
            torch.cuda.empty_cache()
        rows = [{"Config": k, "Backbone": v["arch"], "LR": v["lr"], "Mirror": v["mirror_p"],
                 "Val 4-group acc": v["val_group_acc"], "Val 10-class acc": v["val_fine_acc"],
                 "Val macro-F1": v["val_macro_f1"], "Val loss": v["val_loss"], "Minutes": v["minutes"]}
                for k, v in results.items()]
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for k, v in results.items():
            h = pd.DataFrame(v["history"])
            l, = axes[0].plot(h.epoch, h.train_loss, "--", alpha=.6)
            axes[0].plot(h.epoch, h.val_loss, "-", color=l.get_color(), label=k)
            axes[1].plot(h.epoch, h.val_fine_acc * 100, "-o", color=l.get_color(), label=k)
        axes[0].set_title("Loss (dashed = train, solid = validation)"); axes[0].set_xlabel("epoch")
        axes[1].set_title("Validation 10-class accuracy (%)"); axes[1].set_xlabel("epoch")
        axes[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "curves.png", dpi=130)
    else:
        log(f"[cv] {a.folds}-fold driver-wise CV | arch={a.arch} lr={a.lr} mirror_p={a.mirror_p}")
        for f in range(a.folds):
            va, tr = tr_all[tr_all.fold == f], tr_all[tr_all.fold != f]
            log(f"== fold {f}: val drivers {va.subject.nunique()} ({len(va)} imgs)")
            _, hist = train_one(tr, va, a.arch, a.lr, a.mirror_p, a.epochs, a.samples_per_epoch, device, a.workers,
                                log=log)
            res = {"fold": f, "val_drivers": int(va.subject.nunique()), "val_images": int(len(va)), **hist[-1]}
            # per-dataset accuracy inside the fold
            results[f"fold{f}"] = res
            json.dump(results, open(out / "results.json", "w"), indent=2)
            torch.cuda.empty_cache()
        df = pd.DataFrame(results.values())
        df.to_csv(out / "folds.csv", index=False)
        summ = {m: {"mean": float(df[m].mean()), "std": float(df[m].std())}
                for m in ["val_group_acc", "val_fine_acc", "val_macro_f1"]}
        json.dump(summ, open(out / "summary.json", "w"), indent=2)
        log(json.dumps(summ, indent=2))
    log("done")


if __name__ == "__main__":
    main()
