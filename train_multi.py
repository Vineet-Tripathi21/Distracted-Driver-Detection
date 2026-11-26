"""Train on several datasets at once and evaluate on the held-out test split of EVERY dataset.

  # combined model (used by the demo)
  python train_multi.py --train_sources sf,as,mc --out_dir results_multi/all
  # leave-one-dataset-out: never sees the Multi-Class phone photos -> honest "new cameras/cars" test
  python train_multi.py --train_sources sf,as --out_dir results_multi/sf_as
  # evaluate an existing checkpoint only (e.g. the old State-Farm-only model, trained on the same SF drivers)
  python train_multi.py --eval_only --ckpt results/fold0/model.pt --out_dir results_multi/sf_only

Metrics are always from the FINAL epoch (no picking the best epoch on test data).
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
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from model import ARCHS, build_model, load_checkpoint, save_checkpoint
from multi_data import (CLASSES, GROUP_NAMES, GROUP_OF, SOURCE_NAMES, MultiDataset, group_probs,
                        load_multi_index, partial_label_loss, wild_group)


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="data_multi/index.csv")
    p.add_argument("--train_sources", default="sf,sfx,as")
    p.add_argument("--source_share", default="sf=0.35,sfx=0.15,as=0.5,wild=0.10",
                   help="fraction of each epoch drawn from each source (balances dataset sizes)")
    p.add_argument("--mirror_p", type=float, default=0.0,
                   help="probability of mirroring a training image WITH left/right label swap")
    p.add_argument("--wild_train", default="", choices=["", "A", "B", "all"],
                   help="also train on wild capture session A (2019 cars) / B (2024 cars+buses) / all; "
                        "the remaining wild sessions stay as the test set")
    p.add_argument("--samples_per_epoch", type=int, default=36000)
    p.add_argument("--out_dir", default="results_multi/all")
    p.add_argument("--arch", default="efficientnet_b0", choices=ARCHS)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--ckpt", default="")
    return p


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, df, img_size, device, workers):
    loader = DataLoader(MultiDataset(df, img_size, False), batch_size=128, shuffle=False, num_workers=workers)
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            out.append(torch.softmax(model(x.to(device)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def plot_cm(cm, names, title, path):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * n, 0.8 + 0.65 * n))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(range(n) if n > 4 else names, rotation=0 if n > 4 else 20)
    ax.set_yticklabels([f"{i}: {c}" for i, c in enumerate(names)])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=7 if n > 4 else 9)
    ax.set_xlabel("Predicted"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def evaluate_all(model, img_size, index, out_dir, device, workers, title, exclude=()):
    out_dir = Path(out_dir)
    test = load_multi_index(index, split="test")
    test = test[~test.cache.isin(set(exclude))]  # never score images that were used for training
    res, lines = {}, [f"# {title}\n", "Held-out TEST splits (unseen drivers / unseen photo groups).\n",
                      "| Test set | Images | 4-group acc (Safe/Text/Call/Other) | Safe-vs-distracted acc | "
                      "Distracted recall | False-alarm rate | 10-class acc | 10-class macro-F1 |",
                      "|---|---|---|---|---|---|---|---|"]
    for src in ["sf", "as", "wild"]:
        df = test[test.source == src]
        if df.empty:
            continue
        p = predict(model, df, img_size, device, workers)
        masks = np.stack([np.isin(np.arange(10), [int(x) for x in l.split("|")]) for l in df["labels"]])
        true_g = np.array([GROUP_OF[np.argmax(m)] for m in masks])  # every allowed set lies inside one group
        pred_g = group_probs(p).argmax(1)
        td, pdis = true_g != 0, (1 - p[:, 0]) >= 0.5
        r = {"n": int(len(df)), "group_acc": float((pred_g == true_g).mean()),
             "binary_acc": float((pdis == td).mean()),
             "distracted_recall": float(pdis[td].mean()) if td.any() else None,
             "false_alarm": float(pdis[~td].mean()) if (~td).any() else None}
        plot_cm(confusion_matrix(true_g, pred_g, labels=range(4)), GROUP_NAMES,
                f"{SOURCE_NAMES[src]} - 4 groups (acc {r['group_acc']:.3f})", out_dir / f"cm_groups_{src}.png")
        fine = df["fine"].to_numpy()
        if (fine >= 0).all():
            pf = p.argmax(1)
            r["fine_acc"] = float((pf == fine).mean())
            r["fine_macro_f1"] = float(f1_score(fine, pf, average="macro"))
            plot_cm(confusion_matrix(fine, pf, labels=range(10)), CLASSES,
                    f"{SOURCE_NAMES[src]} - 10 classes (acc {r['fine_acc']:.3f})", out_dir / f"cm_fine_{src}.png")
        np.savez(out_dir / f"test_probs_{src}.npz", probs=p, true_group=true_g, fine=fine)
        res[src] = r
        f = lambda v: "n/a" if v is None else f"{v:.1%}"  # noqa: E731
        lines.append(f"| {SOURCE_NAMES[src]} | {r['n']} | {f(r['group_acc'])} | {f(r['binary_acc'])} | "
                     f"{f(r['distracted_recall'])} | {f(r['false_alarm'])} | {f(r.get('fine_acc'))} | "
                     + (f"{r['fine_macro_f1']:.3f} |" if "fine_macro_f1" in r else "n/a (coarse labels) |"))
        print(lines[-1], flush=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(res, open(out_dir / "test_metrics.json", "w"), indent=2)
    return res


# ----------------------------------------------------------------------------- training
def main():
    a = get_parser().parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    if a.eval_only:
        model, img_size = load_checkpoint(a.ckpt, device)
        evaluate_all(model, img_size, a.index, out, device, a.workers, f"Evaluation of {a.ckpt}")
        return

    sources = a.train_sources.split(",")
    tr = load_multi_index(a.index, sources=sources, split="train")
    if a.wild_train:
        w = load_multi_index(a.index, sources=["wild"])
        if a.wild_train != "all":
            w = w[w.subject.map(wild_group) == a.wild_train]
        tr = pd.concat([tr, w], ignore_index=True)
        sources = sources + ["wild"]
    share = {k: float(v) for k, v in (kv.split("=") for kv in a.source_share.split(","))}
    share = {s: share.get(s, 1.0) for s in sources}
    tot = sum(share.values())
    counts = tr.source.value_counts()
    weights = tr.source.map(lambda s: share[s] / tot / counts[s]).to_numpy()
    print(f"device={device} | train sources={sources} | " +
          ", ".join(f"{s}: {counts[s]} imgs ({tr[tr.source == s].subject.nunique()} subjects/groups), "
                    f"{share[s] / tot:.0%} of each epoch" for s in sources), flush=True)

    sampler = WeightedRandomSampler(weights, a.samples_per_epoch, replacement=True,
                                    generator=torch.Generator().manual_seed(a.seed))
    loader = DataLoader(MultiDataset(tr, a.img_size, True, a.mirror_p), batch_size=a.batch_size, sampler=sampler,
                        num_workers=a.workers, drop_last=True, pin_memory=device == "cuda", persistent_workers=True)

    model = build_model(a.arch, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps, warm = a.epochs * len(loader), len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    history = []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot_loss, hit, n = time.time(), 0.0, 0, 0
        for x, m in loader:
            x, m = x.to(device, non_blocking=True), m.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(x)
            loss = partial_label_loss(logits, m)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot_loss += loss.item() * len(x)
            hit += m.gather(1, logits.argmax(1, keepdim=True)).sum().item()  # prediction inside allowed set
            n += len(x)
        row = {"epoch": ep, "train_loss": tot_loss / n, "train_acc_in_set": hit / n, "sec": round(time.time() - t0)}
        history.append(row)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    save_checkpoint(out / "model.pt", model, a.arch, a.img_size,
                    {"classes": CLASSES, "train_sources": sources, "args": vars(a)})
    title = (f"EfficientNet-B0 trained on {' + '.join(SOURCE_NAMES[s] for s in sources)}"
             + (f" (wild session(s) used for training: {a.wild_train})" if a.wild_train else "")
             + (f", mirror+label-swap p={a.mirror_p}" if a.mirror_p else ""))
    evaluate_all(model, a.img_size, a.index, out, device, a.workers, title, exclude=tr.cache)
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
