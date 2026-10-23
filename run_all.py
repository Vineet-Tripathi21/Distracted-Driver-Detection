"""Everything in one go:
   1) audit + clean the dataset
   2) 5-fold driver-wise cross-validation (honest accuracy +/- std)
   3) train the FINAL model on all drivers  -> results/final/model.pt  (used by the demo)
   4) write a ready-to-paste report: results/summary.md, summary.json, confusion matrix

Resumable: a finished fold/final model is skipped if you run it again.

  python run_all.py --data_dir data --epochs 5 --workers 4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from data import CLASSES, load_index
from train import get_parser, plot_confusion, train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip_cv", action="store_true")
    ap.add_argument("--skip_final", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    res = Path(a.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    # ---- 1) dataset audit -------------------------------------------------
    df = load_index(a.data_dir, verbose=True)
    per_class = df.label.value_counts().sort_index().to_dict()
    print("[data] images per class:", per_class)
    if len(df) != 22424 or df.subject.nunique() != 26:
        print("[data] NOTE: the official Kaggle State Farm train set has 22,424 images / 26 drivers; "
              "describe YOUR dataset by the numbers above in the report.")

    def common(out_dir, **kw):
        argv = ["--data_dir", a.data_dir, "--out_dir", str(out_dir), "--arch", a.arch,
                "--img_size", str(a.img_size), "--epochs", str(a.epochs), "--batch_size", str(a.batch_size),
                "--folds", str(a.folds), "--workers", str(a.workers), "--limit", str(a.limit)]
        if a.no_pretrained:
            argv.append("--no_pretrained")
        for k, v in kw.items():
            argv += [f"--{k}"] if v is True else [f"--{k}", str(v)]
        return get_parser().parse_args(argv)

    # ---- 2) cross-validation ---------------------------------------------
    fold_results = []
    if not a.skip_cv:
        for k in range(a.folds):
            d = res / f"fold{k}"
            if (d / "metrics.json").exists():
                print(f"\n=== fold {k}: already done, skipping ===")
            else:
                print(f"\n=== fold {k + 1}/{a.folds} ===")
                train_one(common(d, val_fold=k))
            fold_results.append(json.load(open(d / "metrics.json")))

    # ---- 3) final model ---------------------------------------------------
    if not a.skip_final:
        d = res / "final"
        if (d / "done.json").exists():
            print("\n=== final model: already done, skipping ===")
        else:
            print("\n=== final model (all drivers) ===")
            train_one(common(d, full=True))

    # ---- 4) report --------------------------------------------------------
    if fold_results:
        accs = np.array([r["val_acc"] for r in fold_results])
        f1s = np.array([r["macro_f1"] for r in fold_results])
        P, L = [], []
        for k in range(a.folds):
            z = np.load(res / f"fold{k}" / "val_preds.npz")
            P.append(z["preds"]); L.append(z["labels"])
        P, L = np.concatenate(P), np.concatenate(L)  # every driver is validated exactly once
        cm = confusion_matrix(L, P, labels=list(range(10)))
        plot_confusion(cm, f"Out-of-fold confusion matrix (all {a.folds} folds, acc={np.mean(P == L):.3f})",
                       res / "confusion_matrix_all_folds.png")
        rep = classification_report(L, P, labels=list(range(10)), target_names=CLASSES,
                                    output_dict=True, zero_division=0)
        summary = {
            "dataset": {"unique_images": int(len(df)), "drivers": int(df.subject.nunique()), "per_class": per_class},
            "arch": a.arch, "img_size": a.img_size, "epochs": a.epochs,
            "fold_acc": accs.tolist(), "fold_macro_f1": f1s.tolist(),
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "macro_f1_mean": float(f1s.mean()),
            "oof_acc": float(np.mean(P == L)), "oof_macro_f1": float(f1_score(L, P, average="macro")),
            "per_class": {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score", "support")}
                          for c in CLASSES},
        }
        json.dump(summary, open(res / "summary.json", "w"), indent=2)

        lines = [f"# Results ({a.arch}, {a.img_size}px, {a.epochs} epochs, {a.folds}-fold driver-wise CV)\n",
                 f"Dataset: {summary['dataset']['unique_images']} unique images, {summary['dataset']['drivers']} drivers, 10 classes.\n",
                 f"**Accuracy on unseen drivers: {summary['acc_mean']:.2%} +/- {summary['acc_std']:.2%}** "
                 f"(macro-F1 {summary['macro_f1_mean']:.3f}); out-of-fold accuracy {summary['oof_acc']:.2%}.\n",
                 "| Fold | Accuracy | Macro-F1 |", "|---|---|---|"]
        lines += [f"| {i} | {x:.2%} | {y:.3f} |" for i, (x, y) in enumerate(zip(accs, f1s))]
        lines += ["", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        lines += [f"| {c} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1-score']:.2f} | {int(v['support'])} |"
                  for c, v in summary["per_class"].items()]
        (res / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n" + "\n".join(lines))

    print(f"\nDone. Final model: {res / 'final' / 'model.pt'}   |   Report: {res / 'summary.md'}")
    print(f"Demo:  streamlit run app.py   (set the model path to {res / 'final' / 'model.pt'})")


if __name__ == "__main__":
    main()
