"""Collect every experiment into one comparison table (+ chart) for the report and the demo app.

  python summarize_multi.py      -> results_multi/REPORT.md, results_multi/comparison.png
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent / "results_multi"

# (label, run folders whose test metrics are averaged, short explanation)
EXPERIMENTS = [
    ("1. Old model: State Farm only", ["sf_only"], "baseline - one car, one camera"),
    ("2. + Ain Shams + extra SF frames", ["multi"], "multi-dataset training, partial-label loss"),
    ("3. + mirror with label swap", ["mirror"], "adds right-hand-drive / mirrored viewpoints"),
    ("4. + wild sessions (cross-session)", ["wildA", "wildB"],
     "trained on one wild session, tested on the other; average of both directions"),
    ("5. Final demo model", ["final"], "everything above, all wild sessions in training (so no wild test)"),
]
TEST_SETS = {"sf": "State Farm", "as": "Ain Shams", "wild": "Wild photos"}
METRICS = {"group_acc": "4-group accuracy", "binary_acc": "Safe-vs-distracted accuracy",
           "false_alarm": "False-alarm rate (safe drivers flagged)", "fine_acc": "10-class accuracy"}


def load_runs():
    out = {}
    for d in ROOT.glob("*/test_metrics.json"):
        out[d.parent.name] = json.load(open(d))
    return out


def comparison_table(metric="group_acc") -> pd.DataFrame:
    runs = load_runs()
    rows = []
    for label, folders, note in EXPERIMENTS:
        if not all(f in runs for f in folders):
            continue
        row = {"Model": label}
        for key, name in TEST_SETS.items():
            vals = [runs[f][key].get(metric) for f in folders if key in runs[f]]
            vals = [v for v in vals if v is not None]
            row[name] = sum(vals) / len(vals) if vals else None
        row["What changed"] = note
        rows.append(row)
    return pd.DataFrame(rows)


def _md(df):
    fmt = lambda v: "-" if v is None or v != v else (f"{v:.1%}" if isinstance(v, float) else str(v))  # noqa: E731
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    return head + "\n" + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False))


def main():
    lines = ["# Multi-dataset experiments\n",
             "All numbers are on held-out data: unseen drivers (State Farm fold 0, Ain Shams test drivers) and "
             "wild phone photos from capture sessions never used for training by that model.\n",
             "4 groups = Safe / Texting / Phone call / Other distraction.\n"]
    for m, name in METRICS.items():
        t = comparison_table(m)
        if t.empty:
            continue
        lines += [f"## {name}\n", _md(t), ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = comparison_table("group_acc").set_index("Model")[list(TEST_SETS.values())]
    ax = (t * 100).plot.bar(figsize=(10, 5), rot=12)
    ax.set_ylabel("4-group accuracy on held-out data (%)"); ax.set_ylim(0, 100)
    ax.set_title("Generalisation to unseen drivers, cars and cameras")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f", fontsize=8)
    plt.tight_layout(); plt.savefig(ROOT / "comparison.png", dpi=140)
    print((ROOT / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
