"""Exploratory data analysis of the three datasets (run after prepare_multi.py).

  python eda.py          -> eda/*.png + eda/EDA_SUMMARY.md (figures are also shown in the demo app)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from data import CLASSES
from multi_data import GROUP_NAMES, load_multi_index

OUT = Path("eda")
SRC = {"sf": "State Farm", "as": "Ain Shams", "sfx": "State Farm extra (coarse)", "wild": "Wild phone photos"}
COL = {"sf": "#2563eb", "as": "#16a34a", "sfx": "#93c5fd", "wild": "#ea580c"}
SHORT = ["c0 Safe", "c1 Text R", "c2 Call R", "c3 Text L", "c4 Call L", "c5 Radio", "c6 Drink",
         "c7 Reach", "c8 Hair", "c9 Talk"]
MC_NAMES = {"safe_driving": "Safe", "texting_phone": "Texting", "talking_phone": "Phone call",
            "other_activities": "Other", "turning": "Turning"}


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=120); plt.close(fig)


def image_stats(paths):
    """mean brightness, colourfulness (mean HSV saturation) and per-channel mean of each image."""
    rows = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        a = np.asarray(im, dtype=np.float32) / 255
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255
        rows.append({"brightness": a.mean(), "saturation": hsv[..., 1].mean(),
                     "r": a[..., 0].mean(), "g": a[..., 1].mean(), "b": a[..., 2].mean()})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    df = load_multi_index()
    rng = np.random.RandomState(0)
    s = {}

    # ---------------------------------------------------------------- 1. sizes and splits
    split = df.groupby(["source", "split"]).agg(images=("cache", "size"), subjects=("subject", "nunique"))
    s["splits"] = {f"{a}/{b}": {"images": int(r.images), "subjects": int(r.subjects)} for (a, b), r in split.iterrows()}
    raw_sf = pd.read_csv("data/driver_imgs_list.csv")
    s["sf_raw_rows"], s["sf_unique_rows"] = len(raw_sf), int(len(raw_sf.drop_duplicates()))

    # ---------------------------------------------------------------- 2. class balance
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [3, 3, 2]})
    for ax, src in zip(axes[:2], ["sf", "as"]):
        c = df[df.source == src].fine.value_counts().reindex(range(10), fill_value=0)
        ax.bar(SHORT, c.values, color=COL[src]); ax.set_title(f"{SRC[src]}: images per class")
        ax.tick_params(axis="x", rotation=60)
        s[f"class_counts_{src}"] = c.tolist()
        s[f"imbalance_ratio_{src}"] = round(float(c.max() / c.min()), 2)
    mc = df[df.source.isin(["sfx", "wild"])]
    ct = mc.groupby(["mc_class", "source"]).size().unstack(fill_value=0).rename(index=MC_NAMES)
    ct.plot.bar(ax=axes[2], color=[COL[c] for c in ct.columns], rot=30)
    axes[2].set_title("Multi-Class set: coarse classes"); axes[2].set_xlabel("")
    axes[2].legend(title=None, labels=["sfx (train)", "wild (test)"])
    s["mc_counts"] = {str(k): {kk: int(vv) for kk, vv in v.items()} for k, v in ct.to_dict("index").items()}
    save(fig, "01_class_balance.png")

    # ---------------------------------------------------------------- 3. images per driver
    fig, ax = plt.subplots(figsize=(13, 3.8))
    x0 = 0
    for src in ["sf", "as"]:
        per = df[df.source == src].groupby("subject").size().sort_values(ascending=False)
        ax.bar(np.arange(x0, x0 + len(per)), per.values, color=COL[src], label=f"{SRC[src]} ({len(per)} drivers)")
        s[f"imgs_per_driver_{src}"] = {"min": int(per.min()), "median": int(per.median()), "max": int(per.max())}
        x0 += len(per) + 3
    ax.set_title("Images per driver (each bar = one person)"); ax.set_xticks([]); ax.legend()
    save(fig, "02_images_per_driver.png")

    # ---------------------------------------------------------------- 4. raw resolution
    res = []
    for src in ["sf", "as", "sfx", "wild"]:
        sub = df[df.source == src]
        for p in sub.src.sample(min(300, len(sub)), random_state=0):
            with Image.open(p) as im:
                res.append({"source": src, "w": im.size[0], "h": im.size[1]})
    res = pd.DataFrame(res)
    res["wh"] = res.w.astype(str) + "x" + res.h.astype(str)
    s["resolutions"] = {src: {k: int(v) for k, v in res[res.source == src].wh.value_counts().head(3).items()}
                        for src in res.source.unique()}

    # ---------------------------------------------------------------- 5. brightness / colour (IR images!)
    stats = []
    for src in ["sf", "as", "wild"]:
        sub = df[df.source == src]
        st = image_stats(sub.cache.sample(min(600, len(sub)), random_state=0))
        st["source"] = src
        stats.append(st)
    stats = pd.concat(stats)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for src, g in stats.groupby("source", sort=False):
        axes[0].hist(g.brightness, bins=40, alpha=.55, color=COL[src], label=SRC[src])
        axes[1].hist(g.saturation, bins=40, alpha=.55, color=COL[src], label=SRC[src])
    axes[0].set_title("Brightness (mean pixel value)"); axes[1].set_title("Colourfulness (mean HSV saturation)")
    axes[0].axvline(0.15, ls="--", c="k", lw=1); axes[0].text(0.155, axes[0].get_ylim()[1] * .9, "night / IR frames")
    axes[0].legend()
    save(fig, "03_brightness_colour.png")
    s["pixel_stats"] = {src: {"brightness_mean": round(float(g.brightness.mean()), 3),
                              "dark_share(<0.15)": round(float((g.brightness < 0.15).mean()), 3),
                              "saturation_mean": round(float(g.saturation.mean()), 3),
                              "rgb_mean": [round(float(g[c].mean()), 3) for c in "rgb"]}
                        for src, g in stats.groupby("source")}

    # ---------------------------------------------------------------- 6. mean image per dataset = camera viewpoint
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, src in zip(axes, ["sf", "as", "wild"]):
        sub = df[df.source == src].cache.sample(400 if src != "wild" else 300, random_state=1)
        acc = np.mean([np.asarray(Image.open(p).convert("RGB").resize((224, 160)), dtype=np.float32) for p in sub], 0)
        ax.imshow(acc.astype(np.uint8)); ax.set_title(f"Average image - {SRC[src]}"); ax.axis("off")
    save(fig, "04_mean_image_per_dataset.png")

    # ---------------------------------------------------------------- 7. sample grid
    fig, axes = plt.subplots(2, 10, figsize=(20, 4.6))
    for row, src in enumerate(["sf", "as"]):
        for c in range(10):
            p = df[(df.source == src) & (df.fine == c)].cache.sample(1, random_state=rng).iloc[0]
            axes[row, c].imshow(Image.open(p)); axes[row, c].axis("off")
            if row == 0:
                axes[row, c].set_title(SHORT[c], fontsize=9)
        axes[row, 0].text(-0.15, 0.5, SRC[src], transform=axes[row, 0].transAxes, rotation=90, va="center",
                          ha="right", fontsize=10)
    save(fig, "05_samples_per_class.png")
    fig, axes = plt.subplots(1, 8, figsize=(20, 2.8))
    for ax, p in zip(axes, df[df.source == "wild"].sample(8, random_state=3).itertuples()):
        ax.imshow(Image.open(p.cache)); ax.axis("off"); ax.set_title(MC_NAMES[p.mc_class], fontsize=9)
    save(fig, "06_wild_samples.png")

    # ---------------------------------------------------------------- 8. 4-group balance (headline output)
    gmap = np.array([0, 1, 2, 1, 2, 3, 3, 3, 3, 3])
    fine = df[df.source.isin(["sf", "as"])]
    g = pd.Series(gmap[fine.fine.to_numpy()]).value_counts().sort_index()
    s["group_share"] = {GROUP_NAMES[i]: round(float(v / g.sum()), 3) for i, v in g.items()}

    json.dump(s, open(OUT / "eda_stats.json", "w"), indent=2)
    md = ["# EDA summary (generated by eda.py)\n", "```json", json.dumps(s, indent=2), "```"]
    (OUT / "EDA_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
