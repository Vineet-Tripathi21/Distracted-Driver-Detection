"""Build ONE index over three datasets + a resized image cache (run once).

  python prepare_multi.py            # defaults match this project's folder layout

Sources
  sf : State Farm (data/)                                      10 classes, 20 drivers (after cleaning)
  as : Ain Shams "Novel Driver Distractions" (../data_ext/dataset)  same 10 classes, 70 drivers, 5 cars, IR
  mc : "Multi-Class Driver Behavior Image Dataset" (Zenodo)    5 COARSE classes, no driver ids

Labels are stored as a SET of allowed fine classes ("1|3" = texting, either hand), so coarse datasets
can train the 10-class model with a partial-label loss (see multi_data.py).

Splits (all leakage-safe):
  sf : driver-wise fold 0 (same seed/assignment as run_all.py) = test, folds 1-4 = train
       -> directly comparable with results/fold0 (the old State-Farm-only model)
  as : the dataset's own driver-wise split: train+validation = train, test = test (14 unseen drivers)
  mc : audited on the fly - most of it turns out to be State Farm frames (see the MC section in main()):
       duplicates of SF train images are dropped, other SF frames become 'sfx' (train only),
       and the genuinely new phone photos become 'wild' (TEST ONLY - the external "new cameras/cars" set)
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

MC_MAP = {  # coarse class -> allowed State Farm classes
    "safe_driving": [0],
    "texting_phone": [1, 3],
    "talking_phone": [2, 4],
    "other_activities": [5, 6, 7, 8, 9],
    "turning": [5, 6, 7, 8, 9],
}
LONG_SIDE = 320


def _cache_one(job):
    src, dst = job
    dst = Path(dst)
    if dst.is_file():
        return dst.name, _dhash_file(dst)
    try:
        im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    im.thumbnail((LONG_SIDE, LONG_SIDE), Image.BILINEAR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=92)
    return dst.name, _dhash(im)


def _dhash(im, size=8):
    g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).flatten()), 2)


def _dhash_file(p):
    return _dhash(Image.open(p))


def _bits(hex_hashes):
    h = np.array([int(x, 16) for x in hex_hashes], dtype=np.uint64)
    return np.unpackbits(h.view(np.uint8).reshape(len(h), 8), axis=1)


def group_near_duplicates(hashes, max_dist=10):
    """Union-find over images whose 64-bit dHash differ by <= max_dist bits."""
    h = np.array(hashes, dtype=np.uint64)
    n = len(h)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bits = np.unpackbits(h.view(np.uint8).reshape(n, 8), axis=1)  # n x 64
    for i in range(n):
        d = (bits[i + 1:] != bits[i]).sum(1)
        for j in np.nonzero(d <= max_dist)[0] + i + 1:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf_dir", default="data")
    ap.add_argument("--as_dir", default="../data_ext/dataset")
    ap.add_argument("--mc_dir", default="../data_ext/Multi-Class Driver Behavior Image Dataset")
    ap.add_argument("--out", default="data_multi")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--mc_test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from data import assign_folds, load_index  # imports torch: keep it out of the worker processes
    out = Path(a.out)
    rows = []

    # --- State Farm
    sf = assign_folds(load_index(a.sf_dir, verbose=True), 5, a.seed)
    for r in sf.itertuples():
        rows.append(dict(source="sf", subject=f"sf_{r.subject}", labels=str(r.label), fine=r.label,
                         split="test" if r.fold == 0 else "train", src=r.path))

    # --- Ain Shams
    asd = Path(a.as_dir)
    ai = pd.read_csv(asd / "Data.csv").rename(columns={"class": "cls"})
    for r in ai.itertuples():
        rows.append(dict(source="as", subject=f"as_{int(r.driver):02d}", labels=str(int(r.cls)), fine=int(r.cls),
                         split="test" if r.type == "test" else "train", src=str(asd / r.path)))

    # --- Multi-Class (coarse)
    mcd = Path(a.mc_dir)
    for cls, allowed in MC_MAP.items():
        for p in sorted((mcd / cls).iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append(dict(source="mc", subject="", labels="|".join(map(str, allowed)), fine=-1,
                                 split="", src=str(p), mc_class=cls))

    df = pd.DataFrame(rows)
    df["cache"] = [str(out / "cache" / s / (hashlib.md5(p.encode()).hexdigest()[:16] + ".jpg"))
                   for s, p in zip(df.source, df.src)]
    print(f"[index] {len(df)} images: " + ", ".join(f"{k}={v}" for k, v in df.source.value_counts().items()))

    print(f"[cache] resizing to long side {LONG_SIDE}px with {a.workers} processes (skips files already cached)...")
    jobs = list(zip(df.src, df.cache))
    hashes, bad = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (name, h) in enumerate(ex.map(_cache_one, jobs, chunksize=64)):
            if name is None:
                bad.append(i); hashes.append(0)
            else:
                hashes.append(h)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    df["dhash"] = [f"{h:016x}" for h in hashes]
    if bad:
        print(f"[cache] WARNING: {len(bad)} unreadable images dropped")
        df = df.drop(index=bad).reset_index(drop=True)

    # --- MC audit: ~91% of it are State Farm frames (file names img_<n>.jpg, 640x480).
    #   * identical to a State Farm TRAIN image  -> dropped (no new information + would leak SF test drivers)
    #   * other img_<n>.jpg (from SF's unlabelled Kaggle test set = different drivers, same car/camera)
    #                                            -> source "sfx", training only
    #   * everything else (phone photos in other cars / buses) -> source "wild", TEST ONLY (never trained on)
    df["name"] = df.src.map(lambda p: Path(p).name)
    m = df.source == "mc"
    is_sf_name = df.name.str.match(r"^img_\d+\.jpg$")
    sf_rows = df[df.source == "sf"]
    sf_bits = _bits(sf_rows.dhash)
    mc_idx = df.index[m & is_sf_name]
    near = np.array([(sf_bits != b).sum(1).argmin() for b in _bits(df.loc[mc_idx, "dhash"])])
    dist = np.array([(sf_bits[j] != b).sum() for j, b in zip(near, _bits(df.loc[mc_idx, "dhash"]))])
    dup = dist <= 4
    sf_lab = sf_rows.fine.to_numpy()[near[dup]]
    agree = np.mean([str(l) in s.split("|") for l, s in zip(sf_lab, df.loc[mc_idx[dup], "labels"])])
    print(f"[mc] {m.sum()} images: {dup.sum()} duplicate State Farm train images (dropped; their MC label agrees "
          f"with the State Farm label {agree:.1%} of the time), {(~dup).sum()} other State Farm-style frames -> 'sfx' "
          f"(train), {(m & ~is_sf_name).sum()} genuinely new photos -> 'wild' (test only)")
    df.loc[mc_idx[~dup], ["source", "subject", "split"]] = ["sfx", "sfx_kaggle_test", "train"]
    wild = m & ~is_sf_name
    df.loc[wild, ["source", "split"]] = ["wild", "test"]
    df.loc[wild, "subject"] = df.loc[wild, "name"].str.replace(r"\d+", "#", regex=True)
    df = df.drop(index=mc_idx[dup]).drop(columns=["name"]).reset_index(drop=True)

    df.to_csv(out / "index.csv", index=False)
    print("\n[split summary]")
    print(df.groupby(["source", "split"]).agg(images=("cache", "size"), subjects=("subject", "nunique")))
    print(f"\nwrote {out / 'index.csv'}")


if __name__ == "__main__":
    main()
