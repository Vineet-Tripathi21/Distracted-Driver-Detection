"""Create a tiny SYNTHETIC dataset in the Kaggle layout, to smoke-test the pipeline only.
(It is not real driver data - never report numbers from it.)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

out = Path(sys.argv[1] if len(sys.argv) > 1 else "dummy")
n_subj, per = 10, 6
rng = np.random.RandomState(0)
rows = []
for s in range(n_subj):
    subj = f"p{s:03d}"
    bg = rng.randint(60, 200, 3)
    for c in range(10):
        d = out / "imgs" / "train" / f"c{c}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per):
            im = Image.new("RGB", (160, 120), tuple(int(v) for v in bg + rng.randint(-15, 15, 3)))
            dr = ImageDraw.Draw(im)
            x = 10 + c * 14  # class-dependent block position => learnable signal
            dr.rectangle([x, 30, x + 25, 80], fill=(255, 40 * (c % 5), 255 - 25 * c))
            name = f"img_{s}_{c}_{i}.jpg"
            im.save(d / name)
            rows.append((subj, f"c{c}", name))
pd.DataFrame(rows, columns=["subject", "classname", "img"]).to_csv(out / "driver_imgs_list.csv", index=False)
print(f"wrote {len(rows)} synthetic images to {out}")
