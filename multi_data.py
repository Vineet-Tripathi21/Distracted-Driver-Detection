"""Multi-dataset loading, partial-label loss and the 4-group view used by the demo.

GROUPS is the camera-robust headline output:
    Safe (c0) | Texting (c1, c3) | Phone call (c2, c4) | Other distraction (c5-c9)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from data import CLASSES, MEAN, STD

GROUP_NAMES = ["Safe driving", "Texting", "Phone call", "Other distraction"]
GROUP_OF = np.array([0, 1, 2, 1, 2, 3, 3, 3, 3, 3])
SOURCE_NAMES = {"sf": "State Farm (unseen drivers)", "sfx": "State Farm Kaggle-test frames (coarse labels)",
                "as": "Ain Shams (unseen drivers, other cars, IR)",
                "wild": "Wild phone photos (new cars, buses, cameras)"}


def group_probs(p):
    """10-class probabilities -> 4 group probabilities (works on 1-D or 2-D arrays)."""
    p = np.atleast_2d(p)
    return np.stack([p[:, GROUP_OF == g].sum(1) for g in range(4)], axis=1)


def load_multi_index(path="data_multi/index.csv", sources=None, split=None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"labels": str}, low_memory=False)
    if sources:
        df = df[df.source.isin(sources)]
    if split:
        df = df[df.split == split]
    return df.reset_index(drop=True)


def label_mask(labels: str) -> np.ndarray:
    m = np.zeros(10, dtype=np.float32)
    m[[int(x) for x in str(labels).split("|")]] = 1
    return m


def multi_transforms(img_size: int, train: bool):
    if train:
        # Wider crop/aspect range + occasional grayscale (IR cameras) for viewpoint/camera robustness.
        # No PLAIN horizontal flip here: mirroring is done in MultiDataset together with a label swap.
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.8)),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.04),
            T.RandomGrayscale(p=0.15),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(MEAN, STD)])


# Mirroring an image turns a left-hand-drive car into a right-hand-drive one and swaps the driver's hands,
# so the label must swap too: texting R<->L (c1<->c3), phone call R<->L (c2<->c4).
MIRROR_PERM = np.array([0, 3, 4, 1, 2, 5, 6, 7, 8, 9])


def wild_group(subject: str) -> str:
    """Capture sessions of the wild set: A = 2019 car photos (*.png), B = 2024 phone photos (cars + buses)."""
    return "A" if str(subject).startswith("#-#") else "B"


class MultiDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, train: bool, mirror_p: float = 0.0):
        self.paths = df["cache"].tolist()
        self.masks = np.stack([label_mask(l) for l in df["labels"]])
        self.tf = multi_transforms(img_size, train)
        self.mirror_p = mirror_p if train else 0.0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        mask = self.masks[i]
        if self.mirror_p and torch.rand(1).item() < self.mirror_p:  # torch RNG is seeded per DataLoader worker
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask[MIRROR_PERM]
        return self.tf(img), torch.from_numpy(np.ascontiguousarray(mask))


def partial_label_loss(logits, mask, smoothing=0.1):
    """-log P(true class is in the allowed set). With a single allowed class this is plain cross-entropy.

    Smoothing mixes in a little uniform cross-entropy (like label smoothing)."""
    logp = F.log_softmax(logits.float(), dim=1)
    nll = -torch.logsumexp(logp.masked_fill(mask == 0, -1e4), dim=1)
    return ((1 - smoothing) * nll - smoothing * logp.mean(1)).mean()


__all__ = ["CLASSES", "GROUP_NAMES", "GROUP_OF", "SOURCE_NAMES", "MIRROR_PERM", "group_probs", "load_multi_index",
           "label_mask", "multi_transforms", "wild_group", "MultiDataset", "partial_label_loss"]
