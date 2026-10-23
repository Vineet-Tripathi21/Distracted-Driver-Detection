"""Data loading for the State Farm Distracted Driver dataset.

Expected layout (the Kaggle download, unzipped):
    <data_dir>/driver_imgs_list.csv
    <data_dir>/imgs/train/c0/*.jpg ... c9/*.jpg

IMPORTANT design choices
  * Splits are made BY DRIVER (subject). Frames of one driver are near-duplicates,
    so a random image split leaks and inflates accuracy.
  * No horizontal flip augmentation: flipping swaps left/right hand, which would
    turn c1<->c3 (texting) and c2<->c4 (phone call) into the wrong label.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASSES = [
    "Safe driving",
    "Texting - right hand",
    "Phone call - right hand",
    "Texting - left hand",
    "Phone call - left hand",
    "Operating the radio",
    "Drinking",
    "Reaching behind",
    "Hair / makeup",
    "Talking to passenger",
]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_index(data_dir, verbose: bool = False) -> pd.DataFrame:
    """Return a clean dataframe with columns: subject, label, path.

    Cleaning: drops duplicated rows and rows whose image file is missing on disk.
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "driver_imgs_list.csv")
    n_raw = len(df)
    df = df.drop_duplicates(subset=["subject", "classname", "img"]).reset_index(drop=True)
    n_unique = len(df)
    df["label"] = df["classname"].str[1:].astype(int)
    df["path"] = [
        str(data_dir / "imgs" / "train" / c / i) for c, i in zip(df["classname"], df["img"])
    ]
    exists = np.array([Path(p).is_file() for p in df["path"]])
    df = df[exists].reset_index(drop=True)
    if verbose:
        print(f"[data] csv rows={n_raw} | unique={n_unique} | found on disk={len(df)} "
              f"| drivers={df.subject.nunique()}")
        if n_unique != len(df):
            print(f"[data] WARNING: {n_unique - len(df)} listed images are missing on disk and were dropped")
    return df[["subject", "label", "path"]]


def assign_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Add a 'fold' column. All images of a driver land in the same fold."""
    subjects = np.array(sorted(df["subject"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    fold_of = {s: i % n_folds for i, s in enumerate(subjects)}
    df = df.copy()
    df["fold"] = df["subject"].map(fold_of)
    return df


def build_transforms(img_size: int, train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(1.2, 1.45)),
            T.RandomRotation(8),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class DriverDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, train: bool):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tf = build_transforms(img_size, train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), torch.tensor(self.labels[idx], dtype=torch.long)