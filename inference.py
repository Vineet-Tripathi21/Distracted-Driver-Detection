"""Shared inference helpers used by the Streamlit app and the webcam script."""
import numpy as np
import torch
from PIL import Image

from data import CLASSES, build_transforms
from model import load_checkpoint


class Predictor:
    def __init__(self, ckpt_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.img_size = load_checkpoint(ckpt_path, self.device)
        self.tf = build_transforms(self.img_size, train=False)

    @torch.no_grad()
    def predict(self, img: Image.Image) -> np.ndarray:
        """Return a length-10 probability vector."""
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(x), dim=1)[0].cpu().numpy()

    @staticmethod
    def describe(probs: np.ndarray):
        k = int(probs.argmax())
        return k, CLASSES[k], float(probs[k])
