"""Model factory: ImageNet-pretrained backbones with a fresh 10-class head."""
import torch
import torch.nn as nn
from torchvision import models

ARCHS = ["mobilenet_v3_large", "efficientnet_b0", "resnet50"]


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True, num_classes: int = 10) -> nn.Module:
    if arch == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        w = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_large(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "resnet50":
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch}; choose from {ARCHS}")
    return m


def save_checkpoint(path, model, arch, img_size, extra=None):
    torch.save(
        {"state_dict": model.state_dict(), "arch": arch, "img_size": img_size, "extra": extra or {}},
        path,
    )


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["arch"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()  # eval(): BatchNorm uses learned stats (the old notebook got this wrong)
    return model, ck["img_size"]
