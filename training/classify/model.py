"""Models for the 27-class per-cell letter classifier.

`TileClassifier` is a small custom CNN over the canonicalized single-channel
60x60 crop (see autoscorer/perception/classify/canonicalize.py) -- the
canonicalization step already collapses tile-color/font variation down to
one visual domain, so a small from-scratch CNN is a reasonable fit for that
narrower problem.

`PretrainedTileClassifier` (WS2 of docs/classifier-accuracy-plan.md) wraps
an ImageNet-pretrained torchvision backbone instead. The bet: with only a
few hundred real tiles, low-level features learned from real photographs
(edges, lighting gradients, JPEG-compression texture) transfer better than
features learned only from our synthetic renderer, however good that
renderer gets. Both arms are trained and compared on the honest held-out
set; whichever wins ships.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

INPUT_SIZE = 60

# MobileNetV3 was trained at 224x224; shrinking a 60x60 canonicalized crop
# straight into it would collapse most of its downsampling stages to
# near-nothing and waste the pretrained spatial hierarchy, so the
# pretrained arm upscales after canonicalization instead. Mean/std are the
# per-channel average of ImageNet's stats, collapsed to one channel to
# roughly match how the first conv layer's weights were averaged below --
# an approximation, not a re-derived statistic, but far better than no
# normalization at all for a backbone this sensitive to input scale.
PRETRAINED_INPUT_SIZE = 224
PRETRAINED_NORMALIZE_MEAN = [0.449]
PRETRAINED_NORMALIZE_STD = [0.226]


class TileClassifier(nn.Module):
    def __init__(self, num_classes: int = 27, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 60 -> 30
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 30 -> 15
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).flatten(1)
        return self.classifier(features)


class PretrainedTileClassifier(nn.Module):
    """MobileNetV3-Small with ImageNet weights, adapted for a 1-channel
    canonicalized input (first-conv weights averaged across the original 3
    input channels rather than discarded) and a 27-way head.
    """

    def __init__(self, num_classes: int = 27) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        first_conv = backbone.features[0][0]
        new_first_conv = nn.Conv2d(
            1, first_conv.out_channels, kernel_size=first_conv.kernel_size,
            stride=first_conv.stride, padding=first_conv.padding, bias=False,
        )
        with torch.no_grad():
            new_first_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = new_first_conv

        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
