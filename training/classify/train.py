"""Trains the per-cell letter classifier on a dataset directory laid out as
<root>/<label>/*.png (exactly what
training.synth_render.tile_renderer.generate_synthetic_dataset produces,
and compatible with a real-photo fine-tuning set in the same layout).
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from autoscorer.perception.classify.canonicalize import canonicalize_pil
from training.classify.model import (
    PRETRAINED_INPUT_SIZE,
    PRETRAINED_NORMALIZE_MEAN,
    PRETRAINED_NORMALIZE_STD,
    PretrainedTileClassifier,
    TileClassifier,
)

MODEL_TYPES = ("cnn", "pretrained")


def best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(model_type: str, num_classes: int) -> nn.Module:
    if model_type == "cnn":
        return TileClassifier(num_classes=num_classes)
    if model_type == "pretrained":
        return PretrainedTileClassifier(num_classes=num_classes)
    raise ValueError(f"unknown model_type {model_type!r}, expected one of {MODEL_TYPES}")


def canonicalized_transform(model_type: str) -> transforms.Compose:
    # canonicalize_pil already resizes to the small CNN's expected input
    # size (and converts to single-channel); the pretrained arm needs a
    # further upscale to something closer to what MobileNetV3 was trained
    # at, plus ImageNet-style normalization, or its pretrained features
    # don't transfer well (see model.py's PRETRAINED_* constants).
    if model_type == "pretrained":
        return transforms.Compose([
            transforms.Lambda(canonicalize_pil),
            transforms.Resize((PRETRAINED_INPUT_SIZE, PRETRAINED_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=PRETRAINED_NORMALIZE_MEAN, std=PRETRAINED_NORMALIZE_STD),
        ])
    return transforms.Compose([
        transforms.Lambda(canonicalize_pil),
        transforms.ToTensor(),
    ])


def _weighted_sampler(dataset: ImageFolder, indices: List[int], seed: int) -> WeightedRandomSampler:
    """Per-sample weights inversely proportional to class frequency *within
    this subset*, so a training batch doesn't end up dominated by whichever
    class happens to have the most source images -- see WS2 of
    docs/classifier-accuracy-plan.md (this replaced an earlier approach that
    duplicated files on disk to balance counts, which worked but doesn't
    compose with random subsetting/splitting).
    """
    labels = [dataset.samples[i][1] for i in indices]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def build_dataloaders(
    data_dir: Path, batch_size: int = 64, val_fraction: float = 0.2, seed: int = 0,
    balanced_sampling: bool = True, model_type: str = "cnn",
) -> Tuple[DataLoader, DataLoader, List[str]]:
    dataset = ImageFolder(str(data_dir), transform=canonicalized_transform(model_type))
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    if balanced_sampling:
        sampler = _weighted_sampler(dataset, train_ds.indices, seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, dataset.classes


@dataclass(frozen=True)
class TrainResult:
    classes: List[str]
    final_val_accuracy: float
    model_type: str


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def run_training(
    data_dir: Path,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = None,
    seed: int = 0,
    init_checkpoint: Optional[Path] = None,
    model_type: str = "cnn",
    balanced_sampling: bool = True,
) -> Tuple[nn.Module, TrainResult]:
    """If `init_checkpoint` is given, continues training (fine-tunes) from
    those weights instead of a random init -- e.g. starting from a
    synthetic-only checkpoint and fine-tuning on a real-photo dataset.
    Requires the checkpoint's class list and model_type to exactly match.
    """
    device = device or best_device()
    # Seed the global RNG, not just the dataset split -- weight init and the
    # DataLoader's shuffling both draw from it, so without this, `seed` gave
    # a false impression of reproducibility while runs actually still varied.
    torch.manual_seed(seed)
    train_loader, val_loader, classes = build_dataloaders(
        data_dir, batch_size=batch_size, seed=seed, balanced_sampling=balanced_sampling, model_type=model_type,
    )

    model = build_model(model_type, num_classes=len(classes)).to(device)
    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint, map_location=device)
        if checkpoint["classes"] != classes:
            raise ValueError(
                f"checkpoint classes {checkpoint['classes']} don't match dataset classes {classes} -- "
                "fine-tuning requires the same class set the checkpoint was originally trained on"
            )
        if checkpoint.get("model_type", "cnn") != model_type:
            raise ValueError(
                f"checkpoint model_type {checkpoint.get('model_type')!r} != requested {model_type!r}"
            )
        model.load_state_dict(checkpoint["state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    accuracy = evaluate(model, val_loader, device)
    return model, TrainResult(classes=classes, final_val_accuracy=accuracy, model_type=model_type)


def save_checkpoint(model: nn.Module, classes: List[str], path: Path, model_type: str = "cnn") -> None:
    torch.save({"state_dict": model.state_dict(), "classes": classes, "model_type": model_type}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="cnn")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("tile_classifier.pt"))
    args = parser.parse_args()

    model, result = run_training(
        args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        model_type=args.model_type, init_checkpoint=args.init_checkpoint,
    )
    save_checkpoint(model, result.classes, args.out, model_type=args.model_type)
    print(f"Final validation accuracy: {result.final_val_accuracy:.3f}")
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
