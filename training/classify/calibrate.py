"""Temperature-scales a trained classifier so confidence actually means
what the PublishGateway's confidence-fallback mode assumes it means: "0.9
confidence" should be correct about 90% of the time, not just be the
largest of 27 softmax outputs. Raw softmax confidence from a freshly
fine-tuned net is well known to be miscalibrated (usually overconfident);
temperature scaling (Guo et al., 2017) is the standard, cheap fix -- a
single learned scalar T divides the logits before softmax. It cannot
change which class wins (argmax is invariant to T), only how much the
winning probability can be trusted.

Caveat worth keeping in mind: ideally the fitting set here is disjoint
from both the training data AND whatever set accuracy is reported against
(a three-way split). Given how little real data exists right now (see
WS3 of docs/classifier-accuracy-plan.md), this script's caller may end up
reusing the same held-out set for both purposes -- acceptable to establish
the mechanism honestly, but worth revisiting with a proper three-way split
once more real data is collected.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import torch
from torch import nn
from PIL import Image

from training.classify.infer import TileClassifierModel


def collect_logits(clf: TileClassifierModel, data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Raw (pre-softmax) logits and integer labels for every image under
    data_dir/<class_name>/*.png, using the model's own class-index mapping.
    """
    class_to_idx = {c: i for i, c in enumerate(clf.classes)}
    logits_list = []
    label_list = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in class_to_idx:
            continue
        true_idx = class_to_idx[class_dir.name]
        for f in sorted(class_dir.glob("*.png")):
            crop = cv2.imread(str(f))
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(crop_rgb)
            tensor = clf._transform(image.convert("RGB")).unsqueeze(0).to(clf.device)
            with torch.no_grad():
                logits = clf.model(tensor)[0]
            logits_list.append(logits)
            label_list.append(true_idx)
    return torch.stack(logits_list), torch.tensor(label_list)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Returns the scalar T > 0 minimizing NLL of softmax(logits/T,
    labels) -- fit via LBFGS. Optimized in log-space (T = exp(log_T))
    rather than directly: an unconstrained LBFGS step on T itself can and
    did drive it negative, which silently flips which class wins under
    softmax (argmax is only invariant to *positive* scaling) -- exactly
    the kind of bug that would corrupt predictions while looking like a
    confidence-only change.
    """
    log_temperature = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / torch.exp(log_temperature), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).item())


def apply_temperature_to_checkpoint(checkpoint_path: Path, temperature: float, out_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["temperature"] = temperature
    torch.save(checkpoint, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("calibration_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="defaults to overwriting checkpoint in place")
    args = parser.parse_args()

    clf = TileClassifierModel(args.checkpoint, device="cpu")
    logits, labels = collect_logits(clf, args.calibration_dir)
    temperature = fit_temperature(logits, labels)
    print(f"fitted temperature: {temperature:.3f}")

    out_path = args.out or args.checkpoint
    apply_temperature_to_checkpoint(args.checkpoint, temperature, out_path)
    print(f"saved calibrated checkpoint to {out_path}")


if __name__ == "__main__":
    main()
