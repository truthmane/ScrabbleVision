"""Inference wrapper around a trained per-cell letter classifier
checkpoint -- what the perception bridge actually calls per occupied cell.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from training.classify.model import PretrainedTileClassifier, TileClassifier
from training.classify.train import canonicalized_transform


class TileClassifierModel:
    def __init__(self, checkpoint_path: Union[str, Path], device: str = "cpu") -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.classes = checkpoint["classes"]
        self.device = torch.device(device)
        model_type = checkpoint.get("model_type", "cnn")
        if model_type == "pretrained":
            self.model = PretrainedTileClassifier(num_classes=len(self.classes)).to(self.device)
        else:
            self.model = TileClassifier(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        # Must stay identical to the transform used at training time (see
        # train.py's canonicalized_transform) -- any divergence here means
        # the model sees a different input distribution live than it was
        # trained on.
        self._transform = canonicalized_transform(model_type)
        # Set by training/classify/calibrate.py; 1.0 (no-op) until a
        # checkpoint has actually been calibrated. See calibrate.py's
        # docstring for why raw softmax confidence shouldn't be trusted
        # as-is by the publish gateway's confidence threshold.
        self.temperature = checkpoint.get("temperature", 1.0)

    def predict(self, image: Union[Image.Image, np.ndarray]) -> Tuple[str, float]:
        """Returns (label, confidence). `label` is 'BLANK' or a single
        uppercase letter -- never a specific letter-a-blank-represents,
        since that can't be determined from the tile image alone."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits / self.temperature, dim=1)[0]
            idx = int(torch.argmax(probs))
        return self.classes[idx], float(probs[idx])

    def predict_topk(self, image: Union[Image.Image, np.ndarray], k: int = 3) -> List[Tuple[str, float]]:
        """Returns the top-k (label, confidence) pairs, most likely first --
        what the constraint decoder needs to fall back to the next-most-
        plausible reading when the top guess is globally infeasible (e.g.
        would require a fourth Q). Not used by the perception bridge's
        default path, only by movedetect/constraint_decoder.py.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits / self.temperature, dim=1)[0]
        k = min(k, len(self.classes))
        top_probs, top_idx = torch.topk(probs, k)
        return [(self.classes[i], float(p)) for p, i in zip(top_probs, top_idx)]
