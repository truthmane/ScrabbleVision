"""Inference wrapper around a trained per-cell letter classifier
checkpoint -- what the perception bridge actually calls per occupied cell.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

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

    def predict_topk_batch(
        self, images: Sequence[Union[Image.Image, np.ndarray]], k: int = 3,
    ) -> List[List[Tuple[str, float]]]:
        """Batched `predict_topk`: classifies many crops in one forward
        pass instead of one at a time. Numerically identical to calling
        `predict_topk` on each image separately -- the model runs in eval
        mode (`self.model.eval()`, set in `__init__`), so BatchNorm uses
        its stored running statistics rather than the current batch's,
        meaning one image's result never depends on what else is in the
        batch. This is where nearly all of the live pipeline's per-
        settled-frame cost lives (`board_reader.read_new_cells_voted`
        calls this once per occupied cell per window frame), so batching
        directly targets the ~5s/settled-frame bottleneck noted in
        `game_watcher.py`'s and the README's scope notes.
        """
        if not images:
            return []
        pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
        tensor = torch.stack([self._transform(img.convert("RGB")) for img in pil_images]).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits / self.temperature, dim=1)
        k = min(k, len(self.classes))
        results = []
        for row_probs in probs:
            top_probs, top_idx = torch.topk(row_probs, k)
            results.append([(self.classes[i], float(p)) for p, i in zip(top_probs, top_idx)])
        return results
