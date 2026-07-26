"""A stand-in classifier for `tests/slow/test_synthetic_full_game.py` that
simulates a real, imperfect classifier's accuracy WITHOUT needing a GPU,
real photos, or training a model at test time.

Identification is nearest-neighbor pixel-space matching against a registry
of every distinct tile crop the test actually rendered (built alongside
the synthetic board frames) -- robust to the tiny numeric differences a
lossless video round-trip and an identity `warpPerspective` can still
introduce, unlike exact byte/hash matching. Given how visually distinct
27 rendered tile classes are, nearest-neighbor identification against a
registry of the test's own renders is effectively exact.

The confusion distribution used to decide WHICH wrong label to assign
(when the coin flip says "wrong") comes from actually running the real
deployed checkpoint (`models/tile_classifier_v1.pt`) against many
synthetic augmented samples per class -- a real, reproducible measurement
using only committed assets (the checkpoint + the synth renderer), not a
hand-guessed confusion table. Its overall accuracy on synthetic data is
higher than on real broadcast photos (that's the whole reason real photos
were needed at all), so this only supplies the *shape* of likely
confusions; the oracle's own `p` parameter controls the actual per-cell
accuracy rate, independent of whatever the checkpoint's synthetic-data
accuracy happens to be.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from training.synth_render.tile_renderer import TILE_CLASSES, augment_tile, render_tile


def build_confusion_distribution(
    checkpoint_path: Union[str, Path], samples_per_class: int = 20, seed: int = 0,
) -> Dict[str, List[Tuple[str, float]]]:
    """For each class, a normalized distribution over the OTHER classes
    the real deployed checkpoint actually confused it with, measured on
    freshly rendered+augmented synthetic samples. A class the checkpoint
    never misclassified in this sample falls back to a uniform
    distribution over the other 26 -- there's no real confusion signal to
    prefer one over another in that case."""
    from training.classify.infer import TileClassifierModel

    classifier = TileClassifierModel(checkpoint_path, device="cpu")
    rng = random.Random(seed)
    confusion: Dict[str, Counter] = {c: Counter() for c in TILE_CLASSES}

    for true_label in TILE_CLASSES:
        letter = None if true_label == "BLANK" else true_label
        base = render_tile(letter, rng=rng)
        crops = [np.array(augment_tile(base, rng=rng).convert("RGB")) for _ in range(samples_per_class)]
        predictions = classifier.predict_topk_batch(crops, k=1)
        for pred in predictions:
            confusion[true_label][pred[0][0]] += 1

    dist: Dict[str, List[Tuple[str, float]]] = {}
    for true_label, counter in confusion.items():
        wrong = {label: count for label, count in counter.items() if label != true_label}
        if not wrong:
            others = [c for c in TILE_CLASSES if c != true_label]
            dist[true_label] = [(c, 1.0) for c in others]
        else:
            total = sum(wrong.values())
            dist[true_label] = [(label, count / total) for label, count in wrong.items()]
    return dist


def _small_gray(image: np.ndarray, size: int = 24) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


class NoisyOracleClassifier:
    """Drop-in replacement for `TileClassifierModel` (same `classes` /
    `predict` / `predict_topk` / `predict_topk_batch` surface) that
    simulates a classifier with per-cell accuracy `p` against a known
    registry of ground-truth crops, instead of running a real model."""

    def __init__(
        self,
        registry: Sequence[Tuple[np.ndarray, str]],
        confusion_dist: Dict[str, List[Tuple[str, float]]],
        p: float,
        rng: random.Random,
        classes: Optional[List[str]] = None,
    ) -> None:
        self.classes = list(classes) if classes is not None else list(TILE_CLASSES)
        self._registry = [(_small_gray(np.asarray(crop)), label) for crop, label in registry]
        self.confusion_dist = confusion_dist
        self.p = p
        self.rng = rng
        self.temperature = 1.0  # TileClassifierModel compatibility; unused here

    def _identify(self, image: Union[Image.Image, np.ndarray]) -> str:
        arr = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else np.asarray(image)
        feat = _small_gray(arr)
        best_label, best_dist = None, float("inf")
        for ref_feat, label in self._registry:
            d = float(np.sum((feat - ref_feat) ** 2))
            if d < best_dist:
                best_dist, best_label = d, label
        if best_label is None:
            raise ValueError("empty registry -- nothing to identify against")
        return best_label

    def _one_prediction(self, image: Union[Image.Image, np.ndarray], k: int) -> List[Tuple[str, float]]:
        true_label = self._identify(image)

        if self.rng.random() < self.p:
            top_label, top_score = true_label, self.rng.uniform(0.6, 0.95)
        else:
            candidates = self.confusion_dist.get(true_label) or [
                (c, 1.0) for c in self.classes if c != true_label
            ]
            labels, weights = zip(*candidates)
            top_label = self.rng.choices(labels, weights=weights, k=1)[0]
            top_score = self.rng.uniform(0.25, 0.6)

        others = [c for c in self.classes if c != top_label]
        raw_weights = [self.rng.random() + 0.01 for _ in others]
        total = sum(raw_weights)
        remaining_mass = max(0.0, 1.0 - top_score)
        distribution = {top_label: top_score}
        for label, weight in zip(others, raw_weights):
            distribution[label] = weight / total * remaining_mass

        ranked = sorted(distribution.items(), key=lambda kv: -kv[1])
        return ranked[:k]

    def predict_topk_batch(
        self, images: Sequence[Union[Image.Image, np.ndarray]], k: int = 3,
    ) -> List[List[Tuple[str, float]]]:
        if not images:
            return []
        return [self._one_prediction(img, k) for img in images]

    def predict_topk(self, image: Union[Image.Image, np.ndarray], k: int = 3) -> List[Tuple[str, float]]:
        return self._one_prediction(image, k)

    def predict(self, image: Union[Image.Image, np.ndarray]) -> Tuple[str, float]:
        return self._one_prediction(image, 1)[0]
