"""Runs a trained rack-tile detector on an image and draws its
predictions -- for sanity-checking a freshly trained checkpoint (does it
detect anything sensible at all?), and as a preview of the shape the
eventual `board_reader.py` rack-reading integration will consume:
detected boxes get cropped and handed to the existing
`TileClassifierModel` (the detector only needs to *localize* tiles, not
read them -- the classifier already does that).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np


def load_model(checkpoint_path: Path):
    from rfdetr import from_checkpoint
    return from_checkpoint(str(checkpoint_path))


def detect_tiles(model, image_path: Path, threshold: float = 0.3):
    """Returns a `supervision.Detections` for the image -- `.xyxy`,
    `.class_id`, `.confidence` arrays, one entry per detected tile."""
    return model.predict(str(image_path), threshold=threshold)


def draw_detections(image: np.ndarray, detections, class_names: Sequence[str]) -> np.ndarray:
    """Draws every detection's box and `label confidence` text onto a
    copy of `image` (BGR, as read by `cv2.imread`) and returns it --
    doesn't touch the input array."""
    annotated = image.copy()
    for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
        x1, y1, x2, y2 = (int(v) for v in box)
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated, f"{label} {confidence:.2f}", (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    detections = detect_tiles(model, args.image, args.threshold)
    image = cv2.imread(str(args.image))
    annotated = draw_detections(image, detections, model.class_names)
    cv2.imwrite(str(args.out), annotated)
    print(f"detected {len(detections.xyxy)} tiles, wrote {args.out}")


if __name__ == "__main__":
    main()
