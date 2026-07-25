"""Reconstructs tile bounding boxes for a real rack photo from the same
(x1, x2, letters) per-group description `training/collect/crop_rack.py`
already uses -- i.e. it turns the manual grouping decision a collector
already made (which contiguous clusters of tiles are which letters) into
detection-training boxes, instead of just individual tile crops. This is
how the real rack photos harvested for WS3 become bounding-box training
data at effectively zero extra manual-annotation cost: the labor of
deciding tile groups was already spent.

The vertical extent of each box isn't manually specified -- `crop_rack.py`
never needed it, since it only produced per-tile PNGs using the full
image height per column. `detect_ledge_top` locates the top of the wooden
rack ledge by color (a warm, distinct wood tone against the cooler/darker
background and the black tile faces), and a fixed offset above/below that
line approximates where a resting tile sits. This is a coarse heuristic
shared across every tile in one photo (not a per-tile segmenter) --
reasonable for a modest amount of real fine-tuning data blended into a
much larger synthetic set (see `rack_scene_renderer.py`), not meant to be
the sole source of ground truth.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from training.synth_render.rack_scene_renderer import TileBox

# Tuned against real photos from the 2026 NASPA broadcast (see
# docs/classifier-accuracy-plan.md's WS3 section): tiles typically extend
# ~33px above and ~52px below the detected ledge-top line in a 900x150
# rack crop. Re-tune per venue/camera, same as every other perception
# threshold in this codebase.
DEFAULT_ABOVE_LEDGE = 33
DEFAULT_BELOW_LEDGE = 52


def detect_ledge_top(
    image: np.ndarray,
    wood_r_minus_b: int = 30,
    min_wood_fraction: float = 0.3,
) -> Optional[int]:
    """The topmost row where a majority of pixels look like the warm wood
    rack ledge color (red channel notably above blue) -- everything above
    this line is background or a tile, everything at/below it is the bare
    ledge (except where a tile occludes it in front). Returns `None` if no
    row meets `min_wood_fraction` (e.g. a photo with no visible wood, or a
    non-wood rack) -- callers should fall back to an explicit y-band.
    """
    b, g, r = cv2.split(image.astype(np.int16))
    wood_mask = ((r - b) > wood_r_minus_b) & (r > 80)
    row_fraction = wood_mask.mean(axis=1)
    wood_rows = np.where(row_fraction > min_wood_fraction)[0]
    if len(wood_rows) == 0:
        return None
    return int(wood_rows.min())


def reconstruct_boxes_from_groups(
    image: np.ndarray,
    groups: Sequence[Tuple[int, int, List[str]]],
    y_band: Optional[Tuple[int, int]] = None,
) -> List[TileBox]:
    """`groups` is exactly `crop_rack.py`'s format: one (x1, x2, letters)
    tuple per contiguous cluster of tiles, evenly divided within each
    group. `y_band`, if given, overrides the auto-detected ledge-relative
    band -- needed on photos where the ledge heuristic doesn't apply
    (e.g. a single tile far from the rest of the frame's visible wood).
    """
    height = image.shape[0]
    if y_band is None:
        ledge_top = detect_ledge_top(image)
        if ledge_top is None:
            raise ValueError("could not detect the rack ledge; pass an explicit y_band")
        y1 = max(0, ledge_top - DEFAULT_ABOVE_LEDGE)
        y2 = min(height, ledge_top + DEFAULT_BELOW_LEDGE)
    else:
        y1, y2 = y_band

    boxes: List[TileBox] = []
    for x1, x2, letters in groups:
        step = (x2 - x1) / len(letters)
        for i, letter in enumerate(letters):
            slot_x1, slot_x2 = int(x1 + i * step), int(x1 + (i + 1) * step)
            boxes.append(TileBox(label=letter.upper(), x1=slot_x1, y1=y1, x2=slot_x2, y2=y2))
    return boxes
