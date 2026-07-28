"""Statistical (non-learned) signal for catching a *missed* blank tile: a
blank tile's face is visually smooth/uniform, while a lettered tile has
ink-stroke contrast -- cheap to compute, needs no training data, and
independent of the CNN classifier's own BLANK class, so it can flag cases
the classifier is confidently wrong about (measured: 2 of 2 held-out real
blank tiles in `training/data/real_tiles`'s Causeway split were
misclassified as a confident "X" by the deployed checkpoint).

Adapted from the coefficient-of-variation heuristic in
github.com/jheidel/scrabble-opencv's `vision/blank_finder.py` ("stage 1" of
its two-stage design -- stage 2, a neighbor Z-score, needs whole-board
context including empty background cells that a single cell crop doesn't
carry, so it isn't reproduced here).

Threshold and separability measured directly against this project's own
`training/data/real_tiles/` (5,880 tiles spanning 27 real venues, no
per-venue tuning): a single global `CV <= 13.49` split gets 72.9% BLANK
recall at a 4.1% false-positive rate on lettered tiles. Not strong enough
to call blanks outright (this project's existing rule that a blank always
needs operator confirmation stays as-is) -- used only as an extra
"something looks off, get a human" trigger alongside that existing rule,
never to silently promote or demote a reading.
"""
from __future__ import annotations

import cv2
import numpy as np

PATCH_FRAC = 0.40
"""Fraction of a cell crop used for the smoothness patch -- matches
scrabble-opencv's own BLANK_PATCH_FRAC, avoiding the tile's outer edge
where grid-line/neighbor bleed would inflate the variance regardless of
whether the tile is blank."""

DEFAULT_CV_THRESHOLD = 13.49
"""Coefficient-of-variation split point measured against real_tiles (see
module docstring). Below this, a patch reads as smooth (blank-like)."""


def patch_coefficient_of_variation(crop_bgr: np.ndarray) -> float:
    """Coefficient of variation (%, stddev/mean*100) of a centered patch of
    `crop_bgr`. Low means visually smooth (blank-like); high means
    textured (glyph-like). `crop_bgr` should be a single cell crop at the
    classifier's own framing (`homography.crop_cell`), since that's what
    the measured threshold was calibrated against."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape
    ph, pw = int(h * PATCH_FRAC), int(w * PATCH_FRAC)
    y0, x0 = (h - ph) // 2, (w - pw) // 2
    patch = gray[y0:y0 + ph, x0:x0 + pw].astype(np.float64)
    mean = patch.mean()
    if mean <= 0:
        return 0.0
    return float(patch.std() / mean * 100)


def looks_smooth_like_a_blank(crop_bgr: np.ndarray, threshold: float = DEFAULT_CV_THRESHOLD) -> bool:
    """True if `crop_bgr` is smooth enough that it might be a blank tile
    the classifier misread as a confident letter -- a trigger for a
    "confirm this isn't a missed blank" operator flag, never a call to
    treat the cell as blank outright (see module docstring)."""
    return patch_coefficient_of_variation(crop_bgr) <= threshold
