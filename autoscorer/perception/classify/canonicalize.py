"""Canonicalizes a tile crop before it ever reaches the classifier.

This is the fix for the single biggest bug found in the first fine-tuning
pass (see docs/classifier-accuracy-plan.md, WS1): the synthetic renderer
draws dark glyphs on cream tiles, but both real tournament tile sets
observed so far are the opposite (white-on-black, white-on-red) -- an
inverted-polarity prior so bad the synthetic-only baseline scored *below
random chance* on real tiles. Rather than trying to render every possible
tile color/font combination synthetically, this collapses every style down
to one canonical representation the model actually has to learn: a
single-channel, contrast-normalized, dark-glyph-on-light-background image.
This also future-proofs against tile sets no one has captured data for yet.

Used identically at training time (via `canonicalize_pil` as a torchvision
transform) and inference time (`canonicalize`) -- they must never diverge,
or the model sees a different input distribution live than it was trained
on.
"""
from __future__ import annotations

from typing import Union

import cv2
import numpy as np
from PIL import Image

FINAL_SIZE = 60  # must match training.classify.model.INPUT_SIZE
MARGIN_FRAC = 0.08  # crop this fraction off each edge before resizing


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _auto_crop_to_tile(gray: np.ndarray) -> np.ndarray:
    """Finds the tile's actual extent within a possibly-loose input crop and
    re-frames tightly around it, so a generously-bounded crop (this matters
    most for rack tiles: manual per-group bounding boxes are far less
    precise than the board's calibrated per-cell homography, and measuring
    board-only vs. rack-only accuracy on the same model showed a ~4x gap
    traceable to exactly this framing mismatch) doesn't leave the glyph
    tiny and off-center relative to what training data looks like.

    Falls back to the original image untouched if no clear tile-sized
    contour is found (a tightly-bounded input, e.g. from `crop_cell`, will
    typically find the tile filling the whole frame and this is a no-op).
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = gray.shape
    frame_area = h * w

    for candidate in (thresh, 255 - thresh):
        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        x, y, cw, ch = cv2.boundingRect(biggest)
        bbox_area = cw * ch
        aspect = cw / ch if ch else 0
        extent = area / bbox_area if bbox_area else 0

        # A tile boundary is a filled, roughly-square rectangle (extent
        # close to 1.0). A glyph's contour is not: even a "solid" letter
        # like O or D has low extent relative to its own bounding box, and
        # letters like I/L/1 have extreme aspect ratios. Without this check,
        # an already-tightly-bound crop (e.g. from crop_cell) gets its
        # glyph mistaken for "the tile" and cropped even tighter -- which
        # measurably made real-tile accuracy worse, not better.
        plausible_tile = extent > 0.75 and 0.6 < aspect < 1.6
        big_enough = 0.15 * frame_area < area < 0.85 * frame_area
        if not (plausible_tile and big_enough):
            continue

        margin_x, margin_y = int(cw * 0.08), int(ch * 0.08)
        x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
        x2, y2 = min(w, x + cw + margin_x), min(h, y + ch + margin_y)
        if x2 - x1 > 4 and y2 - y1 > 4:
            return gray[y1:y2, x1:x2]

    return gray


def _normalize_polarity(gray: np.ndarray) -> np.ndarray:
    """Otsu-split the crop into two intensity clusters and assume the
    smaller-area one is the printed glyph (ink strokes cover less area than
    the tile face, regardless of font or color scheme). If that glyph
    cluster is the *brighter* one, the tile is light-glyph-on-dark and gets
    inverted to the canonical dark-glyph-on-light polarity.

    A blank tile has no real glyph cluster -- Otsu's split becomes close to
    arbitrary on a near-uniform patch, but that's harmless: a near-uniform
    patch inverted is still a near-uniform patch, and training sees the
    same ambiguity, so the class stays learnable as "no glyph texture"
    regardless of which way it happened to flip.
    """
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_fraction = float((mask == 255).mean())
    if bright_fraction < 0.5:
        return 255 - gray
    return gray


def canonicalize(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Returns a FINAL_SIZE x FINAL_SIZE uint8 single-channel array:
    contrast-normalized, dark-glyph-on-light-background, margins trimmed.
    Accepts an RGB/RGBA numpy array, a grayscale numpy array, or a PIL Image.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    gray = _to_gray(image)
    gray = _auto_crop_to_tile(gray)

    h, w = gray.shape
    m = int(min(h, w) * MARGIN_FRAC)
    if h - 2 * m > 4 and w - 2 * m > 4:
        gray = gray[m:h - m, m:w - m]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)

    normalized = _normalize_polarity(normalized)

    return cv2.resize(normalized, (FINAL_SIZE, FINAL_SIZE), interpolation=cv2.INTER_AREA)


def canonicalize_pil(image: Image.Image) -> Image.Image:
    """torchvision-transform-friendly wrapper: PIL in, PIL ('L' mode) out."""
    return Image.fromarray(canonicalize(image), mode="L")
