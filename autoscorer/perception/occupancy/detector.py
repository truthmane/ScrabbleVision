"""Classical (non-ML) per-cell occupancy detection: is there a tile on
this cell, or not? Deliberately not a learned model -- an empty cell is a
near-uniform patch of the known board color, while an occupied cell has a
printed letter/edges, so simple pixel-difference-from-reference and local
gradient strength are enough, and are far more debuggable than a CNN for
this sub-problem (per the architecture plan).

Real broadcast footage found a wrinkle a single static reference can't
capture: a venue's board can have more than one genuinely valid *empty*
appearance -- WESPA Word Wars' center square sometimes shows a plain
premium-square color and sometimes a sponsor-logo graphic overlay,
neither of which means a tile is there, but a single fixed reference photo
only ever captures one of those states and reads the other as spuriously
"occupied" (see `configs/venues/wespa_word_wars.json`'s notes for exactly
how this was found: 61 false detections on that one cell across one real
clip). `reference_board` therefore accepts either one image or a sequence
of them -- a cell only counts as occupied if it looks different from
*every* known-valid empty state, not just one.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Union

import cv2
import numpy as np

from autoscorer.gamelogic.board import BOARD_SIZE, Coord
from autoscorer.perception.calibration.homography import crop_cell

ReferenceBoard = Union[np.ndarray, Sequence[np.ndarray]]


def _as_reference_list(reference_board: ReferenceBoard) -> List[np.ndarray]:
    if isinstance(reference_board, np.ndarray):
        return [reference_board]
    references = list(reference_board)
    if not references:
        raise ValueError("reference_board must have at least one reference image")
    return references

DEFAULT_DIFF_THRESHOLD = 25.0
# Calibrated against clean synthetic renders (near-zero background texture),
# this used to be 15.0 -- which fired on every single cell of a real photo,
# 0% precision, confirmed by an end-to-end test against real broadcast
# footage: genuinely empty real cells scored 47-68 gradient just from board
# printing/JPEG noise, already above that threshold, while occupied cells
# scored 116-128. The `diff` signal turned out to be the reliable one on
# real data (empty ~0.03-0.09 vs. occupied 56-86, a huge margin) -- `diff`
# alone would have worked; this raised threshold just stops `gradient` from
# uselessly firing on everything rather than removing it as a signal.
# Per-venue calibration (see the architecture plan) should still re-tune
# this against that venue's own empty-board reference rather than trusting
# this default blindly.
DEFAULT_GRADIENT_THRESHOLD = 100.0


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def occupancy_scores(cell_crop: np.ndarray, reference_crop: np.ndarray) -> Dict[str, float]:
    """Two complementary signals, higher = more likely occupied:
    - `diff`: mean absolute pixel difference from the known-empty reference
      (catches a tile that changes the cell's overall brightness/color).
    - `gradient`: local edge/texture strength within the current crop alone
      (catches a tile even if its average brightness happens to be close
      to the empty square's, since a printed glyph still has sharp edges
      an empty square doesn't).
    """
    current_gray = _to_gray_float(cell_crop)
    reference_gray = _to_gray_float(reference_crop)

    diff_score = float(np.abs(current_gray - reference_gray).mean())

    gx = cv2.Sobel(current_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(current_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_score = float(np.sqrt(gx ** 2 + gy ** 2).std())

    return {"diff": diff_score, "gradient": gradient_score}


def is_occupied(
    cell_crop: np.ndarray,
    reference_crop: np.ndarray,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
) -> bool:
    scores = occupancy_scores(cell_crop, reference_crop)
    return scores["diff"] > diff_threshold or scores["gradient"] > gradient_threshold


def detect_occupancy(
    rectified_board: np.ndarray,
    reference_board: ReferenceBoard,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
) -> Dict[Coord, bool]:
    """Occupancy for every one of the 225 cells, given the current
    rectified board image and one or more empty-board references captured
    at calibration time (same camera, same rectification).

    `reference_board` accepts a single image (the common case) or a
    sequence of them, for a venue where "empty" genuinely has more than
    one valid appearance (see the module docstring) -- a cell counts as
    occupied only if it looks different from *every* reference given, not
    just the first/only one.
    """
    references = _as_reference_list(reference_board)
    occupancy: Dict[Coord, bool] = {}
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            current_cell = crop_cell(rectified_board, row, col)
            occupancy[(row, col)] = all(
                is_occupied(current_cell, crop_cell(reference, row, col), diff_threshold, gradient_threshold)
                for reference in references
            )
    return occupancy
