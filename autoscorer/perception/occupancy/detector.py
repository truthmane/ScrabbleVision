"""Classical (non-ML) per-cell occupancy detection: is there a tile on
this cell, or not? Deliberately not a learned model -- an empty cell is a
near-uniform patch of the known board color, while an occupied cell has a
printed letter/edges, so simple pixel-difference-from-reference and local
gradient strength are enough, and are far more debuggable than a CNN for
this sub-problem (per the architecture plan).
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from autoscorer.gamelogic.board import BOARD_SIZE, Coord
from autoscorer.perception.calibration.homography import crop_cell

DEFAULT_DIFF_THRESHOLD = 25.0
DEFAULT_GRADIENT_THRESHOLD = 15.0


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
    reference_board: np.ndarray,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
) -> Dict[Coord, bool]:
    """Occupancy for every one of the 225 cells, given the current
    rectified board image and an empty-board reference captured at
    calibration time (same camera, same rectification)."""
    occupancy: Dict[Coord, bool] = {}
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            current_cell = crop_cell(rectified_board, row, col)
            reference_cell = crop_cell(reference_board, row, col)
            occupancy[(row, col)] = is_occupied(
                current_cell, reference_cell, diff_threshold, gradient_threshold,
            )
    return occupancy
