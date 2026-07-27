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

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

from autoscorer.gamelogic.board import BOARD_SIZE, Coord
from autoscorer.perception.calibration.homography import crop_cell_inset

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
    """Four complementary signals, higher = more likely occupied:
    - `diff`: mean absolute pixel difference from the known-empty reference
      (catches a tile that changes the cell's overall brightness/color).
    - `gradient`: local edge/texture strength within the current crop alone
      (catches a tile even if its average brightness happens to be close
      to the empty square's, since a printed glyph still has sharp edges
      an empty square doesn't).
    - `edge_diff`: mean absolute difference between the current and
      reference crops' own Sobel-magnitude maps -- unlike `gradient`
      (computed from the current crop alone, so a busy static graphic
      scores high purely from its own texture, indistinguishable from a
      real tile's edges), this compares edges to the reference: a static
      graphic has near-identical edges in both, so it scores low here
      even though its `gradient` score is high; a real tile adds edges
      the reference never had.
    - `ncc`: `1 - normalized cross-correlation` against the reference --
      NCC is invariant to a uniform brightness/gain shift (unlike `diff`,
      a plain mean-pixel-difference), so this is a good complementary
      signal specifically under lighting drift across a long broadcast,
      where `diff` alone can creep upward on a still-genuinely-empty cell.
      Inverted (`1 - correlation`) so it obeys the same "higher = more
      occupied" convention as every other signal here: identical crops
      correlate near 1.0, giving a score near 0. NCC is undefined when a
      crop has essentially zero variance (a perfectly flat patch -- a
      real photo almost always has some sensor noise, but a synthetic
      render or a saturated/blown-out real capture can be perfectly
      flat), and OpenCV's `TM_CCOEFF_NORMED` returns a degenerate 1.0 in
      that case rather than raising -- silently reporting "identical" for
      a flat reference against ANY current crop, occupied or not. Falls
      back to a diff-based dissimilarity (still on a comparable 0-1ish
      scale) whenever either crop's standard deviation is negligible.
    """
    current_gray = _to_gray_float(cell_crop)
    reference_gray = _to_gray_float(reference_crop)

    diff_score = float(np.abs(current_gray - reference_gray).mean())

    gx = cv2.Sobel(current_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(current_gray, cv2.CV_32F, 0, 1, ksize=3)
    current_magnitude = np.sqrt(gx ** 2 + gy ** 2)
    gradient_score = float(current_magnitude.std())

    ref_gx = cv2.Sobel(reference_gray, cv2.CV_32F, 1, 0, ksize=3)
    ref_gy = cv2.Sobel(reference_gray, cv2.CV_32F, 0, 1, ksize=3)
    reference_magnitude = np.sqrt(ref_gx ** 2 + ref_gy ** 2)
    edge_diff_score = float(np.abs(current_magnitude - reference_magnitude).mean())

    if current_gray.std() < 1e-3 or reference_gray.std() < 1e-3:
        ncc_score = min(1.0, diff_score / 255.0)
    else:
        correlation = float(cv2.matchTemplate(current_gray, reference_gray, cv2.TM_CCOEFF_NORMED)[0, 0])
        ncc_score = 1.0 - correlation

    return {"diff": diff_score, "gradient": gradient_score, "edge_diff": edge_diff_score, "ncc": ncc_score}


@dataclass(frozen=True)
class OccupancyThresholds:
    """Bundles every occupancy-signal threshold in one place, so adding a
    signal doesn't mean adding another positional float parameter to
    `is_occupied_multi`/every one of its callers. `edge_diff`/`ncc`
    default to `None` (disabled) -- only `diff`/`gradient` have ever been
    tuned against a real venue (see `configs/venues/wespa_word_wars.json`'s
    notes); a venue opts into the newer signals explicitly by setting a
    threshold for them, rather than getting them on by surprise with an
    untuned default."""
    diff: float = DEFAULT_DIFF_THRESHOLD
    gradient: float = DEFAULT_GRADIENT_THRESHOLD
    edge_diff: Optional[float] = None
    ncc: Optional[float] = None


def is_occupied_multi(scores: Dict[str, float], thresholds: OccupancyThresholds) -> bool:
    """Combines every signal in `scores` (as returned by `occupancy_scores`)
    against `thresholds`: occupied if ANY enabled signal exceeds its own
    threshold. `diff`/`gradient` are always checked (matching
    `is_occupied`'s existing behavior exactly when `edge_diff`/`ncc` are
    left at their default `None`); `edge_diff`/`ncc` are checked only if
    their threshold is set."""
    if scores["diff"] > thresholds.diff or scores["gradient"] > thresholds.gradient:
        return True
    if thresholds.edge_diff is not None and scores["edge_diff"] > thresholds.edge_diff:
        return True
    if thresholds.ncc is not None and scores["ncc"] > thresholds.ncc:
        return True
    return False


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
            # Inset, not the classifier's full-cell `crop_cell` -- see
            # `crop_cell_inset`'s docstring: trims the grid line/neighbor
            # bleed at the cell boundary before scoring, which never
            # affects what the classifier itself is handed elsewhere.
            current_cell = crop_cell_inset(rectified_board, row, col)
            occupancy[(row, col)] = all(
                is_occupied(current_cell, crop_cell_inset(reference, row, col), diff_threshold, gradient_threshold)
                for reference in references
            )
    return occupancy
