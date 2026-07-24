"""Bridges the perception layer (calibration, occupancy, classification) to
the existing camera-independent game logic. `read_board` turns a raw camera
frame into per-cell observations; `partition_observations` then splits
those into what's safe to submit automatically versus what a human must
resolve first -- reusing exactly the same `new_tiles` shape a human
operator already types into the manual-entry form (api.session.NewTile).

Key correctness point (see training/synth_render/tile_renderer.py for the
full rationale): a detected "BLANK" has no letter the image can reveal --
a physical blank tile carries no printed glyph even once played. Our own
`Tile`/`BoardState` model already refuses to place a letter-less blank on
the board (see board.py's `with_placements`), so a detected blank can
never be auto-submitted, in ANY publish mode -- it always needs an
operator to say what it's being played as. This isn't a confidence
threshold decision, it's a structural one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from autoscorer.api.session import NewTile
from autoscorer.gamelogic.board import BOARD_SIZE, Coord
from autoscorer.perception.calibration.homography import BoardCalibration, crop_cell
from autoscorer.perception.occupancy.detector import detect_occupancy
from training.classify.infer import TileClassifierModel


@dataclass(frozen=True)
class CellObservation:
    coord: Coord
    letter: Optional[str]  # None means "detected a blank tile, letter unknown"
    is_blank: bool
    confidence: float


def read_board(
    raw_frame: np.ndarray,
    calibration: BoardCalibration,
    reference_board: np.ndarray,
    classifier: TileClassifierModel,
) -> List[CellObservation]:
    """One observation per occupied cell (never for empty ones -- those
    are filtered by occupancy detection before the classifier ever runs).
    """
    rectified = calibration.rectify(raw_frame)
    occupancy = detect_occupancy(rectified, reference_board)

    observations = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if not occupancy[(row, col)]:
                continue
            # crop_cell/rectify operate on OpenCV-convention BGR arrays
            # throughout the perception layer (see occupancy/detector.py),
            # but the classifier was trained on plain RGB tile renders --
            # convert at this boundary or every prediction is subtly wrong.
            crop = crop_cell(rectified, row, col)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            label, confidence = classifier.predict(crop_rgb)
            is_blank = label == "BLANK"
            letter = None if is_blank else label
            observations.append(CellObservation(coord=(row, col), letter=letter, is_blank=is_blank, confidence=confidence))
    return observations


@dataclass(frozen=True)
class PartitionedObservations:
    submittable: List[NewTile]  # safe to hand to GameSession.submit_move directly
    needs_operator_input: List[CellObservation]  # blanks, and/or low-confidence reads


def partition_observations(
    observations: List[CellObservation],
    confidence_threshold: float = 0.9,
) -> PartitionedObservations:
    """Splits detected cells into what can flow straight into
    `GameSession.submit_move` versus what an operator must resolve first.

    Any single low-confidence or blank cell forces the *whole* move to
    operator review, not just that cell -- submitting a partial placement
    would leave a gap in the word and fail placement validation anyway, so
    there's no useful partial-submit here.
    """
    needs_input = [
        obs for obs in observations if obs.is_blank or obs.confidence < confidence_threshold
    ]
    if needs_input:
        return PartitionedObservations(submittable=[], needs_operator_input=observations)

    submittable = [(obs.coord, obs.letter, obs.is_blank) for obs in observations]
    return PartitionedObservations(submittable=submittable, needs_operator_input=[])
