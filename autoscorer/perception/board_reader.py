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
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from autoscorer.api.session import NewTile
from autoscorer.gamelogic.board import BOARD_SIZE, BoardState, Coord, Tile
from autoscorer.gamelogic.movedetect.constraint_decoder import CellCandidates
from autoscorer.gamelogic.movedetect.temporal_vote import temporal_vote
from autoscorer.perception.calibration.homography import BoardCalibration, crop_cell
from autoscorer.perception.occupancy.detector import (
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_GRADIENT_THRESHOLD,
    ReferenceBoard,
    detect_occupancy,
)
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
    reference_board: ReferenceBoard,
    classifier: TileClassifierModel,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
) -> List[CellObservation]:
    """One observation per occupied cell (never for empty ones -- those
    are filtered by occupancy detection before the classifier ever runs).

    `diff_threshold`/`gradient_threshold` default to the values calibrated
    against the original NASPA broadcast (see occupancy/detector.py) --
    **do not trust those defaults for a new venue.** A venue with busier
    premium-square graphics (printed "DOUBLE LETTER SCORE" text, e.g. the
    WESPA broadcast) can score real empty cells well above these defaults
    on both signals, causing the classifier to fire on dozens of spurious
    "occupied" cells every call. Pull the tuned values from that venue's
    `VenueProfile` instead (see perception/calibration/venue_profile.py).
    """
    rectified = calibration.rectify(raw_frame)
    occupancy = detect_occupancy(rectified, reference_board, diff_threshold, gradient_threshold)

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


def read_new_cells(
    raw_frame: np.ndarray,
    calibration: BoardCalibration,
    reference_board: ReferenceBoard,
    classifier: TileClassifierModel,
    board_before: BoardState,
    top_k: int = 3,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
) -> List[CellCandidates]:
    """Like `read_board`, but scoped to exactly one turn's new tiles (cells
    occupied in this frame but empty in `board_before`) and returning each
    cell's top-k classifier candidates rather than just its top-1 -- the
    shape `constraint_decoder.decode_feasible_reading` needs to fall back
    to a globally-feasible reading when the top guess isn't one.

    This is the per-turn diffing the constraint decoder was missing (see
    the WS4 section of docs/classifier-accuracy-plan.md): `read_board`
    reads the whole board every call, which only makes sense for a
    one-shot read, not for scoring what changed on a single turn.

    See `read_board`'s docstring on `diff_threshold`/`gradient_threshold` --
    the defaults are NASPA-specific; use a venue's own tuned values.
    """
    rectified = calibration.rectify(raw_frame)
    occupancy = detect_occupancy(rectified, reference_board, diff_threshold, gradient_threshold)

    results = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            coord = (row, col)
            if not occupancy[coord] or not board_before.is_empty(coord):
                continue
            crop = crop_cell(rectified, row, col)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            candidates = classifier.predict_topk(crop_rgb, k=top_k)
            results.append(CellCandidates(coord=coord, candidates=candidates))
    return results


def read_new_cells_voted(
    raw_frames: Sequence[np.ndarray],
    calibration: BoardCalibration,
    reference_board: ReferenceBoard,
    classifier: TileClassifierModel,
    board_before: BoardState,
    top_k: int = 3,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
) -> List[CellCandidates]:
    """Like `read_new_cells`, but combines classifier readings across
    several frames of the *same* stable board moment via
    `temporal_vote` before returning candidates -- the M2 lever from
    docs/classifier-accuracy-plan.md ("per-tile after temporal voting
    over >=5 stable frames"). A tile misread in one frame (glare, motion
    blur, a bad angle) gets outvoted by the frames that read it correctly,
    which raw single-frame confidence has no way to recover from.

    Assumes `raw_frames` all show the same board state -- that's what a
    stillness gate upstream (not built here; see the master architecture
    plan's move-detection state machine) is responsible for guaranteeing.

    Each cell gets a **support tier** from its vote count across the
    window rather than a hard unanimous AND: `votes == len(raw_frames)` is
    HARD (identical to the original all-frames-agree behaviour -- see
    `CellCandidates.is_soft`, False here); `0 < votes < len(raw_frames)` is
    SOFT (`is_soft=True`). Unanimity was added because a hand hovering
    near (not over) the board can be too small a change for the coarse
    whole-frame stillness gate to reject, yet still nudge one frame's diff
    score for a specific cell across the occupancy threshold -- checking
    only the first frame treated that one noisy frame as ground truth for
    the whole window, and requiring unanimous agreement fixed it. But real
    full-game footage also showed the opposite failure: a genuine tile
    (WESPA "RAGBOLT"'s final T) crossed the occupancy threshold in *some*
    frames of its window but not all (diff 38.96 against a 38.0
    threshold, a bare margin), so a hard AND silently dropped a real tile
    rather than just noise.

    **A SOFT cell is only ever surfaced if it's orthogonally adjacent to a
    HARD cell in this same window** (not "anywhere `votes > 0` on the
    board"). Confirmed against the real WESPA broadcast: any-vote-at-all
    is far too permissive on real noisy footage -- ordinary compression/
    lighting flicker nudges plenty of genuinely empty cells scattered
    across the whole 225-cell board over threshold in a single frame out
    of the window, exactly the class of noise unanimity was built to
    reject in the first place, and each one classified across every frame
    at `top_k=len(classifier.classes)` (see this docstring's cost warning
    below) turned a several-minute real-video run into 30+ minutes with
    no corresponding accuracy gain. Restricting SOFT to HARD's immediate
    neighbors keeps the RAGBOLT-style fix (a real tile just outside an
    already-detected run) while bounding the extra cost to a small
    multiple of the genuine new-tile count, not the whole board. A cell
    with partial support far from any HARD cell is, correctly, still
    invisible here -- it can never anchor a placement on its own (see
    `GameWatcher._soft_cells`), so there was never anything to gain by
    classifying it.

    See `read_board`'s docstring on `diff_threshold`/`gradient_threshold` --
    **getting these wrong here is expensive, not just inaccurate**: every
    cell occupancy spuriously flags as "new" gets classified across all
    `len(raw_frames)` frames at `top_k=len(classifier.classes)`, so loose
    thresholds on a venue with busy premium squares can turn a handful of
    real new cells into dozens of spurious ones and multiply the
    classifier calls per observation accordingly.
    """
    if not raw_frames:
        raise ValueError("read_new_cells_voted needs at least one frame")

    num_classes = len(classifier.classes)
    rectified_frames = [calibration.rectify(frame) for frame in raw_frames]
    per_frame_occupancy = [
        detect_occupancy(rectified, reference_board, diff_threshold, gradient_threshold)
        for rectified in rectified_frames
    ]
    num_frames = len(rectified_frames)
    votes = {
        coord: sum(1 for occ in per_frame_occupancy if occ[coord])
        for coord in per_frame_occupancy[0]
    }

    hard_cells = {
        (row, col)
        for row in range(BOARD_SIZE)
        for col in range(BOARD_SIZE)
        if votes[(row, col)] == num_frames and board_before.is_empty((row, col))
    }
    soft_neighbors = set()
    for row, col in hard_cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (row + dr, col + dc)
            if (
                0 <= neighbor[0] < BOARD_SIZE and 0 <= neighbor[1] < BOARD_SIZE
                and neighbor not in hard_cells
                and 0 < votes[neighbor] < num_frames
                and board_before.is_empty(neighbor)
            ):
                soft_neighbors.add(neighbor)
    new_cells = sorted(hard_cells | soft_neighbors)
    if not new_cells:
        return []

    # One batched forward pass for every (cell, frame) crop instead of a
    # separate classifier call each -- mathematically identical results
    # (see TileClassifierModel.predict_topk_batch's docstring), just
    # sharply fewer, larger forward passes. Every frame in the window is
    # classified regardless of that frame's own vote -- a SOFT cell's
    # occupancy *score* fluctuated near threshold, but the physical scene
    # (tile or no tile) is the same settled moment in every frame, so
    # there's no basis to exclude any frame's crop from the vote.
    crops = [
        cv2.cvtColor(crop_cell(rectified, row, col), cv2.COLOR_BGR2RGB)
        for (row, col) in new_cells
        for rectified in rectified_frames
    ]
    all_candidates = classifier.predict_topk_batch(crops, k=num_classes)

    results = []
    for i, coord in enumerate(new_cells):
        per_frame_candidates = all_candidates[i * num_frames:(i + 1) * num_frames]
        voted = temporal_vote(per_frame_candidates)[:top_k]
        results.append(CellCandidates(coord=coord, candidates=voted, is_soft=votes[coord] < num_frames))
    return results


@dataclass(frozen=True)
class RackTileObservation:
    letter: Optional[str]  # None means "detected a blank tile, letter unknown"
    is_blank: bool
    confidence: float
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2 in rack_frame's pixel space


def read_rack(
    rack_frame: np.ndarray,
    rack_detector,
    classifier: TileClassifierModel,
    detection_threshold: float = 0.3,
) -> List[RackTileObservation]:
    """Reads every tile currently on a rack camera frame: `rack_detector`
    (an RF-DETR model, e.g. loaded via `training.detect.visualize_rack_detections.load_model`)
    localizes individual tiles -- real rack tiles aren't evenly spaced like
    board cells, so there's no fixed grid to crop against -- then each
    detected box is cropped and handed to the same `TileClassifierModel`
    used for board cells. The detector only needs to find *where* the
    tiles are; classification stays the classifier's job, since it's the
    more mature, better-calibrated component -- and measurably the better
    reader: on the two held-out photos, RF-DETR's own classification head
    gets 12/14 while routing the same boxes through the classifier gets
    14/14 (see training/detect/README.md for the full comparison and the
    provenance caveats on each photo).

    Results are returned **left-to-right by box position**, not in the
    detector's own (confidence-ordered) output order -- a rack only reads
    as a rack in spatial order, and every caller that reconstructs rack
    contents for display or operator review would otherwise have to
    re-sort. Pool accounting doesn't care about order, but nothing else
    should have to know that.

    `rack_frame` and each crop follow the same BGR (OpenCV) convention as
    `read_board`/`read_new_cells` -- both the detector and the classifier
    expect RGB, so both get converted at this boundary, not before.
    """
    rgb_frame = cv2.cvtColor(rack_frame, cv2.COLOR_BGR2RGB)
    detections = rack_detector.predict(rgb_frame, threshold=detection_threshold)

    observations = []
    for box in detections.xyxy:
        x1, y1, x2, y2 = (int(v) for v in box)
        crop_rgb = rgb_frame[y1:y2, x1:x2]
        label, confidence = classifier.predict(crop_rgb)
        is_blank = label == "BLANK"
        letter = None if is_blank else label
        observations.append(
            RackTileObservation(letter=letter, is_blank=is_blank, confidence=confidence, box=(x1, y1, x2, y2))
        )
    return sorted(observations, key=lambda obs: obs.box[0])


def rack_observations_to_tiles(observations: Sequence[RackTileObservation]) -> List[Tile]:
    """Converts detected rack tiles into the `Tile` shape
    `constraint_decoder.decode_feasible_reading`'s `racks` parameter and
    `bag_engine`'s pool derivation expect. A detected blank's `letter` is
    left `None` (unplayed blanks never carry a letter -- see `Tile`'s own
    validation in board.py) -- pool accounting only needs the `is_blank`
    flag, never which letter a low-confidence guess attached to it.
    """
    return [Tile(letter=obs.letter, is_blank=obs.is_blank) for obs in observations]


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
