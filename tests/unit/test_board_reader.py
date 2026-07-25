import random

import numpy as np
import pytest
import torch

from autoscorer.gamelogic.board import BOARD_SIZE, PREMIUM_SQUARES, BoardState, Tile
from autoscorer.perception.board_reader import (
    partition_observations,
    rack_observations_to_tiles,
    read_board,
    read_new_cells,
    read_new_cells_voted,
    read_rack,
    CellObservation,
    RackTileObservation,
)
from autoscorer.perception.calibration.homography import CANONICAL_SIZE, BoardCalibration, cell_bounds
from training.classify.infer import TileClassifierModel
from training.classify.train import run_training, save_checkpoint
from training.synth_render.tile_renderer import FINAL_SIZE, SQUARE_COLORS, augment_tile, render_tile

IDENTITY_CALIBRATION = BoardCalibration(homography=np.eye(3, dtype=np.float64))


def _blank_board_image() -> np.ndarray:
    img = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            code = PREMIUM_SQUARES.get((row, col), "plain")
            color = SQUARE_COLORS[code]
            x1, y1, x2, y2 = cell_bounds(row, col)
            img[y1:y2, x1:x2] = color[::-1]
    return img


def _place_tile(board_img: np.ndarray, row: int, col: int, letter, rng: random.Random) -> None:
    # render_tile's canvas has black corners outside the rounded tile shape
    # (see tile_renderer.py) -- only augment_tile masks/composites those
    # away onto a proper background. Pasting render_tile's raw output
    # directly would hand the classifier an input shape (black corner
    # triangles) it never saw in training, so route through augment_tile
    # here too, same as every real training sample.
    tile = augment_tile(render_tile(letter, rng=rng), rng=rng)
    tile_arr = np.array(tile)[:, :, ::-1]
    x1, y1, x2, y2 = cell_bounds(row, col)
    board_img[y1:y2, x1:x2] = tile_arr


def _train_tiny_classifier(tmp_path, labels=("A", "N", "T", "BLANK")):
    rng = random.Random(0)
    for label in labels:
        letter = None if label == "BLANK" else label
        class_dir = tmp_path / label
        class_dir.mkdir(parents=True, exist_ok=True)
        base = render_tile(letter, rng=rng)
        for i in range(30):
            augment_tile(base, rng=rng).save(class_dir / f"{i:03d}.png")

    # seed=1: with training now properly deterministic (see train.py's
    # torch.manual_seed fix), seed=0 happens to land in a bad local optimum
    # for this specific tiny 4-class/30-sample setup and reliably collapses
    # to one class -- seed=1 reliably converges to 1.0 val accuracy.
    model, result = run_training(tmp_path, epochs=25, batch_size=8, device=torch.device("cpu"), seed=1)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(model, result.classes, checkpoint_path)
    return TileClassifierModel(checkpoint_path, device="cpu")


def test_read_board_detects_occupied_cells_and_classifies_them(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    reference = _blank_board_image()
    current = reference.copy()
    rng = random.Random(7)
    _place_tile(current, 4, 0, "A", rng)
    _place_tile(current, 4, 1, "N", rng)
    _place_tile(current, 4, 2, "T", rng)

    observations = read_board(current, IDENTITY_CALIBRATION, reference, classifier)

    coords = {obs.coord for obs in observations}
    assert coords == {(4, 0), (4, 1), (4, 2)}
    by_coord = {obs.coord: obs for obs in observations}
    assert by_coord[(4, 0)].letter == "A"
    assert by_coord[(4, 1)].letter == "N"
    assert by_coord[(4, 2)].letter == "T"
    assert all(not obs.is_blank for obs in observations)


def test_read_board_flags_blank_tile_with_no_letter(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    reference = _blank_board_image()
    current = reference.copy()
    _place_tile(current, 4, 0, None, random.Random(7))

    observations = read_board(current, IDENTITY_CALIBRATION, reference, classifier)

    assert len(observations) == 1
    obs = observations[0]
    assert obs.coord == (4, 0)
    assert obs.is_blank
    assert obs.letter is None


def test_read_new_cells_excludes_cells_already_in_board_before(tmp_path):
    # Same photo shows three occupied cells, but board_before already
    # accounts for one of them (a prior turn's tile) -- read_new_cells
    # should only report the other two as this turn's new tiles, exactly
    # the per-turn diffing the constraint decoder needs (WS4).
    classifier = _train_tiny_classifier(tmp_path)
    reference = _blank_board_image()
    current = reference.copy()
    rng = random.Random(7)
    _place_tile(current, 4, 0, "A", rng)
    _place_tile(current, 4, 1, "N", rng)
    _place_tile(current, 4, 2, "T", rng)

    board_before = BoardState({(4, 0): Tile("A")})

    candidates = read_new_cells(current, IDENTITY_CALIBRATION, reference, classifier, board_before, top_k=2)

    coords = {cc.coord for cc in candidates}
    assert coords == {(4, 1), (4, 2)}
    for cc in candidates:
        assert len(cc.candidates) == 2
        assert all(isinstance(label, str) and isinstance(conf, float) for label, conf in cc.candidates)


def test_read_new_cells_respects_custom_occupancy_thresholds(tmp_path):
    # Stands in for the real WESPA finding: a venue's busy premium-square
    # graphics can create incidental diff/gradient signal even with no
    # tile present, causing the NASPA-tuned defaults to spuriously flag
    # empty cells as occupied. A brightness shift with no real texture
    # (unlike a genuine tile) triggers `diff` but barely moves `gradient`,
    # so a caller who knows this venue's tuned (higher) thresholds should
    # be able to exclude it while still detecting the real placement.
    classifier = _train_tiny_classifier(tmp_path)
    reference = _blank_board_image()
    current = reference.copy()
    rng = random.Random(9)
    _place_tile(current, 4, 0, "A", rng)

    x1, y1, x2, y2 = cell_bounds(6, 6)
    shifted = np.clip(current[y1:y2, x1:x2].astype(np.int16) + 60, 0, 255).astype(np.uint8)
    current[y1:y2, x1:x2] = shifted

    board_before = BoardState()

    default_candidates = read_new_cells(current, IDENTITY_CALIBRATION, reference, classifier, board_before, top_k=2)
    default_coords = {cc.coord for cc in default_candidates}
    assert (6, 6) in default_coords, "the brightness-shifted empty cell should spuriously read as occupied by default"
    assert (4, 0) in default_coords

    tuned_candidates = read_new_cells(
        current, IDENTITY_CALIBRATION, reference, classifier, board_before, top_k=2,
        diff_threshold=50.0, gradient_threshold=150.0,
    )
    tuned_coords = {cc.coord for cc in tuned_candidates}
    assert (6, 6) not in tuned_coords, "a venue-tuned (higher) threshold should exclude the spurious cell"
    assert (4, 0) in tuned_coords, "the genuine placement must still be detected regardless of threshold"


def test_read_new_cells_voted_outvotes_a_single_bad_frame(tmp_path):
    # Four frames genuinely show an "A" tile at (4, 0); a fifth frame
    # shows a "T" tile there instead -- standing in for a single bad look
    # (glare, motion blur, a bad angle) at the *same* physical cell.
    # Voting across all five should still land on "A", the true majority,
    # exactly the M2 lever docs/classifier-accuracy-plan.md calls for.
    classifier = _train_tiny_classifier(tmp_path)
    reference = _blank_board_image()
    board_before = BoardState()
    rng = random.Random(11)

    good_frames = []
    for _ in range(4):
        frame = reference.copy()
        _place_tile(frame, 4, 0, "A", rng)
        good_frames.append(frame)

    bad_frame = reference.copy()
    _place_tile(bad_frame, 4, 0, "T", rng)

    frames = good_frames + [bad_frame]

    candidates = read_new_cells_voted(
        frames, IDENTITY_CALIBRATION, reference, classifier, board_before, top_k=2,
    )

    assert len(candidates) == 1
    cc = candidates[0]
    assert cc.coord == (4, 0)
    assert cc.candidates[0][0] == "A"


def test_read_new_cells_voted_requires_at_least_one_frame(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    reference = _blank_board_image()
    with pytest.raises(ValueError):
        read_new_cells_voted([], IDENTITY_CALIBRATION, reference, classifier, BoardState())


class _FakeRackDetector:
    """Stands in for a trained RF-DETR checkpoint: returns a fixed set of
    boxes regardless of input, so `read_rack`'s crop-and-classify wiring
    can be tested without a real checkpoint or GPU (mirrors how
    test_visualize_rack_detections.py stubs out `supervision.Detections`
    rather than loading a real model)."""

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, image, threshold=0.3):
        sv = pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")
        return sv.Detections(
            xyxy=np.array(self._boxes, dtype=np.float64),
            confidence=np.full(len(self._boxes), 0.9),
            class_id=np.zeros(len(self._boxes), dtype=int),
        )


def _rack_image_with_tiles(letters, rng, gap=10):
    """Builds a rack image with `letters` pasted at known, non-overlapping
    pixel boxes -- returns (image, boxes) so a test can hand the exact
    boxes to `_FakeRackDetector` and know precisely what `read_rack` should
    crop and classify. Tile size is fixed at `tile_renderer.FINAL_SIZE`
    since `augment_tile` always downsamples to that size regardless of
    the size passed to `render_tile`."""
    tile_size = FINAL_SIZE
    width = len(letters) * (tile_size + gap) + gap
    height = tile_size + 2 * gap
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (200, 200, 200)  # neutral rack-colored background

    boxes = []
    for i, letter in enumerate(letters):
        tile = augment_tile(render_tile(letter, rng=rng), rng=rng)
        tile_arr = np.array(tile)[:, :, ::-1]  # RGB -> BGR to match rack_frame's convention
        x1 = gap + i * (tile_size + gap)
        y1 = gap
        x2, y2 = x1 + tile_size, y1 + tile_size
        image[y1:y2, x1:x2] = tile_arr
        boxes.append((x1, y1, x2, y2))
    return image, boxes


def test_read_rack_crops_detected_boxes_and_classifies_them(tmp_path):
    pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")
    classifier = _train_tiny_classifier(tmp_path)
    rng = random.Random(5)
    rack_image, boxes = _rack_image_with_tiles(["A", "N", "T"], rng)
    detector = _FakeRackDetector(boxes)

    observations = read_rack(rack_image, detector, classifier)

    assert len(observations) == 3
    assert [obs.letter for obs in observations] == ["A", "N", "T"]
    assert all(not obs.is_blank for obs in observations)
    assert [obs.box for obs in observations] == boxes


def test_read_rack_returns_tiles_left_to_right_not_in_detector_order(tmp_path):
    # RF-DETR returns detections in its own (confidence) order, not spatial
    # order -- observed on a real checkpoint, where a real "SPRAEVG" rack
    # came back as A,S,R,E,P,V,G. A rack only reads as a rack left-to-right,
    # so read_rack sorts; this pins that guarantee by handing the fake
    # detector deliberately shuffled boxes.
    pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")
    classifier = _train_tiny_classifier(tmp_path)
    rng = random.Random(5)
    rack_image, boxes = _rack_image_with_tiles(["A", "N", "T"], rng)

    shuffled = [boxes[2], boxes[0], boxes[1]]
    observations = read_rack(rack_image, _FakeRackDetector(shuffled), classifier)

    assert [obs.box for obs in observations] == boxes
    assert [obs.letter for obs in observations] == ["A", "N", "T"]


def test_read_rack_flags_blank_tile_with_no_letter(tmp_path):
    pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")
    classifier = _train_tiny_classifier(tmp_path)
    rng = random.Random(5)
    rack_image, boxes = _rack_image_with_tiles([None], rng)
    detector = _FakeRackDetector(boxes)

    observations = read_rack(rack_image, detector, classifier)

    assert len(observations) == 1
    assert observations[0].is_blank
    assert observations[0].letter is None


def test_rack_observations_to_tiles_preserves_blank_flag_without_a_letter():
    observations = [
        RackTileObservation(letter="A", is_blank=False, confidence=0.95, box=(0, 0, 10, 10)),
        RackTileObservation(letter=None, is_blank=True, confidence=0.7, box=(10, 0, 20, 10)),
    ]
    tiles = rack_observations_to_tiles(observations)
    assert tiles == [Tile("A"), Tile(None, is_blank=True)]


def test_partition_submits_directly_when_all_confident_and_no_blanks():
    observations = [
        CellObservation(coord=(7, 6), letter="A", is_blank=False, confidence=0.99),
        CellObservation(coord=(7, 7), letter="N", is_blank=False, confidence=0.95),
    ]
    result = partition_observations(observations, confidence_threshold=0.9)
    assert result.needs_operator_input == []
    assert set(result.submittable) == {((7, 6), "A", False), ((7, 7), "N", False)}


def test_partition_routes_whole_move_to_operator_if_any_blank_present():
    observations = [
        CellObservation(coord=(7, 6), letter="A", is_blank=False, confidence=0.99),
        CellObservation(coord=(7, 7), letter=None, is_blank=True, confidence=0.6),
    ]
    result = partition_observations(observations, confidence_threshold=0.9)
    assert result.submittable == []
    assert result.needs_operator_input == observations


def test_partition_routes_whole_move_to_operator_on_low_confidence():
    observations = [
        CellObservation(coord=(7, 6), letter="A", is_blank=False, confidence=0.99),
        CellObservation(coord=(7, 7), letter="Z", is_blank=False, confidence=0.4),
    ]
    result = partition_observations(observations, confidence_threshold=0.9)
    assert result.submittable == []
    assert len(result.needs_operator_input) == 2
