import random

import numpy as np
import torch

from autoscorer.gamelogic.board import BOARD_SIZE, PREMIUM_SQUARES
from autoscorer.perception.board_reader import partition_observations, read_board, CellObservation
from autoscorer.perception.calibration.homography import CANONICAL_SIZE, BoardCalibration, cell_bounds
from training.classify.infer import TileClassifierModel
from training.classify.train import run_training, save_checkpoint
from training.synth_render.tile_renderer import SQUARE_COLORS, augment_tile, render_tile

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
