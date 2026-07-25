import random
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch

from autoscorer.gamelogic.board import BOARD_SIZE, PREMIUM_SQUARES, BoardState, Coord, Tile
from autoscorer.gamelogic.movedetect.word_resolver import words_formed
from autoscorer.gamelogic.scoring.rules_engine import score_move
from autoscorer.perception.calibration.homography import CANONICAL_SIZE, cell_bounds
from training.classify.infer import TileClassifierModel
from training.classify.train import run_training, save_checkpoint
from training.collect.eval_harvested_moves import evaluate_move
from training.synth_render.tile_renderer import SQUARE_COLORS, augment_tile, render_tile


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
    tile = augment_tile(render_tile(letter, rng=rng), rng=rng)
    tile_arr = np.array(tile)[:, :, ::-1]
    x1, y1, x2, y2 = cell_bounds(row, col)
    board_img[y1:y2, x1:x2] = tile_arr


def _train_tiny_classifier(tmp_path: Path, labels=("A", "N", "T", "BLANK")) -> TileClassifierModel:
    rng = random.Random(0)
    for label in labels:
        letter = None if label == "BLANK" else label
        class_dir = tmp_path / "training_data" / label
        class_dir.mkdir(parents=True, exist_ok=True)
        base = render_tile(letter, rng=rng)
        for i in range(30):
            augment_tile(base, rng=rng).save(class_dir / f"{i:03d}.png")

    model, result = run_training(tmp_path / "training_data", epochs=25, batch_size=8, device=torch.device("cpu"), seed=1)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(model, result.classes, checkpoint_path)
    return TileClassifierModel(checkpoint_path, device="cpu")


def _expected_score(board_before: BoardState, placements: Dict[Coord, Tile]) -> int:
    board_after = board_before.with_placements(placements)
    new_cells = list(placements.keys())
    return score_move(board_after, words_formed(board_after, new_cells), new_cells).total


def test_evaluate_move_correctly_decodes_and_scores_a_clean_real_move(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)

    # Move 1: ANT across through the center (7,7), first move.
    move1 = {(7, 7): Tile("A"), (7, 8): Tile("N"), (7, 9): Tile("T")}
    score1 = _expected_score(BoardState(), move1)

    gcg_path = tmp_path / "game.gcg"
    gcg_path.write_text(f"#player1 P1 Player One\n#player2 P2 Player Two\n>P1: AANT??? 8H ANT +{score1} {score1}\n")

    image = _blank_board_image()
    rng = random.Random(3)
    for coord, tile in move1.items():
        _place_tile(image, *coord, tile.letter, rng)
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), image)

    result = evaluate_move(image_path, gcg_path, 1, classifier)

    assert result.cell_count == 3
    assert result.decoded_correct == 3
    assert not result.has_blank
    # Not asserting on `would_route_to_operator` here: this test's classifier
    # is a tiny 4-class model trained on 30 samples, so it can be correct
    # without being confident -- that's a calibration question, not what
    # this test is checking (the decode+score wiring is correct either way).
    assert result.move_score_matches is True
    assert result.computed_score == score1


def test_evaluate_move_routes_blank_moves_to_operator_without_scoring(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)

    # Move 1: ANT across through the center, same as above.
    move1 = {(7, 7): Tile("A"), (7, 8): Tile("N"), (7, 9): Tile("T")}
    score1 = _expected_score(BoardState(), move1)

    # Move 2: hooks the T at (7,9), plays a blank (played as "A") straight down.
    board_after_move1 = BoardState().with_placements(move1)
    move2 = {(8, 9): Tile("A", is_blank=True)}
    score2 = _expected_score(board_after_move1, move2)

    gcg_path = tmp_path / "game.gcg"
    gcg_path.write_text(
        f"#player1 P1 Player One\n#player2 P2 Player Two\n"
        f">P1: AANTXYZ 8H ANT +{score1} {score1}\n"
        f">P2: A?????? J8 .a +{score2} {score2}\n"
    )

    image = _blank_board_image()
    rng = random.Random(3)
    for coord, tile in move1.items():
        _place_tile(image, *coord, tile.letter, rng)
    _place_tile(image, 8, 9, None, rng)  # a blank has no printed glyph
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), image)

    result = evaluate_move(image_path, gcg_path, 2, classifier)

    assert result.has_blank
    assert result.would_route_to_operator
    assert result.move_score_matches is None
    assert result.computed_score is None
