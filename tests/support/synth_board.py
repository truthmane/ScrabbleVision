"""Shared synthetic-board-frame helpers, factored out of
`test_game_watcher.py`/`test_run_watcher.py` (which both had identical
copies) so `tests/slow/test_synthetic_full_game.py` can build on the same,
already-proven rendering path instead of a third copy.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from autoscorer.gamelogic.board import BOARD_SIZE, PREMIUM_SQUARES
from autoscorer.perception.calibration.homography import CANONICAL_SIZE, cell_bounds
from training.synth_render.tile_renderer import SQUARE_COLORS, augment_tile, render_tile


def blank_board_image() -> np.ndarray:
    img = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            code = PREMIUM_SQUARES.get((row, col), "plain")
            color = SQUARE_COLORS[code]
            x1, y1, x2, y2 = cell_bounds(row, col)
            img[y1:y2, x1:x2] = color[::-1]
    return img


def place_tile(
    board_img: np.ndarray, row: int, col: int, letter: Optional[str], rng: random.Random,
) -> np.ndarray:
    """Renders and augments one tile into `board_img` at (row, col),
    in-place, and returns the exact augmented crop (BGR, board-image
    orientation) that was pasted -- callers that need to know precisely
    which pixels ended up at that cell (e.g. a ground-truth lookup keyed
    by crop content) can use the return value instead of re-cropping."""
    tile = augment_tile(render_tile(letter, rng=rng), rng=rng)
    tile_arr = np.array(tile)[:, :, ::-1]
    x1, y1, x2, y2 = cell_bounds(row, col)
    board_img[y1:y2, x1:x2] = tile_arr
    return tile_arr
