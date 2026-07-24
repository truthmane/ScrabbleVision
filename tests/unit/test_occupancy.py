import numpy as np

from autoscorer.gamelogic.board import BOARD_SIZE, PREMIUM_SQUARES
from autoscorer.perception.calibration.homography import CANONICAL_SIZE, cell_bounds
from autoscorer.perception.occupancy.detector import detect_occupancy, is_occupied
from training.synth_render.tile_renderer import SQUARE_COLORS, render_tile


def _blank_board_image() -> np.ndarray:
    """A synthetic canonical board image with each cell filled in its
    premium-square color and no tiles -- stands in for a real empty-board
    reference photo captured at calibration time."""
    img = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            code = PREMIUM_SQUARES.get((row, col), "plain")
            color = SQUARE_COLORS[code]
            x1, y1, x2, y2 = cell_bounds(row, col)
            img[y1:y2, x1:x2] = color[::-1]  # RGB -> BGR for OpenCV-style arrays
    return img


def _place_tile(board_img: np.ndarray, row: int, col: int, letter) -> None:
    tile = render_tile(letter, size=60)
    tile_arr = np.array(tile)[:, :, ::-1]  # PIL RGB -> BGR
    x1, y1, x2, y2 = cell_bounds(row, col)
    board_img[y1:y2, x1:x2] = tile_arr


def test_occupied_cells_detected_diff_from_empty_reference():
    reference = _blank_board_image()
    current = reference.copy()
    occupied_cells = [(7, 6), (7, 7), (7, 8)]
    for row, col in occupied_cells:
        _place_tile(current, row, col, "A")

    occupancy = detect_occupancy(current, reference)

    for coord in occupied_cells:
        assert occupancy[coord] is True
    # Spot-check a handful of cells that should remain empty.
    for coord in [(0, 0), (3, 3), (14, 14), (7, 9)]:
        assert occupancy[coord] is False


def test_blank_tile_is_still_detected_as_occupied_despite_no_glyph():
    # A blank tile has no printed letter, but its physical presence still
    # changes the cell's texture/brightness relative to the bare board
    # square -- occupancy detection must not require a glyph to fire.
    reference = _blank_board_image()
    current = reference.copy()
    _place_tile(current, 7, 7, None)

    assert is_occupied(
        current[cell_bounds(7, 7)[1]:cell_bounds(7, 7)[3], cell_bounds(7, 7)[0]:cell_bounds(7, 7)[2]],
        reference[cell_bounds(7, 7)[1]:cell_bounds(7, 7)[3], cell_bounds(7, 7)[0]:cell_bounds(7, 7)[2]],
    )


def test_identical_crops_are_never_occupied():
    reference = _blank_board_image()
    x1, y1, x2, y2 = cell_bounds(5, 5)
    crop = reference[y1:y2, x1:x2]
    assert not is_occupied(crop, crop)
