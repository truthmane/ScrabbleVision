import io
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

from autoscorer.gamelogic.board import BOARD_SIZE, PREMIUM_SQUARES
from autoscorer.perception.calibration.homography import CANONICAL_SIZE, cell_bounds
from autoscorer.perception.occupancy.detector import (
    DEFAULT_GRADIENT_THRESHOLD,
    detect_occupancy,
    is_occupied,
    occupancy_scores,
)
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


def _add_realistic_texture_noise(cell_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulates what a genuinely empty real-photo cell actually looks like:
    printed premium-square label text, sensor/JPEG noise, none of which a
    clean synthetic render has. This is what a real end-to-end test against
    actual broadcast footage found: real empty cells scored 47-68 on the
    `gradient` signal from exactly this kind of texture -- already above the
    old default threshold of 15.0, causing every cell on the board to read
    as "occupied" (0% precision). See detector.py's DEFAULT_GRADIENT_THRESHOLD
    comment for the full story; this reproduces the failure mode
    synthetically so it can't silently regress without a real photo on hand.
    """
    img = Image.fromarray(cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    size = img.size[0]

    # Line color is an offset *relative to this cell's own base color*, not
    # a fixed absolute gray -- a fixed gray reads as a huge, unrealistic
    # jump against a dark premium square but barely registers against a
    # light one, which made the fixture's noise level wildly inconsistent
    # across premium-square colors. Relative contrast behaves consistently
    # regardless of the underlying tile color, same as real printed text
    # would.
    base = tuple(int(c) for c in cell_bgr[size // 2, size // 2][::-1])
    line_color = tuple(max(0, min(255, c - 60)) for c in base)
    # Short text-like strokes, standing in for printed premium-square labels
    # ("DOUBLE LETTER SCORE" etc.) -- real edges, real ink. Tuned (alongside
    # the noise/compression below) to land safely between the old broken
    # threshold (15.0) and the corrected one (100.0), the way genuinely
    # empty real cells measured on actual broadcast footage did.
    for _ in range(60):
        x, y = rng.uniform(2, size - 8), rng.uniform(2, size - 8)
        draw.line([(x, y), (x + rng.uniform(4, 10), y + rng.uniform(-3, 3))], fill=line_color, width=1)

    arr = np.array(img).astype(np.int16)
    noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(0, 6, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(arr)

    buf = io.BytesIO()
    noisy.save(buf, format="JPEG", quality=60)  # real-world broadcast-style compression artifacts
    buf.seek(0)
    compressed = np.array(Image.open(buf).convert("RGB"))
    return cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR)


def test_detect_occupancy_accepts_a_single_reference_unchanged():
    # Backward-compat pin: passing one ndarray (not a list) must behave
    # exactly as it always did.
    reference = _blank_board_image()
    current = reference.copy()
    _place_tile(current, 7, 7, "A")

    occupancy = detect_occupancy(current, reference)
    assert occupancy[(7, 7)] is True
    assert occupancy[(0, 0)] is False


def test_detect_occupancy_with_multiple_references_treats_either_as_valid_empty():
    # Stands in for a venue where "empty" genuinely has more than one
    # appearance (see the module docstring): a cell matching EITHER
    # reference on `diff` should read as empty, not just the first one.
    # Uses a plain, untextured color swap (not a letter glyph) for the
    # alternate state -- `gradient` is computed from the current crop
    # alone and would fire on real glyph texture regardless of which
    # reference matches on `diff`, which would defeat the point of this
    # test (see occupancy/detector.py's module docstring: this exact
    # texture-independent-of-reference behavior is why the real WESPA fix
    # needed a raised gradient_threshold too, not multi-reference alone).
    reference_a = _blank_board_image()
    reference_b = reference_a.copy()
    x1, y1, x2, y2 = cell_bounds(7, 7)
    reference_b[y1:y2, x1:x2] = (40, 40, 40)  # reference_b's own "empty" state: a plain dark patch at (7,7)

    current = reference_a.copy()
    current[y1:y2, x1:x2] = (40, 40, 40)  # matches reference_b's plain patch exactly, not reference_a

    occupancy_single = detect_occupancy(current, reference_a)
    assert occupancy_single[(7, 7)] is True, "against only reference_a, this looks occupied"

    occupancy_multi = detect_occupancy(current, [reference_a, reference_b])
    assert occupancy_multi[(7, 7)] is False, "matching reference_b should count as a valid empty state"

    # A real tile placed somewhere neither reference has anything must
    # still be correctly detected -- multi-reference must not go blind.
    assert occupancy_multi[(4, 4)] is False
    current_with_real_tile = current.copy()
    _place_tile(current_with_real_tile, 4, 4, "A")
    occupancy_with_tile = detect_occupancy(current_with_real_tile, [reference_a, reference_b])
    assert occupancy_with_tile[(4, 4)] is True


def test_detect_occupancy_rejects_an_empty_reference_list():
    import pytest
    reference = _blank_board_image()
    with pytest.raises(ValueError):
        detect_occupancy(reference, [])


def test_realistic_background_texture_does_not_trigger_false_occupancy():
    reference = _blank_board_image()
    rng = random.Random(0)
    empty_cells = [(0, 0), (3, 3), (7, 9), (10, 10), (14, 14)]

    for row, col in empty_cells:
        x1, y1, x2, y2 = cell_bounds(row, col)
        cur = _add_realistic_texture_noise(reference[y1:y2, x1:x2], rng)
        ref = reference[y1:y2, x1:x2]

        # This is the actual regression check: at the OLD default (15.0),
        # this kind of ordinary real-world texture alone was enough to read
        # as "occupied" on real footage -- reproduce that here so the fix
        # can't silently revert.
        scores = occupancy_scores(cur, ref)
        assert scores["gradient"] > 15.0, "test fixture isn't reproducing the real failure mode"

        # At the current (corrected) default, ordinary background texture
        # must NOT be mistaken for a tile.
        assert not is_occupied(cur, ref, gradient_threshold=DEFAULT_GRADIENT_THRESHOLD)
