import base64
import json

import cv2
import numpy as np
import pytest

from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.perception.calibration.homography import CANONICAL_CELL_PX, cell_bounds
from training.collect.click_calibrate import (
    CalibrationTarget,
    board_from_woogles_document,
    build_spotcheck_montage,
    fit_homography_from_clicks,
    generate_click_tool_html,
    harvest_labeled_cells,
    load_frame,
    pick_calibration_targets,
)


def test_load_frame_rotate_180_matches_cv2_directly(tmp_path):
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[0, 0] = (1, 2, 3)  # a distinctive corner pixel
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), image)

    rotated = load_frame(image_path, rotate=180)

    assert rotated.shape == image.shape
    assert tuple(rotated[-1, -1]) == (1, 2, 3)


def test_load_frame_rejects_unreadable_path(tmp_path):
    with pytest.raises(ValueError):
        load_frame(tmp_path / "does_not_exist.png")


def test_pick_calibration_targets_spreads_across_the_board():
    # A line of occupied cells plus one far outlier -- farthest-point
    # sampling should pick the outlier and the line's own extremes rather
    # than clustering all picks in one area.
    board = BoardState({
        (7, 7): Tile("A"), (7, 8): Tile("B"), (7, 9): Tile("C"),
        (7, 10): Tile("D"), (7, 11): Tile("E"),
        (0, 0): Tile("Z"),
    })
    targets = pick_calibration_targets(board, count=3)
    assert len(targets) == 3
    coords = {(t.row, t.col) for t in targets}
    assert (0, 0) in coords  # the outlier must be picked


def test_pick_calibration_targets_caps_at_available_occupied_cells():
    board = BoardState({(7, 7): Tile("A"), (7, 8): Tile("B")})
    targets = pick_calibration_targets(board, count=8)
    assert len(targets) == 2


def test_pick_calibration_targets_labels_blanks_correctly():
    board = BoardState({(7, 7): Tile("Q", is_blank=True)})
    targets = pick_calibration_targets(board, count=1)
    assert targets[0].label == "BLANK"


def test_pick_calibration_targets_rejects_empty_board():
    with pytest.raises(ValueError):
        pick_calibration_targets(BoardState({}))


def test_board_from_woogles_document_decodes_letters_and_blanks(tmp_path):
    # 225-byte row-major board: A at (0,0), Z at (0,1), a blank playing M at (1,0).
    tiles = bytearray(225)
    tiles[0] = 1  # A
    tiles[1] = 26  # Z
    tiles[15] = 128 + 13  # blank playing M
    doc = {"board": {"tiles": base64.b64encode(bytes(tiles)).decode("ascii")}}
    doc_path = tmp_path / "doc.json"
    doc_path.write_text(json.dumps(doc))

    board = board_from_woogles_document(doc_path)

    assert board.get((0, 0)) == Tile("A")
    assert board.get((0, 1)) == Tile("Z")
    tile = board.get((1, 0))
    assert tile.letter == "M" and tile.is_blank is True
    assert len(list(board.occupied_cells())) == 3


def test_board_from_woogles_document_reconstructs_state_through_an_earlier_event(tmp_path):
    doc = {
        "events": [
            {"type": "TILE_PLACEMENT_MOVE", "row": 7, "column": 6, "direction": "HORIZONTAL",
             "played_tiles": base64.b64encode(bytes([1, 22, 15])).decode("ascii")},  # AVO @ (7,6..8)
            {"type": "TILE_PLACEMENT_MOVE", "row": 11, "column": 6, "direction": "HORIZONTAL",
             "played_tiles": base64.b64encode(bytes([12, 5, 0, 19])).decode("ascii")},  # hooks through an existing tile
        ],
        "board": {"tiles": base64.b64encode(bytes(225)).decode("ascii")},  # not used when through_event is set
    }
    doc_path = tmp_path / "doc.json"
    doc_path.write_text(json.dumps(doc))

    board = board_from_woogles_document(doc_path, through_event=1)

    assert len(list(board.occupied_cells())) == 3
    assert board.get((7, 6)) == Tile("A")
    assert board.get((7, 7)) == Tile("V")
    assert board.get((7, 8)) == Tile("O")


def test_board_from_woogles_document_skips_hooked_cells_without_overwriting(tmp_path):
    doc = {
        "events": [
            {"type": "TILE_PLACEMENT_MOVE", "row": 7, "column": 6, "direction": "HORIZONTAL",
             "played_tiles": base64.b64encode(bytes([1, 22, 15])).decode("ascii")},
            # second move hooks through (7,7)='V' (byte 0) and adds two real new letters either side
            {"type": "TILE_PLACEMENT_MOVE", "row": 7, "column": 6, "direction": "HORIZONTAL",
             "played_tiles": base64.b64encode(bytes([12, 0, 19])).decode("ascii")},
        ],
        "board": {"tiles": base64.b64encode(bytes(225)).decode("ascii")},
    }
    doc_path = tmp_path / "doc.json"
    doc_path.write_text(json.dumps(doc))

    board = board_from_woogles_document(doc_path, through_event=2)

    assert board.get((7, 6)) == Tile("L")
    assert board.get((7, 7)) == Tile("V")  # untouched by the hook, not overwritten
    assert board.get((7, 8)) == Tile("S")
    assert len(list(board.occupied_cells())) == 3


def test_generate_click_tool_html_embeds_targets_and_image(tmp_path):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    targets = [CalibrationTarget(row=1, col=2, label="Q")]
    out_html = tmp_path / "clicker.html"

    generate_click_tool_html(image, targets, out_html)

    html = out_html.read_text()
    assert '"row": 1' in html
    assert '"label": "Q"' in html
    assert "data:image/jpeg;base64," in html


def test_fit_homography_from_clicks_recovers_a_known_transform():
    # Simulate a perfect pure-scale-and-translate source: click position =
    # 2x the canonical center, offset by (100, 50). The fit should recover
    # this closely and report near-zero reprojection error for every point.
    targets = [(0, 0), (0, 14), (14, 0), (14, 14), (7, 7), (3, 11)]
    clicks = []
    for row, col in targets:
        canon_x = col * CANONICAL_CELL_PX + CANONICAL_CELL_PX / 2
        canon_y = row * CANONICAL_CELL_PX + CANONICAL_CELL_PX / 2
        click_x = canon_x * 2 + 100
        click_y = canon_y * 2 + 50
        clicks.append((row, col, click_x, click_y))

    calibration, errors = fit_homography_from_clicks(clicks)

    assert all(err < 1.0 for err in errors)
    # A point not in the fit set should also rectify to the right place.
    probe_row, probe_col = 5, 9
    canon_x = probe_col * CANONICAL_CELL_PX + CANONICAL_CELL_PX / 2
    canon_y = probe_row * CANONICAL_CELL_PX + CANONICAL_CELL_PX / 2
    probe_click = np.array([canon_x * 2 + 100, canon_y * 2 + 50, 1.0])
    projected = calibration.homography @ probe_click
    projected = projected[:2] / projected[2]
    assert projected == pytest.approx([canon_x, canon_y], abs=1.0)


def test_fit_homography_from_clicks_requires_at_least_four_points():
    with pytest.raises(ValueError):
        fit_homography_from_clicks([(0, 0, 1.0, 1.0), (0, 1, 2.0, 2.0), (1, 0, 3.0, 3.0)])


def test_harvest_labeled_cells_end_to_end_with_identity_calibration(tmp_path):
    from autoscorer.perception.calibration.homography import CANONICAL_SIZE, BoardCalibration

    board = BoardState({(3, 5): Tile("Z")})
    image = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8)
    x1, y1, x2, y2 = cell_bounds(3, 5)
    image[y1:y2, x1:x2] = (10, 20, 30)

    identity_calib = BoardCalibration(homography=np.eye(3, dtype=np.float64))
    out_dir = tmp_path / "harvest"
    count = harvest_labeled_cells(image, board, identity_calib, out_dir, "probe")

    assert count == 1
    crop = cv2.imread(str(out_dir / "Z_probe_3_5.png"))
    assert (crop == [10, 20, 30]).all()


def test_build_spotcheck_montage_writes_one_tile_per_label(tmp_path):
    harvest_dir = tmp_path / "harvest"
    harvest_dir.mkdir()
    for label, fname in [("A", "A_p_0_0.png"), ("A", "A_p_0_1.png"), ("BLANK", "BLANK_p_1_1.png")]:
        cv2.imwrite(str(harvest_dir / fname), np.full((60, 60, 3), 5, dtype=np.uint8))

    out_path = tmp_path / "montage.jpg"
    build_spotcheck_montage(harvest_dir, out_path)

    assert out_path.exists()
    montage = cv2.imread(str(out_path))
    assert montage is not None


def test_build_spotcheck_montage_rejects_empty_harvest_dir(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        build_spotcheck_montage(empty_dir, tmp_path / "out.jpg")
