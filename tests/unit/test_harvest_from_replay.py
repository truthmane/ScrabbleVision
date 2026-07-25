import numpy as np
import cv2

from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.perception.calibration.homography import CANONICAL_SIZE, cell_bounds
from training.collect.harvest_from_replay import harvest_board_cells


def _synthetic_rectified_board() -> np.ndarray:
    """A plain canonical-sized image -- content doesn't matter for this
    test, only that a crop of the right size and location is written per
    occupied cell (the real photo case is exercised manually, since a
    real broadcast frame can't be committed to the repo)."""
    return np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8)


def test_harvest_writes_one_labeled_crop_per_occupied_cell(tmp_path):
    board = BoardState({
        (7, 7): Tile("C"),
        (7, 8): Tile("A"),
        (7, 9): Tile("T", is_blank=True),
    })
    rectified_path = tmp_path / "rectified.png"
    cv2.imwrite(str(rectified_path), _synthetic_rectified_board())

    out_dir = tmp_path / "out"
    count = harvest_board_cells(rectified_path, board, out_dir, "game1")

    assert count == 3
    written = sorted(p.name for p in out_dir.glob("*.png"))
    assert written == ["A_game1_7_8.png", "BLANK_game1_7_9.png", "C_game1_7_7.png"]


def test_harvest_crops_the_correct_cell_region(tmp_path):
    # Paint a distinct color into exactly one cell of the canonical image,
    # then confirm the crop for that cell -- and only that cell -- picks
    # it up, proving cell_bounds/crop_cell addressing is used correctly
    # rather than e.g. an off-by-one cell.
    image = _synthetic_rectified_board()
    x1, y1, x2, y2 = cell_bounds(3, 5)
    image[y1:y2, x1:x2] = (10, 20, 30)

    board = BoardState({(3, 5): Tile("Z"), (0, 0): Tile("Q")})
    rectified_path = tmp_path / "rectified.png"
    cv2.imwrite(str(rectified_path), image)

    out_dir = tmp_path / "out"
    harvest_board_cells(rectified_path, board, out_dir, "t")

    painted_crop = cv2.imread(str(out_dir / "Z_t_3_5.png"))
    assert (painted_crop == [10, 20, 30]).all()

    other_crop = cv2.imread(str(out_dir / "Q_t_0_0.png"))
    assert not (other_crop == [10, 20, 30]).all()
