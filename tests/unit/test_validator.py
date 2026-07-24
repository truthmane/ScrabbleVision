from autoscorer.gamelogic.board import CENTER, BoardState, Tile
from autoscorer.gamelogic.movedetect.validator import validate_placement


def _place(board_before, placements):
    board_after = board_before.with_placements(placements)
    return board_after, list(placements.keys())


def test_first_move_must_cover_center():
    board_before = BoardState()
    board_after, new_cells = _place(board_before, {(0, 0): Tile("A"), (0, 1): Tile("T")})

    result = validate_placement(board_before, board_after, new_cells)

    assert not result.ok
    assert "center" in result.reason


def test_first_move_covering_center_is_legal():
    board_before = BoardState()
    board_after, new_cells = _place(board_before, {(7, 6): Tile("C"), (7, 7): Tile("A"), (7, 8): Tile("T")})

    result = validate_placement(board_before, board_after, new_cells)

    assert result.ok


def test_non_first_move_must_connect_to_existing_tile():
    board_before = BoardState({(7, 6): Tile("C"), (7, 7): Tile("A"), (7, 8): Tile("T")})
    # A disconnected word placed far away from the existing "CAT".
    board_after, new_cells = _place(board_before, {(0, 0): Tile("A"), (0, 1): Tile("T")})

    result = validate_placement(board_before, board_after, new_cells)

    assert not result.ok
    assert "connected" in result.reason


def test_move_connected_only_via_perpendicular_cross_word_is_legal():
    board_before = BoardState({(8, 7): Tile("S")})
    board_after, new_cells = _place(board_before, {
        (7, 6): Tile("C"), (7, 7): Tile("A"), (7, 8): Tile("T"),
    })

    result = validate_placement(board_before, board_after, new_cells)

    assert result.ok


def test_new_tiles_must_lie_in_single_row_or_column():
    board_before = BoardState({(7, 7): Tile("A")})
    board_after, new_cells = _place(board_before, {(6, 6): Tile("B"), (8, 8): Tile("C")})

    result = validate_placement(board_before, board_after, new_cells)

    assert not result.ok
    assert "single row or column" in result.reason


def test_placement_with_gap_is_illegal():
    board_before = BoardState()
    # (7,6) and (7,8) are new, but (7,7) is left empty -- a gap.
    board_after, new_cells = _place(board_before, {(7, 6): Tile("C"), (7, 8): Tile("T")})

    result = validate_placement(board_before, board_after, new_cells)

    assert not result.ok
    assert "contiguous" in result.reason


def test_cell_already_occupied_before_move_is_rejected():
    board_before = BoardState({(7, 7): Tile("A")})
    # Simulate a caller incorrectly claiming an already-occupied cell as new.
    board_after = board_before
    result = validate_placement(board_before, board_after, [(7, 7)])

    assert not result.ok
    assert "already occupied" in result.reason


def test_no_new_cells_is_rejected():
    board_before = BoardState()
    result = validate_placement(board_before, board_before, [])

    assert not result.ok
