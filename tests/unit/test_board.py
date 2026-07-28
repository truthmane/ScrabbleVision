import pytest

from autoscorer.gamelogic.board import (
    BLANK,
    BOARD_SIZE,
    CENTER,
    PREMIUM_SQUARES,
    STANDARD_DISTRIBUTION,
    TOTAL_TILE_COUNT,
    BoardState,
    Tile,
)


def test_standard_distribution_totals_100_tiles():
    assert TOTAL_TILE_COUNT == 100
    assert STANDARD_DISTRIBUTION[BLANK] == 2
    assert STANDARD_DISTRIBUTION["E"] == 12
    assert STANDARD_DISTRIBUTION["Q"] == 1


def test_premium_square_counts_match_official_board():
    from collections import Counter

    counts = Counter(PREMIUM_SQUARES.values())
    assert counts["3W"] == 8
    assert counts["3L"] == 12
    assert counts["2L"] == 24
    assert counts["2W"] == 17
    assert PREMIUM_SQUARES[CENTER] == "2W"
    assert CENTER == (7, 7)


def test_board_state_placement_and_occupancy():
    board = BoardState()
    assert board.is_blank_board()
    board2 = board.with_placements({(7, 7): Tile("A")})
    # original board is untouched
    assert board.is_blank_board()
    assert not board2.is_blank_board()
    assert board2.get((7, 7)) == Tile("A")


def test_without_cells_reverses_with_placements():
    board = BoardState()
    placed = board.with_placements({(7, 7): Tile("A"), (7, 8): Tile("N")})
    reverted = placed.without_cells([(7, 7), (7, 8)])

    assert reverted.is_blank_board()
    assert placed.get((7, 7)) == Tile("A")  # original untouched


def test_without_cells_leaves_other_tiles_in_place():
    board = BoardState({(7, 7): Tile("A"), (7, 8): Tile("N")})
    reverted = board.without_cells([(7, 8)])

    assert reverted.get((7, 7)) == Tile("A")
    assert reverted.get((7, 8)) is None


def test_cannot_place_on_occupied_cell():
    board = BoardState({(7, 7): Tile("A")})
    with pytest.raises(ValueError):
        board.with_placements({(7, 7): Tile("B")})


def test_unplayed_blank_cannot_be_placed_on_board():
    board = BoardState()
    unplayed_blank = Tile(None, is_blank=True)
    with pytest.raises(ValueError):
        board.with_placements({(7, 7): unplayed_blank})


def test_blank_played_as_letter_has_pool_key_blank_not_letter():
    played_blank = Tile("Z", is_blank=True)
    assert played_blank.letter == "Z"
    assert played_blank.pool_key == BLANK
