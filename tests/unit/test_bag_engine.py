import pytest

from autoscorer.gamelogic.board import BLANK, STANDARD_DISTRIBUTION, TOTAL_TILE_COUNT, BoardState, Tile
from autoscorer.gamelogic.pool.bag_engine import (
    PoolInvariantViolation,
    check_total_invariant,
    compute_pool_state,
)


def test_empty_board_and_racks_bag_equals_full_distribution():
    pool = compute_pool_state(BoardState(), racks=[[], []])
    assert pool.bag == STANDARD_DISTRIBUTION
    assert pool.bag_count == TOTAL_TILE_COUNT


def test_bag_reduced_by_tiles_on_board_and_racks():
    board = BoardState({(7, 7): Tile("A"), (7, 8): Tile("A")})
    rack1 = [Tile("A"), Tile("B")]
    rack2 = [Tile("E")]

    pool = compute_pool_state(board, racks=[rack1, rack2])

    assert pool.bag["A"] == STANDARD_DISTRIBUTION["A"] - 3  # 2 on board + 1 in rack1
    assert pool.bag["B"] == STANDARD_DISTRIBUTION["B"] - 1
    assert pool.bag["E"] == STANDARD_DISTRIBUTION["E"] - 1
    assert pool.bag_count == TOTAL_TILE_COUNT - 5  # 2 A's on board + A, B in rack1 + E in rack2


def test_unplayed_blank_in_rack_counts_against_blank_bucket():
    rack = [Tile(None, is_blank=True)]
    pool = compute_pool_state(BoardState(), racks=[rack])
    assert pool.bag[BLANK] == STANDARD_DISTRIBUTION[BLANK] - 1


def test_blank_played_as_letter_on_board_counts_against_blank_not_letter():
    board = BoardState({(7, 7): Tile("Z", is_blank=True)})
    pool = compute_pool_state(board, racks=[[]])
    assert pool.bag[BLANK] == STANDARD_DISTRIBUTION[BLANK] - 1
    # The letter Z itself is untouched -- the blank, not a real Z, was used.
    assert pool.bag["Z"] == STANDARD_DISTRIBUTION["Z"]


def test_exceeding_standard_supply_raises_invariant_violation():
    # The standard set has only one Q; claiming two is impossible and
    # signals a misread somewhere upstream.
    board = BoardState({(7, 7): Tile("Q"), (7, 8): Tile("Q")})
    with pytest.raises(PoolInvariantViolation):
        compute_pool_state(board, racks=[[]])


def test_total_invariant_holds_for_a_consistent_state():
    board = BoardState({(7, 7): Tile("A")})
    rack1 = [Tile("B"), Tile("C")]
    racks = [rack1, []]
    pool = compute_pool_state(board, racks=racks)
    assert check_total_invariant(pool, board)
