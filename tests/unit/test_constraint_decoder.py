from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.movedetect.constraint_decoder import (
    CellCandidates,
    decode_feasible_reading,
    remaining_supply,
)


def test_remaining_supply_subtracts_visible_tiles():
    board = BoardState({(7, 7): Tile("Q")})
    supply = remaining_supply(board, racks=[[]])
    assert supply["Q"] == 0  # the standard set has exactly 1 Q, already on the board


def test_uncontested_candidates_all_get_their_top_choice():
    board = BoardState()
    candidates = [
        CellCandidates(coord=(7, 6), candidates=[("A", 0.9), ("R", 0.05)]),
        CellCandidates(coord=(7, 7), candidates=[("N", 0.8), ("M", 0.1)]),
    ]
    assignment = decode_feasible_reading(candidates, board, racks=[[]])
    assert assignment == {(7, 6): "A", (7, 7): "N"}


def test_scarce_letter_goes_to_the_more_confident_cell():
    # Only 1 Q exists in the standard set and none are visible elsewhere,
    # so with two cells both guessing "Q", only one can actually get it --
    # it should go to whichever cell the classifier is more sure about.
    board = BoardState()
    candidates = [
        CellCandidates(coord=(7, 6), candidates=[("Q", 0.55), ("O", 0.2)]),
        CellCandidates(coord=(7, 7), candidates=[("Q", 0.9), ("O", 0.05)]),
    ]
    assignment = decode_feasible_reading(candidates, board, racks=[[]])
    assert assignment[(7, 7)] == "Q"  # higher confidence wins the scarce letter
    assert assignment[(7, 6)] == "O"  # falls back to its next candidate


def test_falls_back_to_top_choice_when_nothing_in_topk_is_feasible():
    # Both existing Qs are already visible on the board; a cell whose
    # entire top-k guess list is "Q" has no feasible candidate at all --
    # falls back to the raw top-1 rather than fabricating a reading.
    board = BoardState({(0, 0): Tile("Q")})
    candidates = [CellCandidates(coord=(7, 7), candidates=[("Q", 0.6)])]
    assignment = decode_feasible_reading(candidates, board, racks=[[]])
    assert assignment[(7, 7)] == "Q"


def test_blank_tile_labels_map_to_the_blank_pool_bucket():
    # The standard set has 2 blanks; a rack already holding one leaves
    # exactly 1 remaining -- a "BLANK" classifier label must be checked
    # against that budget, not treated as if it named a literal letter.
    board = BoardState()
    racks = [[Tile(None, is_blank=True)], []]
    candidates = [CellCandidates(coord=(7, 7), candidates=[("BLANK", 0.7), ("O", 0.1)])]
    assignment = decode_feasible_reading(candidates, board, racks)
    assert assignment[(7, 7)] == "BLANK"
