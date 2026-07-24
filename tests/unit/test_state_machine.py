from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.models import MoveProcessingError, MoveType, ScoredMove
from autoscorer.gamelogic.movedetect.state_machine import process_turn


def test_play_move_is_scored():
    board_before = BoardState()
    # A, N, T are each worth 1 point, keeping the arithmetic simple.
    board_after = board_before.with_placements({
        (7, 6): Tile("A"), (7, 7): Tile("N"), (7, 8): Tile("T"),
    })
    racks_before = [[Tile("A"), Tile("N"), Tile("T")], []]
    racks_after = [[], []]

    result = process_turn(1, "p1", board_before, board_after, racks_before, racks_after)

    assert isinstance(result, ScoredMove)
    assert result.candidate.move_type == MoveType.PLAY
    assert result.move_score is not None
    assert result.move_score.total == (1 + 1 + 1) * 2  # center is a double-word square


def test_exchange_move_has_no_score():
    board = BoardState()
    racks_before = [[Tile("Q"), Tile("Z")], []]
    racks_after = [[Tile("E"), Tile("R")], []]  # same board, rack composition changed

    result = process_turn(2, "p1", board, board, racks_before, racks_after)

    assert isinstance(result, ScoredMove)
    assert result.candidate.move_type == MoveType.EXCHANGE
    assert result.move_score is None


def test_pass_move_has_no_score():
    board = BoardState()
    racks = [[Tile("Q")], []]

    result = process_turn(3, "p1", board, board, racks, racks)

    assert isinstance(result, ScoredMove)
    assert result.candidate.move_type == MoveType.PASS
    assert result.move_score is None


def test_invalid_placement_returns_processing_error():
    board_before = BoardState()
    # Off-center, not covering the center square -- illegal first move.
    board_after = board_before.with_placements({(0, 0): Tile("A"), (0, 1): Tile("T")})
    racks_before = [[Tile("A"), Tile("T")], []]
    racks_after = [[], []]

    result = process_turn(1, "p1", board_before, board_after, racks_before, racks_after)

    assert isinstance(result, MoveProcessingError)
    assert "center" in result.reason
