from autoscorer.gamelogic.models import MoveProcessingError, ScoredMove
from autoscorer.gamelogic.publish import PublishMode
from autoscorer.api.session import GameSession

FIRST_MOVE_TILES = [((7, 6), "A", False), ((7, 7), "N", False), ((7, 8), "T", False)]


def test_manual_mode_queues_move_until_approved():
    session = GameSession(mode=PublishMode.MANUAL)
    result = session.submit_move("p1", new_tiles=FIRST_MOVE_TILES)

    assert not result.published
    assert isinstance(result.outcome, ScoredMove)
    assert session.game_state.board.is_blank_board()  # not yet applied
    assert len(session.list_pending()) == 1

    applied = session.decide(turn_number=1, action="approve")

    assert applied
    assert not session.game_state.board.is_blank_board()
    assert session.game_state.scores["p1"] == (1 + 1 + 1) * 2
    assert session.list_pending() == []


def test_rejected_move_is_dropped_without_applying():
    session = GameSession(mode=PublishMode.MANUAL)
    session.submit_move("p1", new_tiles=FIRST_MOVE_TILES)

    applied = session.decide(turn_number=1, action="reject")

    assert applied is False  # found and processed, but not applied
    assert session.game_state.board.is_blank_board()
    assert session.list_pending() == []


def test_deciding_an_unknown_turn_number_returns_none():
    session = GameSession(mode=PublishMode.MANUAL)
    assert session.decide(turn_number=999, action="approve") is None


def test_autonomous_mode_applies_immediately():
    session = GameSession(mode=PublishMode.AUTONOMOUS)
    result = session.submit_move("p1", new_tiles=FIRST_MOVE_TILES)

    assert result.published
    assert not session.game_state.board.is_blank_board()
    assert session.list_pending() == []


def test_invalid_placement_does_not_consume_a_turn_number():
    session = GameSession(mode=PublishMode.AUTONOMOUS)
    # Doesn't cover the center square -- illegal first move.
    bad_tiles = [((0, 0), "A", False), ((0, 1), "T", False)]

    first_attempt = session.submit_move("p1", new_tiles=bad_tiles)
    assert isinstance(first_attempt.outcome, MoveProcessingError)

    second_attempt = session.submit_move("p1", new_tiles=FIRST_MOVE_TILES)
    assert second_attempt.outcome.candidate.turn_number == 1


def test_mode_can_be_changed_mid_game():
    session = GameSession(mode=PublishMode.MANUAL)
    session.submit_move("p1", new_tiles=FIRST_MOVE_TILES)
    assert len(session.list_pending()) == 1

    session.set_mode(PublishMode.AUTONOMOUS)
    session.decide(turn_number=1, action="approve")

    result = session.submit_move("p2", new_tiles=[((7, 9), "S", False)])
    assert result.published


def test_undo_last_move_reverses_board_and_score():
    session = GameSession(mode=PublishMode.AUTONOMOUS)
    session.submit_move("p1", new_tiles=FIRST_MOVE_TILES)
    assert session.game_state.scores["p1"] == (1 + 1 + 1) * 2
    assert not session.game_state.board.is_blank_board()

    undone = session.undo_last_move()

    assert undone.candidate.player_id == "p1"
    assert session.game_state.board.is_blank_board()
    assert session.game_state.scores["p1"] == 0


def test_undo_with_nothing_committed_returns_none():
    session = GameSession(mode=PublishMode.AUTONOMOUS)
    assert session.undo_last_move() is None


def test_rejected_reading_is_not_re_queued_until_the_board_changes():
    """Regression test for a real bug found on a live run: a candidate
    that keeps getting re-detected off the same unchanged (misread) board
    was re-queued as a brand-new pending turn every time, even after an
    operator had already rejected the identical reading once. Rejecting
    must make the exact same (player, cells, letters) a no-op next time,
    while a genuinely different reading (or the same reading for a
    different player/cells) is unaffected.
    """
    session = GameSession(mode=PublishMode.MANUAL)
    session.submit_move("p1", new_tiles=FIRST_MOVE_TILES)
    session.decide(turn_number=1, action="reject")
    assert session.list_pending() == []

    result = session.submit_move("p1", new_tiles=FIRST_MOVE_TILES)

    assert isinstance(result.outcome, MoveProcessingError)
    assert not result.published
    assert session.list_pending() == []  # not re-queued as a new turn
    assert session.game_state.board.is_blank_board()

    # A different reading for the same cells is not blocked.
    different_tiles = [((7, 6), "A", False), ((7, 7), "N", False), ((7, 8), "D", False)]
    different_result = session.submit_move("p1", new_tiles=different_tiles)
    assert not isinstance(different_result.outcome, MoveProcessingError)
