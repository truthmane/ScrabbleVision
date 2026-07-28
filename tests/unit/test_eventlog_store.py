from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.eventlog.store import GameState
from autoscorer.gamelogic.movedetect.state_machine import process_turn


def test_apply_play_move_updates_board_score_and_history():
    game = GameState()
    board_after = game.board.with_placements({
        (7, 6): Tile("A"), (7, 7): Tile("N"), (7, 8): Tile("T"),
    })
    racks_after = {"p1": []}

    scored = process_turn(1, "p1", game.board, board_after, [[Tile("A"), Tile("N"), Tile("T")]], [[]])
    game.apply_move(scored, board_after, racks_after)

    assert game.board.get((7, 7)) == Tile("N")
    assert game.scores["p1"] == (1 + 1 + 1) * 2
    assert len(game.history) == 1


def test_pool_state_reflects_current_board_and_racks():
    game = GameState()
    game.racks = {"p1": [Tile("A")], "p2": [Tile("B")]}
    pool = game.pool_state()
    assert pool.bag["A"] < 9  # fewer than the full supply of 9 A's remain
    assert pool.bag["B"] < 2


def test_undo_last_reverses_the_most_recent_move_only():
    game = GameState()
    board_after_1 = game.board.with_placements({(7, 6): Tile("A"), (7, 7): Tile("N"), (7, 8): Tile("T")})
    scored_1 = process_turn(1, "p1", game.board, board_after_1, [[Tile("A"), Tile("N"), Tile("T")]], [[]])
    game.apply_move(scored_1, board_after_1, {"p1": []})

    board_after_2 = game.board.with_placements({(8, 6): Tile("S")})
    scored_2 = process_turn(2, "p2", game.board, board_after_2, [[], [Tile("S")]], [[], []])
    game.apply_move(scored_2, board_after_2, {"p1": [], "p2": []})

    undone = game.undo_last()

    assert undone.candidate.player_id == "p2"
    assert game.board.get((8, 6)) is None  # move 2 reversed
    assert game.board.get((7, 7)) == Tile("N")  # move 1 untouched
    assert "p2" not in game.scores or game.scores["p2"] == 0
    assert len(game.history) == 1


def test_undo_last_on_empty_history_returns_none():
    game = GameState()
    assert game.undo_last() is None
