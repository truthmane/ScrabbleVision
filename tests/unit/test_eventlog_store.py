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
