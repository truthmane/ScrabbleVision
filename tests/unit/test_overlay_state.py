from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.eventlog.store import GameState
from autoscorer.gamelogic.movedetect.state_machine import process_turn
from autoscorer.overlay.state import build_overlay_state


def test_overlay_state_before_any_move():
    game = GameState()
    state = build_overlay_state(game)
    assert state["turn_number"] == 0
    assert state["scores"] == {}
    assert state["last_word"] is None
    assert state["bag_count"] == 100


def test_overlay_state_reflects_last_play():
    game = GameState()
    board_after = game.board.with_placements({
        (7, 6): Tile("A"), (7, 7): Tile("N"), (7, 8): Tile("T"),
    })
    scored = process_turn(1, "p1", game.board, board_after, [[Tile("A"), Tile("N"), Tile("T")]], [[]])
    game.apply_move(scored, board_after, {"p1": []})

    state = build_overlay_state(game)

    assert state["turn_number"] == 1
    assert state["last_player"] == "p1"
    assert state["last_move_type"] == "PLAY"
    assert state["last_word"] == "ANT"
    assert state["last_points"] == (1 + 1 + 1) * 2
    assert state["scores"]["p1"] == (1 + 1 + 1) * 2
    assert state["bag_count"] == 100 - 3
