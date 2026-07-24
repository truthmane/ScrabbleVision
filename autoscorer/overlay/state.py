"""Builds the minimal, display-ready `OverlayState` the stream overlay
subscribes to. Deliberately small and flat: the overlay should never need to
interpret board coordinates or premium-square logic, just render numbers
and text.
"""
from __future__ import annotations

from typing import Any, Dict

from autoscorer.gamelogic.eventlog.store import GameState
from autoscorer.gamelogic.models import MoveType


def build_overlay_state(game_state: GameState) -> Dict[str, Any]:
    pool = game_state.pool_state()
    last_move = game_state.history[-1] if game_state.history else None

    last_word = None
    last_points = None
    last_move_type = None
    turn_number = 0
    last_player = None

    if last_move is not None:
        last_move_type = last_move.candidate.move_type.value
        turn_number = last_move.candidate.turn_number
        last_player = last_move.candidate.player_id
        if last_move.candidate.move_type == MoveType.PLAY and last_move.move_score is not None:
            last_word = "/".join(w.text for w in last_move.move_score.words)
            last_points = last_move.move_score.total

    return {
        "turn_number": turn_number,
        "scores": dict(game_state.scores),
        "last_player": last_player,
        "last_move_type": last_move_type,
        "last_word": last_word,
        "last_points": last_points,
        "bag_count": pool.bag_count,
    }
