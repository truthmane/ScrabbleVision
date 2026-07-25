"""Orchestrates one turn: classify it, validate any tile placement, resolve
the word(s) formed, and score the move. This is the seam between the
perception layer (which supplies before/after board+rack states) and the
rest of the game logic -- it has no dependency on how those states were
observed (camera, operator manual entry, or test fixtures).
"""
from __future__ import annotations

from typing import Sequence, Union

from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.models import MoveCandidate, MoveProcessingError, MoveType, ScoredMove
from autoscorer.gamelogic.movedetect.classifier import classify_move, diff_new_cells
from autoscorer.gamelogic.movedetect.validator import validate_placement
from autoscorer.gamelogic.movedetect.word_resolver import words_formed
from autoscorer.gamelogic.scoring.rules_engine import score_move


def process_turn(
    turn_number: int,
    player_id: str,
    board_before: BoardState,
    board_after: BoardState,
    racks_before: Sequence[Sequence[Tile]],
    racks_after: Sequence[Sequence[Tile]],
) -> Union[ScoredMove, MoveProcessingError]:
    move_type = classify_move(board_before, board_after, racks_before, racks_after)

    if move_type != MoveType.PLAY:
        candidate = MoveCandidate(turn_number=turn_number, player_id=player_id, move_type=move_type)
        return ScoredMove(candidate=candidate, move_score=None)

    new_cells = diff_new_cells(board_before, board_after)
    validation = validate_placement(board_before, board_after, new_cells)
    if not validation.ok:
        return MoveProcessingError(reason=validation.reason or "invalid placement")

    words = words_formed(board_after, new_cells)
    move_score = score_move(board_after, words, new_cells)
    blank_cells = tuple(c for c in new_cells if board_after.get(c).is_blank)

    candidate = MoveCandidate(
        turn_number=turn_number,
        player_id=player_id,
        move_type=move_type,
        new_cells=tuple(new_cells),
        blank_cells=blank_cells,
    )
    return ScoredMove(candidate=candidate, move_score=move_score)
