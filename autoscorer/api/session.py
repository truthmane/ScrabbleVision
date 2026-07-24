"""Orchestrates a single live game: wraps GameState + PublishGateway with the
operations the API layer needs (submit a move, list/decide on pending moves,
change publish mode). Kept free of any ASGI/FastAPI concepts so it's testable
with plain function calls -- the web layer is a thin adapter around this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from autoscorer.gamelogic.board import BoardState, Coord, Tile
from autoscorer.gamelogic.eventlog.store import GameState
from autoscorer.gamelogic.models import MoveProcessingError, ScoredMove
from autoscorer.gamelogic.movedetect.state_machine import process_turn
from autoscorer.gamelogic.publish import PendingMove, PublishGateway, PublishMode

NewTile = Tuple[Coord, str, bool]  # (coord, letter, is_blank)
RackTile = Tuple[Optional[str], bool]  # (letter, is_blank) -- letter is None only for an unplayed blank


@dataclass
class SubmissionResult:
    outcome: Union[ScoredMove, MoveProcessingError]
    published: bool  # True if applied immediately, False if queued for operator review


class GameSession:
    def __init__(self, mode: PublishMode = PublishMode.MANUAL) -> None:
        self.game_state = GameState()
        self.gateway = PublishGateway(mode=mode)
        self.pending: Dict[int, PendingMove] = {}
        self._next_turn = 1

    def set_mode(self, mode: PublishMode) -> None:
        self.gateway.mode = mode

    def submit_move(
        self,
        player_id: str,
        new_tiles: Optional[List[NewTile]] = None,
        rack_after: Optional[List[RackTile]] = None,
        confidence: float = 1.0,
    ) -> SubmissionResult:
        game_state = self.game_state
        if player_id not in game_state.racks:
            game_state.racks[player_id] = []

        board_before = game_state.board
        if new_tiles:
            placements = {coord: Tile(letter, is_blank) for (coord, letter, is_blank) in new_tiles}
            board_after = board_before.with_placements(placements)
        else:
            board_after = board_before

        player_order = list(game_state.racks.keys())
        racks_before = [list(game_state.racks[p]) for p in player_order]

        racks_after_map = {p: list(game_state.racks[p]) for p in player_order}
        if rack_after is not None:
            racks_after_map[player_id] = [Tile(letter, is_blank) for (letter, is_blank) in rack_after]
        racks_after = [racks_after_map[p] for p in player_order]

        turn_number = self._next_turn
        result = process_turn(turn_number, player_id, board_before, board_after, racks_before, racks_after)

        if isinstance(result, MoveProcessingError):
            # Validation failed -- don't consume a turn number, let the
            # operator correct the input and resubmit.
            return SubmissionResult(outcome=result, published=False)

        self._next_turn += 1

        if self.gateway.should_auto_publish(confidence):
            self._apply(result, board_after, racks_after_map)
            return SubmissionResult(outcome=result, published=True)

        self.pending[turn_number] = PendingMove(
            scored_move=result, board_after=board_after, racks_after=racks_after_map, confidence=confidence,
        )
        return SubmissionResult(outcome=result, published=False)

    def decide(self, turn_number: int, action: str) -> Optional[bool]:
        """Approve or reject a pending move.

        Returns `None` if no such pending move exists (already decided, or
        an invalid turn number) -- the caller should treat that as "not
        found". Otherwise returns whether the move was applied to the
        canonical GameState (True for approve, False for a reject, which
        simply drops it so the operator can resubmit a correction).
        """
        pending = self.pending.pop(turn_number, None)
        if pending is None:
            return None
        if action == "approve":
            self._apply(pending.scored_move, pending.board_after, pending.racks_after)
            return True
        return False

    def list_pending(self) -> List[PendingMove]:
        return [self.pending[k] for k in sorted(self.pending)]

    def _apply(self, scored_move: ScoredMove, board_after: BoardState, racks_after: Dict[str, list]) -> None:
        self.game_state.apply_move(scored_move, board_after, racks_after)
