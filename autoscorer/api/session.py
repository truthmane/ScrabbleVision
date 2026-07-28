"""Orchestrates a single live game: wraps GameState + PublishGateway with the
operations the API layer needs (submit a move, list/decide on pending moves,
change publish mode). Kept free of any ASGI/FastAPI concepts so it's testable
with plain function calls -- the web layer is a thin adapter around this.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Union

from autoscorer.gamelogic.board import BoardState, Coord, Tile
from autoscorer.gamelogic.eventlog.store import GameState
from autoscorer.gamelogic.models import MoveProcessingError, MoveType, ScoredMove
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
        # (player_id, sorted (coord, letter, is_blank) tuples) for every
        # reading an operator has explicitly rejected -- see submit_move's
        # check below for why this exists: without it, a detector that
        # keeps sampling the same unchanged (mis-detected) board just
        # re-submits the identical wrong reading forever, since nothing
        # about the source pixels ever changes on its own.
        self._rejected_signatures: set = set()

    def set_mode(self, mode: PublishMode) -> None:
        self.gateway.mode = mode

    def submit_move(
        self,
        player_id: str,
        new_tiles: Optional[List[NewTile]] = None,
        rack_after: Optional[List[RackTile]] = None,
        confidence: float = 1.0,
        source_frame_jpeg: Optional[bytes] = None,
    ) -> SubmissionResult:
        game_state = self.game_state
        if player_id not in game_state.racks:
            game_state.racks[player_id] = []

        if new_tiles and self._signature(player_id, new_tiles) in self._rejected_signatures:
            # Identical cells + letters + player as something an operator
            # already rejected -- don't consume a turn number or re-queue
            # it, since nothing about the input changed and re-offering it
            # is just noise (see the module-level note on
            # `_rejected_signatures`). A caller detecting moves from an
            # unchanging board will keep hitting this until either the
            # board genuinely changes or the operator corrects it another
            # way (e.g. the notation endpoint), at which point the new
            # signature won't match and this stops applying.
            return SubmissionResult(
                outcome=MoveProcessingError(
                    reason="identical to a previously rejected reading; not re-queuing until the board changes",
                ),
                published=False,
            )

        board_before = game_state.board
        if new_tiles:
            placements = {coord: Tile(letter, is_blank) for (coord, letter, is_blank) in new_tiles}
            try:
                board_after = board_before.with_placements(placements)
            except ValueError as exc:
                # An impossible placement (an occupied cell, or an
                # unplayed blank with no letter) -- don't consume a turn
                # number, same treatment as a MoveProcessingError below, so
                # the operator can correct the input and resubmit.
                return SubmissionResult(outcome=MoveProcessingError(reason=str(exc)), published=False)
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

        # GCG export needs the acting player's rack as it stood before this
        # move (and, for an exchange, exactly which tiles left it) -- data
        # only available here, not inside process_turn, which never learns
        # which rack in racks_before/racks_after belongs to `player_id`.
        rack_before_player = tuple(game_state.racks.get(player_id, []))
        exchanged_tiles: Tuple[Tile, ...] = ()
        if result.candidate.move_type == MoveType.EXCHANGE:
            rack_after_player = tuple(racks_after_map[player_id])
            exchanged_tiles = tuple((Counter(rack_before_player) - Counter(rack_after_player)).elements())
        result = ScoredMove(
            candidate=replace(result.candidate, rack_before=rack_before_player, exchanged_tiles=exchanged_tiles),
            move_score=result.move_score,
        )

        if self.gateway.should_auto_publish(confidence):
            self._apply(result, board_after, racks_after_map)
            return SubmissionResult(outcome=result, published=True)

        self.pending[turn_number] = PendingMove(
            scored_move=result, board_after=board_after, racks_after=racks_after_map, confidence=confidence,
            source_frame_jpeg=source_frame_jpeg,
        )
        return SubmissionResult(outcome=result, published=False)

    def decide(self, turn_number: int, action: str) -> Optional[bool]:
        """Approve or reject a pending move.

        Returns `None` if no such pending move exists (already decided, or
        an invalid turn number) -- the caller should treat that as "not
        found". Otherwise returns whether the move was applied to the
        canonical GameState (True for approve, False for a reject, which
        simply drops it so the operator can resubmit a correction).

        A reject also remembers this exact reading (see
        `_rejected_signatures`) so a detector still looking at the same
        unchanged board doesn't just re-queue the identical wrong answer.
        """
        pending = self.pending.pop(turn_number, None)
        if pending is None:
            return None
        if action == "approve":
            self._apply(pending.scored_move, pending.board_after, pending.racks_after)
            return True

        candidate = pending.scored_move.candidate
        new_tiles = tuple(
            (coord, tile.letter, tile.is_blank)
            for coord in sorted(candidate.new_cells)
            for tile in [pending.board_after.get(coord)]
            if tile is not None
        )
        if new_tiles:
            self._rejected_signatures.add(self._signature(candidate.player_id, new_tiles))
        return False

    def undo_last_move(self) -> Optional[ScoredMove]:
        """Reverts the most recently committed move -- see
        `GameState.undo_last` for exactly what this can and can't do."""
        return self.game_state.undo_last()

    def list_pending(self) -> List[PendingMove]:
        return [self.pending[k] for k in sorted(self.pending)]

    @staticmethod
    def _signature(player_id: str, new_tiles: List[NewTile]) -> tuple:
        return (player_id, tuple(sorted(new_tiles)))

    def _apply(self, scored_move: ScoredMove, board_after: BoardState, racks_after: Dict[str, list]) -> None:
        self.game_state.apply_move(scored_move, board_after, racks_after)
