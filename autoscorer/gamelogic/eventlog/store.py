"""In-memory canonical game state: current board/racks/scores plus an
append-only history of applied moves. This is the single place turn results
get applied, so scores and pool derivation always read from one consistent
source of truth.

Persistence (SQLite, per the architecture plan) is deliberately left out of
this first pass -- nothing downstream needs it yet, and adding it now would
be plumbing without a caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.models import MoveType, ScoredMove
from autoscorer.gamelogic.pool.bag_engine import PoolState, compute_pool_state


@dataclass
class GameState:
    board: BoardState = field(default_factory=BoardState)
    racks: Dict[str, List[Tile]] = field(default_factory=dict)
    scores: Dict[str, int] = field(default_factory=dict)
    history: List[ScoredMove] = field(default_factory=list)

    def apply_move(
        self,
        scored_move: ScoredMove,
        board_after: BoardState,
        racks_after: Dict[str, Sequence[Tile]],
    ) -> None:
        """Commit a confirmed move: update board/racks, add to the acting
        player's score (if any), and append to the history log.
        """
        self.board = board_after
        self.racks = {player: list(rack) for player, rack in racks_after.items()}
        if scored_move.move_score is not None:
            player = scored_move.candidate.player_id
            self.scores[player] = self.scores.get(player, 0) + scored_move.move_score.total
        self.history.append(scored_move)

    def undo_last(self) -> Optional[ScoredMove]:
        """Reverts the single most recently applied move: removes its
        placed tiles from the board and subtracts its score, then drops it
        from history. Returns the undone move, or None if history is empty.

        Only ever reverses the tail -- undoing an arbitrary earlier turn
        isn't supported, since a later move may have scored a cross-word
        through cells an earlier one placed, and silently invalidating
        that isn't this method's call to make. To fix an older mistake,
        undo back to it one call at a time, then resubmit the moves in
        between.

        Racks are deliberately left untouched: board-only detection (the
        only source of committed moves so far) never populates real rack
        contents in the first place, so there's nothing meaningful to
        revert there.
        """
        if not self.history:
            return None
        scored_move = self.history.pop()
        candidate = scored_move.candidate
        if candidate.move_type == MoveType.PLAY and candidate.new_cells:
            self.board = self.board.without_cells(candidate.new_cells)
        if scored_move.move_score is not None:
            player = candidate.player_id
            self.scores[player] = self.scores.get(player, 0) - scored_move.move_score.total
        return scored_move

    def pool_state(self) -> PoolState:
        return compute_pool_state(self.board, list(self.racks.values()))
