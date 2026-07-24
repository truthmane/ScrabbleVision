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
from typing import Dict, List, Sequence

from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.models import ScoredMove
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

    def pool_state(self) -> PoolState:
        return compute_pool_state(self.board, list(self.racks.values()))
