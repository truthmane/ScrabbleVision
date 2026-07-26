"""Ground-truth turn list derived from an official .gcg game record.

Deliberately reuses `training.collect.replay_game.read_gcg_moves` and
`autoscorer.gamelogic.notation.resolve_new_tiles_gcg` rather than
re-deriving anything -- this module must never invent its own notion of
"what actually happened," only reformat the existing, already-validated
one (`tests/unit/test_gcg_replay.py` reproduces all 302 real plays across
13 fixtures exactly). `turn_number` numbers only real plays -- passes,
exchanges, challenges, and the end-of-game bonus line are not plays and
`read_gcg_moves` already skips them, so a game with those in its raw file
doesn't inflate this count.

Pure logic: no cv2, no torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List

from autoscorer.gamelogic.board import Coord
from autoscorer.gamelogic.notation import resolve_new_tiles_gcg
from training.collect.replay_game import read_gcg_moves


@dataclass(frozen=True)
class TruthTurn:
    turn_number: int  # 1-indexed position among real plays only
    player: str
    cells: FrozenSet[Coord]
    letters: Dict[Coord, str]  # the letter actually there; for a blank, the letter it was played as
    blank_cells: FrozenSet[Coord]
    position: str
    turn_score: int
    cumulative_score: int


def load_truth_turns(gcg_path: Path) -> List[TruthTurn]:
    moves = read_gcg_moves(gcg_path)
    turns: List[TruthTurn] = []
    for i, move in enumerate(moves, start=1):
        placements = resolve_new_tiles_gcg(move)
        cells = frozenset(p.coord for p in placements)
        letters = {p.coord: p.letter for p in placements}
        blank_cells = frozenset(p.coord for p in placements if p.is_blank)
        turns.append(TruthTurn(
            turn_number=i,
            player=move.player,
            cells=cells,
            letters=letters,
            blank_cells=blank_cells,
            position=move.position,
            turn_score=move.turn_score,
            cumulative_score=move.cumulative_score,
        ))
    return turns
