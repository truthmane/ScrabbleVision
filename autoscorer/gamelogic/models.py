"""Shared event contracts passed between the move-detection state machine,
the event log, and (in later phases) the operator UI / overlay. Kept as
plain, serializable dataclasses so they can cross a process boundary later
without any rework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from autoscorer.gamelogic.board import Coord, Tile
from autoscorer.gamelogic.scoring.rules_engine import MoveScore


class MoveType(str, Enum):
    PLAY = "PLAY"
    EXCHANGE = "EXCHANGE"
    PASS = "PASS"


@dataclass(frozen=True)
class MoveCandidate:
    turn_number: int
    player_id: str
    move_type: MoveType
    new_cells: Tuple[Coord, ...] = field(default_factory=tuple)
    blank_cells: Tuple[Coord, ...] = field(default_factory=tuple)
    """Subset of `new_cells` played as a blank -- GCG export needs this to
    lowercase the right letters; nothing else in the codebase reads it."""
    rack_before: Tuple[Tile, ...] = field(default_factory=tuple)
    """The acting player's rack immediately before this move. Populated by
    `GameSession.submit_move` (which has the rack history), not by
    `process_turn` (which doesn't track player identity across racks)."""
    exchanged_tiles: Tuple[Tile, ...] = field(default_factory=tuple)
    """Tiles that left the rack this turn -- only meaningful/populated for
    an EXCHANGE move."""


@dataclass(frozen=True)
class ScoredMove:
    candidate: MoveCandidate
    move_score: Optional[MoveScore]  # None for EXCHANGE / PASS


@dataclass(frozen=True)
class MoveProcessingError:
    reason: str
