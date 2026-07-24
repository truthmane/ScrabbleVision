"""Shared event contracts passed between the move-detection state machine,
the event log, and (in later phases) the operator UI / overlay. Kept as
plain, serializable dataclasses so they can cross a process boundary later
without any rework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from autoscorer.gamelogic.board import Coord
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


@dataclass(frozen=True)
class ScoredMove:
    candidate: MoveCandidate
    move_score: Optional[MoveScore]  # None for EXCHANGE / PASS


@dataclass(frozen=True)
class MoveProcessingError:
    reason: str
