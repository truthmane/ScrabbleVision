"""The autonomous/manual publish gate. A `ScoredMove` always exists as soon
as the state machine produces it; this module decides whether it goes
straight to the canonical GameState (and thus the overlay) or waits in a
queue for an operator's explicit approval.

Deliberately a single runtime-mutable `PublishGateway.mode`, not a
build-time flag or separate code path -- an operator can flip modes
mid-game without restarting anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.models import ScoredMove


class PublishMode(str, Enum):
    AUTONOMOUS = "AUTONOMOUS"
    MANUAL = "MANUAL"
    AUTONOMOUS_WITH_CONFIDENCE_FALLBACK = "AUTONOMOUS_WITH_CONFIDENCE_FALLBACK"


@dataclass(frozen=True)
class PendingMove:
    """A scored move awaiting an operator decision, plus everything needed
    to actually apply it to the GameState once approved."""
    scored_move: ScoredMove
    board_after: BoardState
    racks_after: Dict[str, list]
    confidence: float
    # JPEG-encoded bytes of the board frame the detection came from, if
    # the caller had one (vision-detected moves do; a human's typed-in
    # submission doesn't) -- lets an operator see what the model actually
    # saw instead of reviewing a bare row/col/letter list blind.
    source_frame_jpeg: Optional[bytes] = None


@dataclass(frozen=True)
class OperatorDecision:
    turn_number: int
    action: str  # "approve" | "reject"
    operator_id: str = "operator"


class PublishGateway:
    """Decides, given the active mode and a move's confidence, whether a
    move auto-publishes or must wait for operator approval. Confidence is
    always 1.0 for operator-entered moves (Phase 2); it becomes meaningful
    once camera-derived confidence scores exist (Phase 4+).
    """

    def __init__(self, mode: PublishMode = PublishMode.MANUAL, confidence_threshold: float = 0.9) -> None:
        self.mode = mode
        self.confidence_threshold = confidence_threshold

    def should_auto_publish(self, confidence: float = 1.0) -> bool:
        if self.mode == PublishMode.AUTONOMOUS:
            return True
        if self.mode == PublishMode.MANUAL:
            return False
        if self.mode == PublishMode.AUTONOMOUS_WITH_CONFIDENCE_FALLBACK:
            return confidence >= self.confidence_threshold
        raise ValueError(f"unknown publish mode: {self.mode}")
