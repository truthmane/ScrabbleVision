"""The move-detection state machine the master architecture plan
describes (`IDLE_STILL -> BOARD_SETTLED -> DIFF_COMPUTED ->
CANDIDATE_VALIDATED -> SCORED -> APPLIED`, with `HANDS_OVER_BOARD`
reachable from any state) -- the piece that actually *sequences* the
perception/gamelogic components this project already built and validated
in isolation (stillness gate, temporal-voted classification, constraint
decoding, placement validation, word resolution, scoring, the publish
gateway) into something that can watch a stream of frames, not just score
one already-known move.

Every component `GameWatcher` calls was previously exercised only from a
script or a unit test feeding it hand-picked inputs; nothing in the repo
before this module called them in sequence against a rolling frame
buffer. That gap -- not model accuracy -- was the real blocker to this
being a system rather than a collection of validated parts.

**Honest scope, read before assuming more than this does:**

- Board-camera processing (`observe_board_frame`) is the real thing:
  every state below is genuine, driven by the actual stillness gate,
  temporal voting, constraint decoding, and scoring engine already built
  and tested elsewhere in this repo.
- Rack-camera processing (`record_rack`) is real but deliberately
  simpler: a single-frame read via `board_reader.read_rack`, not
  multi-frame voted like the board path (no rack-specific stillness gate
  exists yet), and not synchronized with the board camera's frame clock
  at all -- call it whenever a rack frame is available, on whatever
  schedule the caller has. Whose turn it is, is supplied by the caller
  (`player_id`), not inferred from vision -- there is no game-clock
  integration here to derive that from.
- **PASS moves can never be detected from vision alone** (nothing changes
  when a player passes) and this module does not attempt to -- that
  requires an external turn-clock signal, out of scope here.
- No cross-camera synchronization exists between a board camera and any
  number of rack cameras -- each is processed independently as its own
  frames arrive. The master plan's "only combine observations once all
  relevant cameras are simultaneously settled" is not implemented.
- Validated so far only against synthetic frame sequences (see
  `tests/unit/test_game_watcher.py`) -- not yet run against a real
  continuous video. Every real-photo validation this project has done
  (WS3, WS5, the rack-detector held-out tests) was single-frame; proving
  this against real continuous footage is the natural next step, not
  something this module can claim on its own.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.movedetect.constraint_decoder import CLASSIFIER_BLANK_LABEL, decode_feasible_reading
from autoscorer.gamelogic.movedetect.validator import validate_placement
from autoscorer.gamelogic.movedetect.word_resolver import words_formed
from autoscorer.gamelogic.models import MoveCandidate, MoveType, ScoredMove
from autoscorer.gamelogic.pool.bag_engine import PoolInvariantViolation, compute_pool_state
from autoscorer.gamelogic.publish import PendingMove, PublishGateway
from autoscorer.gamelogic.scoring.rules_engine import score_move
from autoscorer.perception.board_reader import read_new_cells_voted, read_rack, rack_observations_to_tiles
from autoscorer.perception.calibration.homography import BoardCalibration
from autoscorer.perception.occupancy.detector import DEFAULT_DIFF_THRESHOLD, DEFAULT_GRADIENT_THRESHOLD
from autoscorer.perception.stillness.detector import (
    DEFAULT_MOTION_THRESHOLD,
    DEFAULT_STILL_FRAME_COUNT,
    frame_motion_score,
    stable_window,
)
from training.classify.infer import TileClassifierModel


class WatcherState(str, Enum):
    IDLE_STILL = "IDLE_STILL"
    HANDS_OVER_BOARD = "HANDS_OVER_BOARD"
    BOARD_SETTLED = "BOARD_SETTLED"
    DIFF_COMPUTED = "DIFF_COMPUTED"
    CANDIDATE_VALIDATED = "CANDIDATE_VALIDATED"
    SCORED = "SCORED"
    APPLIED = "APPLIED"


@dataclass(frozen=True)
class WatcherEvent:
    state: WatcherState
    scored_move: Optional[ScoredMove] = None
    confidence: float = 1.0
    needs_operator: bool = False
    reason: Optional[str] = None
    pending: Optional[PendingMove] = None


def _rack_multiset(tiles: Sequence[Tile]) -> Counter:
    return Counter((tile.letter, tile.is_blank) for tile in tiles)


class GameWatcher:
    """Owns the running game state (`board`, per-player `racks`,
    `turn_number`) and the rolling board-camera frame buffer; each call to
    `observe_board_frame`/`record_rack` advances `state` and returns a
    `WatcherEvent` describing what happened, if anything.

    `rack_detector` is optional -- omit it if only board-camera processing
    is needed; `record_rack` raises if called without one.
    """

    def __init__(
        self,
        calibration: BoardCalibration,
        reference_board: np.ndarray,
        classifier: TileClassifierModel,
        publish_gateway: PublishGateway,
        rack_detector=None,
        motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
        still_frame_count: int = DEFAULT_STILL_FRAME_COUNT,
        occupancy_diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
        occupancy_gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    ) -> None:
        self.calibration = calibration
        self.reference_board = reference_board
        self.classifier = classifier
        self.publish_gateway = publish_gateway
        self.rack_detector = rack_detector
        self.motion_threshold = motion_threshold
        self.still_frame_count = still_frame_count
        self.occupancy_diff_threshold = occupancy_diff_threshold
        self.occupancy_gradient_threshold = occupancy_gradient_threshold

        self.board = BoardState()
        self.racks: Dict[str, List[Tile]] = {}
        self.turn_number = 0
        self.state = WatcherState.IDLE_STILL
        self._frame_buffer: List[np.ndarray] = []

    def observe_board_frame(self, frame: np.ndarray, player_id: str) -> WatcherEvent:
        """Feed one sampled board-camera frame in. `player_id` is whose
        turn is currently active -- the caller's responsibility to track
        (alternates deterministically in a 2-player game); vision alone
        can't determine this.
        """
        self._frame_buffer.append(frame)
        cap = self.still_frame_count + 1
        if len(self._frame_buffer) > cap:
            self._frame_buffer = self._frame_buffer[-cap:]

        window = stable_window(self._frame_buffer, self.motion_threshold, self.still_frame_count)
        if window is None:
            moving = (
                len(self._frame_buffer) >= 2
                and frame_motion_score(self._frame_buffer[-2], self._frame_buffer[-1]) > self.motion_threshold
            )
            self.state = WatcherState.HANDS_OVER_BOARD if moving else WatcherState.IDLE_STILL
            return WatcherEvent(state=self.state)

        self.state = WatcherState.BOARD_SETTLED

        candidates = read_new_cells_voted(
            window, self.calibration, self.reference_board, self.classifier, self.board,
            top_k=len(self.classifier.classes),
            diff_threshold=self.occupancy_diff_threshold,
            gradient_threshold=self.occupancy_gradient_threshold,
        )
        if not candidates:
            # Settled, but nothing new since the last processed turn --
            # this is what makes repeated observation of the same stable
            # moment harmless rather than something callers must guard
            # against themselves.
            return WatcherEvent(state=self.state)

        self.state = WatcherState.DIFF_COMPUTED

        decoded = decode_feasible_reading(candidates, self.board, list(self.racks.values()))
        conf_by_coord = {cc.coord: dict(cc.candidates) for cc in candidates}
        min_confidence = min(conf_by_coord[coord][label] for coord, label in decoded.items())

        if any(label == CLASSIFIER_BLANK_LABEL for label in decoded.values()):
            return WatcherEvent(
                state=self.state, confidence=min_confidence, needs_operator=True,
                reason="a detected tile is a blank -- letter unknown until an operator confirms what it's played as",
            )

        placements = {coord: Tile(letter=label, is_blank=False) for coord, label in decoded.items()}
        try:
            board_after = self.board.with_placements(placements)
        except ValueError as exc:
            return WatcherEvent(state=self.state, confidence=min_confidence, needs_operator=True, reason=str(exc))

        new_cells = list(placements.keys())
        validation = validate_placement(self.board, board_after, new_cells)
        if not validation.ok:
            return WatcherEvent(
                state=self.state, confidence=min_confidence, needs_operator=True, reason=validation.reason,
            )

        self.state = WatcherState.CANDIDATE_VALIDATED

        words = words_formed(board_after, new_cells)
        move_score = score_move(board_after, words, new_cells)
        provisional_turn = self.turn_number + 1
        candidate = MoveCandidate(
            turn_number=provisional_turn, player_id=player_id, move_type=MoveType.PLAY,
            new_cells=tuple(new_cells),
        )
        scored_move = ScoredMove(candidate=candidate, move_score=move_score)
        self.state = WatcherState.SCORED

        try:
            compute_pool_state(board_after, list(self.racks.values()))
        except PoolInvariantViolation as exc:
            return WatcherEvent(
                state=self.state, scored_move=scored_move, confidence=min_confidence,
                needs_operator=True, reason=f"pool invariant violated: {exc}",
            )

        if not self.publish_gateway.should_auto_publish(min_confidence):
            pending = PendingMove(
                scored_move=scored_move, board_after=board_after, racks_after={}, confidence=min_confidence,
            )
            return WatcherEvent(
                state=self.state, scored_move=scored_move, confidence=min_confidence,
                needs_operator=True, pending=pending, reason="confidence below gateway threshold",
            )

        self.board = board_after
        self.turn_number = provisional_turn
        self.state = WatcherState.APPLIED
        return WatcherEvent(state=self.state, scored_move=scored_move, confidence=min_confidence)

    def record_rack(
        self, player_id: str, rack_frame: np.ndarray, detection_threshold: float = 0.3,
    ) -> Optional[WatcherEvent]:
        """Reads one player's current rack (single frame, see the module
        docstring's scope notes). The *first* call for a given `player_id`
        just establishes their starting rack silently (returns None) --
        every player starts with a full rack, so that's not an event, it's
        initial knowledge. After that, returns an EXCHANGE event if the
        rack genuinely changed since the last *confirmed* read, or None if
        unchanged (including pure re-arrangement -- order never matters
        for a rack) -- which is most calls, since this doesn't attempt to
        infer whether now is actually a meaningful moment to check.

        A low-confidence read is neither committed to `self.racks` nor
        reported as a confirmed exchange -- exactly the board path's
        policy applied to racks: don't let an unconfirmed read silently
        become part of the tracked game state.
        """
        if self.rack_detector is None:
            raise ValueError("record_rack needs a rack_detector; pass one to GameWatcher.__init__")

        is_first_observation = player_id not in self.racks
        observations = read_rack(rack_frame, self.rack_detector, self.classifier, detection_threshold)
        new_rack = rack_observations_to_tiles(observations)
        previous_rack = self.racks.get(player_id, [])

        if not is_first_observation and _rack_multiset(new_rack) == _rack_multiset(previous_rack):
            return None

        min_confidence = min((obs.confidence for obs in observations), default=1.0)
        if not self.publish_gateway.should_auto_publish(min_confidence):
            return WatcherEvent(
                state=self.state, confidence=min_confidence, needs_operator=True,
                reason="rack read confidence below gateway threshold",
            )

        self.racks[player_id] = new_rack
        if is_first_observation:
            return None

        self.turn_number += 1
        candidate = MoveCandidate(turn_number=self.turn_number, player_id=player_id, move_type=MoveType.EXCHANGE)
        scored_move = ScoredMove(candidate=candidate, move_score=None)
        self.state = WatcherState.APPLIED
        return WatcherEvent(state=self.state, scored_move=scored_move, confidence=min_confidence)
