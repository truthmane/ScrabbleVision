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
  `tests/unit/test_game_watcher.py`) and against real broadcast footage
  run through short clips (see `perception/capture/run_watcher.py` and
  `docs/`/README status notes) -- not yet a complete real game
  end-to-end.
- **Standalone by default, but can delegate to a live `GameSession`.**
  Pass `session=` and this stops tracking its own board/racks/turn
  numbers/gateway decision -- every detected move goes through
  `GameSession.submit_move` instead, exactly the same call path a human
  operator's typed-in move already takes (`POST /moves`). That's the
  point: a CV-detected move and an operator-typed move become
  indistinguishable once they reach the session, so they show up in the
  same `/pending` list, the same overlay, and the same operator-approval
  flow, rather than perception output going nowhere a human can act on
  it. `GameSession` lives under `autoscorer/api/` in this repo's current
  layout but is itself ASGI-free (see its own docstring) -- importing it
  here is a real, deliberate layering call, not an accident: it's the
  connective tissue this project was missing between "a validated
  pipeline" and "a product."
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np

from autoscorer.api.session import GameSession
from autoscorer.gamelogic.board import BoardState, Coord, Tile
from autoscorer.gamelogic.movedetect.constraint_decoder import CLASSIFIER_BLANK_LABEL, decode_feasible_reading
from autoscorer.gamelogic.movedetect.validator import validate_placement
from autoscorer.gamelogic.movedetect.word_resolver import words_formed
from autoscorer.gamelogic.models import MoveCandidate, MoveProcessingError, MoveType, ScoredMove
from autoscorer.gamelogic.pool.bag_engine import PoolInvariantViolation, compute_pool_state
from autoscorer.gamelogic.publish import PendingMove, PublishGateway
from autoscorer.gamelogic.scoring.rules_engine import score_move
from autoscorer.perception.board_reader import read_new_cells_voted, read_rack, rack_observations_to_tiles
from autoscorer.perception.calibration.homography import BoardCalibration
from autoscorer.perception.occupancy.detector import DEFAULT_DIFF_THRESHOLD, DEFAULT_GRADIENT_THRESHOLD, ReferenceBoard
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


def _cluster_cells(cells: frozenset, board_before: BoardState) -> List[frozenset]:
    """Groups new-cell coordinates into clusters connected through
    contiguous occupied cells.

    A single word's new tiles aren't always directly adjacent to each
    OTHER -- a play can hook through an existing tile in the middle (e.g.
    "ARB.RIZE", where the "." is a letter already on the board), which
    would otherwise split one legal word's new cells into two groups that
    never touch. Two new cells belong to the same cluster if they're
    reachable through a straight run of occupied cells (new or already on
    `board_before`) with no empty gap -- the same notion of "connected"
    `word_resolver.run_through`/`validate_placement` already use, just
    computed before any of these cells have actually been placed.

    Clustering still matters in general: `board_before` can genuinely
    have more than one turn's worth of unaccounted-for tiles at once --
    e.g. a slow multi-tile word still being placed in one part of the
    board while a separate, unrelated cell elsewhere is also new -- and
    lumping unrelated cells into a single set would never validate as one
    legal line.
    """
    remaining = set(cells)
    clusters: List[frozenset] = []
    while remaining:
        seed = next(iter(remaining))
        cluster = {seed}
        frontier = [seed]
        remaining.discard(seed)
        while frontier:
            r, c = frontier.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                # Walk through a run of already-occupied cells (old tiles
                # this play hooks through) to find the next new cell in
                # this direction, if any.
                while (nr, nc) not in cells and board_before.get((nr, nc)) is not None:
                    nr, nc = nr + dr, nc + dc
                if (nr, nc) in remaining:
                    remaining.discard((nr, nc))
                    cluster.add((nr, nc))
                    frontier.append((nr, nc))
        clusters.append(frozenset(cluster))
    return clusters


class GameWatcher:
    """Owns the rolling board-camera frame buffer; each call to
    `observe_board_frame`/`record_rack` advances `state` and returns a
    `WatcherEvent` describing what happened, if anything.

    Two modes, chosen by whether `session` is given:

    - **Standalone** (`session=None`, the default): tracks its own
      `board`/`racks`/`turn_number` and makes its own auto-publish
      decision via the required `publish_gateway`. Simplest to construct
      and test in isolation (see `tests/unit/test_game_watcher.py`).
    - **Delegated** (`session=<GameSession>`): `board`/`racks`/
      `turn_number` become read-only views onto `session.game_state`, and
      every detected move is submitted through `session.submit_move` --
      the exact path a human operator's typed-in move already takes.
      `publish_gateway` is unused in this mode (the session has its own).
      This is what makes a detected move show up in the real product's
      `/pending` list, overlay, and operator-approval flow instead of
      going nowhere.

    `rack_detector` is optional -- omit it if only board-camera processing
    is needed; `record_rack` raises if called without one.
    """

    def __init__(
        self,
        calibration: BoardCalibration,
        reference_board: ReferenceBoard,
        classifier: TileClassifierModel,
        publish_gateway: Optional[PublishGateway] = None,
        session: Optional[GameSession] = None,
        rack_detector=None,
        motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
        still_frame_count: int = DEFAULT_STILL_FRAME_COUNT,
        occupancy_diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
        occupancy_gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD,
    ) -> None:
        if session is None and publish_gateway is None:
            raise ValueError(
                "GameWatcher needs either publish_gateway (standalone mode) or "
                "session (delegated mode) -- see the class docstring"
            )

        self.calibration = calibration
        self.reference_board = reference_board
        self.classifier = classifier
        self.publish_gateway = publish_gateway
        self.session = session
        self.rack_detector = rack_detector
        self.motion_threshold = motion_threshold
        self.still_frame_count = still_frame_count
        self.occupancy_diff_threshold = occupancy_diff_threshold
        self.occupancy_gradient_threshold = occupancy_gradient_threshold

        self._board = BoardState()
        self._racks: Dict[str, List[Tile]] = {}
        self._turn_number = 0
        self.state = WatcherState.IDLE_STILL
        self._frame_buffer: List[np.ndarray] = []
        # A settled read is never committed the moment it's first seen --
        # only once a cell has appeared as a new candidate in two
        # consecutive settled observations does it count as confirmed.
        # A player who places part of a word, then pauses to think (fully
        # satisfying "hands not moving" while genuinely mid-turn), would
        # otherwise get that partial placement committed as if it were the
        # whole move -- this is exactly what split one real move into two
        # turns (see the "HUIA" fragmentation found running this against a
        # real, complete game).
        #
        # Confirmation is tracked per CELL, not as one atomic set-equality
        # check on the whole reading, because real footage also showed a
        # single marginal/borderline-visibility tile can keep flickering
        # in and out of the candidate set indefinitely -- an exact-set
        # match would never fire while that one cell is unsettled, even
        # though every other cell in the placement is rock solid. Tracking
        # confirmation per cell lets a stable subset commit without
        # waiting forever on a cell that never stabilizes, while a
        # genuinely new cell still has to survive one full extra
        # observation, same as before, before it's trusted.
        self._confirmed_cells: frozenset = frozenset()
        self._last_candidate_cells: frozenset = frozenset()

    @property
    def board(self) -> BoardState:
        return self.session.game_state.board if self.session is not None else self._board

    @property
    def racks(self) -> Dict[str, List[Tile]]:
        return self.session.game_state.racks if self.session is not None else self._racks

    @property
    def turn_number(self) -> int:
        if self.session is not None:
            return len(self.session.game_state.history)
        return self._turn_number

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
            self._confirmed_cells = frozenset()
            self._last_candidate_cells = frozenset()
            return WatcherEvent(state=self.state)

        current_cells = frozenset(cc.coord for cc in candidates)
        newly_confirmed = current_cells & self._last_candidate_cells
        # Intersecting with current_cells means a cell that stops
        # appearing (a marginal one that turned out not to be real, or a
        # transient blip) drops back out on its own -- it never blocks a
        # stable subset from committing, and it isn't permanently baked in
        # either.
        self._confirmed_cells = (self._confirmed_cells | newly_confirmed) & current_cells
        self._last_candidate_cells = current_cells

        # board_before can genuinely have more than one turn's worth of
        # unaccounted-for tiles new at once (e.g. a slow multi-tile word
        # still being placed in one part of the board while a separate,
        # already-stable cell elsewhere is also new) -- clustering by
        # adjacency before checking confirmation means one still-forming
        # cluster elsewhere never prevents a different, fully-confirmed
        # cluster from being acted on. Only ever act on ONE cluster per
        # call, matching the one-turn-per-observation model; if more than
        # one happens to be fully confirmed simultaneously, the rest stay
        # exactly as confirmed as they are now and get picked up (now
        # against an updated board_before) on the next call.
        ready_cluster = next(
            (cluster for cluster in _cluster_cells(current_cells, self.board) if cluster and cluster <= self._confirmed_cells),
            None,
        )
        if ready_cluster is None:
            # No cluster has (yet) had every one of its cells survive a
            # second independent look -- could be a completed turn, or a
            # player mid-placement who just happened to pause, or one
            # cell whose visibility is still marginal. Don't act yet.
            self.state = WatcherState.DIFF_COMPUTED
            return WatcherEvent(state=self.state)

        # Confirmed: every cell in this cluster has now been seen in (at
        # least) two consecutive settled observations -- the placement
        # has stopped growing, so it's safe to treat as a genuinely
        # completed turn. Confirmation state is deliberately NOT cleared
        # here yet -- only once we actually reach a scored outcome below.
        # A confirmed cluster that fails validation/pool-check for an
        # unrelated reason should stay confirmed for the next attempt,
        # rather than forcing the whole two-observation wait to repeat
        # from scratch every single time a retry fails.
        self.state = WatcherState.DIFF_COMPUTED
        candidates = [cc for cc in candidates if cc.coord in ready_cluster]

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

        try:
            compute_pool_state(board_after, list(self.racks.values()))
        except PoolInvariantViolation as exc:
            return WatcherEvent(
                state=self.state, confidence=min_confidence,
                needs_operator=True, reason=f"pool invariant violated: {exc}",
            )

        # Every check passed -- this reading is genuinely being acted on,
        # so the next observation starts a fresh confirmation cycle.
        self._confirmed_cells = frozenset()
        self._last_candidate_cells = frozenset()

        if self.session is not None:
            return self._submit_play_via_session(decoded, player_id, min_confidence)

        words = words_formed(board_after, new_cells)
        move_score = score_move(board_after, words, new_cells)
        provisional_turn = self._turn_number + 1
        candidate = MoveCandidate(
            turn_number=provisional_turn, player_id=player_id, move_type=MoveType.PLAY,
            new_cells=tuple(new_cells),
        )
        scored_move = ScoredMove(candidate=candidate, move_score=move_score)
        self.state = WatcherState.SCORED

        if not self.publish_gateway.should_auto_publish(min_confidence):
            pending = PendingMove(
                scored_move=scored_move, board_after=board_after, racks_after={}, confidence=min_confidence,
            )
            return WatcherEvent(
                state=self.state, scored_move=scored_move, confidence=min_confidence,
                needs_operator=True, pending=pending, reason="confidence below gateway threshold",
            )

        self._board = board_after
        self._turn_number = provisional_turn
        self.state = WatcherState.APPLIED
        return WatcherEvent(state=self.state, scored_move=scored_move, confidence=min_confidence)

    def _submit_play_via_session(
        self, decoded: Dict[Coord, str], player_id: str, min_confidence: float,
    ) -> WatcherEvent:
        """The delegated-mode path: hands the decoded reading to
        `GameSession.submit_move` -- the same call a human operator's
        typed-in move already goes through -- instead of this class
        re-implementing scoring/gateway/apply bookkeeping a second time.
        Placement validity and the pool invariant were already checked by
        the caller before reaching here (against `self.board`, the same
        board `submit_move` will use), so `MoveProcessingError` shouldn't
        normally trigger -- handled anyway since `submit_move`'s contract
        allows it.
        """
        new_tiles = [(coord, label, False) for coord, label in decoded.items()]
        result = self.session.submit_move(player_id, new_tiles=new_tiles, confidence=min_confidence)

        if isinstance(result.outcome, MoveProcessingError):
            return WatcherEvent(
                state=self.state, confidence=min_confidence, needs_operator=True, reason=result.outcome.reason,
            )

        self.state = WatcherState.APPLIED if result.published else WatcherState.SCORED
        turn_number = result.outcome.candidate.turn_number
        pending = None if result.published else self.session.pending.get(turn_number)
        return WatcherEvent(
            state=self.state, scored_move=result.outcome, confidence=min_confidence,
            needs_operator=not result.published, pending=pending,
            reason=None if result.published else "confidence below gateway threshold",
        )

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

        if is_first_observation:
            # Establishing a player's starting rack isn't a move -- every
            # player starts with a full rack, so there's no gateway
            # decision or turn to record, in either mode.
            self.racks[player_id] = new_rack
            return None

        if self.session is not None:
            return self._submit_exchange_via_session(player_id, new_rack, min_confidence)

        if not self.publish_gateway.should_auto_publish(min_confidence):
            return WatcherEvent(
                state=self.state, confidence=min_confidence, needs_operator=True,
                reason="rack read confidence below gateway threshold",
            )

        self.racks[player_id] = new_rack
        self._turn_number += 1
        candidate = MoveCandidate(turn_number=self._turn_number, player_id=player_id, move_type=MoveType.EXCHANGE)
        scored_move = ScoredMove(candidate=candidate, move_score=None)
        self.state = WatcherState.APPLIED
        return WatcherEvent(state=self.state, scored_move=scored_move, confidence=min_confidence)

    def _submit_exchange_via_session(
        self, player_id: str, new_rack: List[Tile], min_confidence: float,
    ) -> WatcherEvent:
        rack_after = [(tile.letter, tile.is_blank) for tile in new_rack]
        result = self.session.submit_move(player_id, rack_after=rack_after, confidence=min_confidence)

        if isinstance(result.outcome, MoveProcessingError):
            return WatcherEvent(
                state=self.state, confidence=min_confidence, needs_operator=True, reason=result.outcome.reason,
            )

        self.state = WatcherState.APPLIED if result.published else WatcherState.SCORED
        turn_number = result.outcome.candidate.turn_number
        pending = None if result.published else self.session.pending.get(turn_number)
        return WatcherEvent(
            state=self.state, scored_move=result.outcome, confidence=min_confidence,
            needs_operator=not result.published, pending=pending,
            reason=None if result.published else "rack read confidence below gateway threshold",
        )
