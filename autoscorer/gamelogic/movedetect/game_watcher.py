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
- **Never-jam commit search.** A confirmed set of new cells can yield
  more than one legal candidate placement (see
  `placement_search.enumerate_candidate_placements`) -- every one is
  tried, ranked by cell count first, until one commits or none do,
  instead of picking a single cluster and giving up on the whole
  observation if it fails. A cell that keeps failing across
  `FAILURE_QUARANTINE_THRESHOLD` observations is quarantined (excluded
  from consideration) for `QUARANTINE_TTL_OBSERVATIONS` observations,
  which is what stops a single problematic cell (a persistent false
  occupancy reading, or a genuinely blank tile this state machine cannot
  resolve alone) from blocking every other, unrelated placement on the
  board for the rest of the game -- found running this against a
  complete real game: a non-collinear cluster produced the identical
  `'new tiles must lie in a single row or column'` rejection for **163
  consecutive settled observations**, because nothing else was ever
  tried. After `STALL_OBSERVATIONS` observations with no commit at all,
  a `STALLED` watchdog event fires and force-quarantines every confirmed
  cell with any failure history, as a backstop beyond per-cell
  quarantine alone.
- Rack-camera processing (`record_rack`) is real but deliberately
  simpler than the board path: each player's rack frames are buffered
  and gated through the same stillness check (`stable_window`) the board
  path uses, so a rack mid-rearrangement (a hand moving tiles around)
  doesn't get read as a settled state -- but there's no cross-frame
  temporal voting once settled (unlike the board path's
  `read_new_cells_voted`), since voting per detected tile would first
  need to match detections across frames (a rack has no fixed grid to
  vote against, unlike board cells), which isn't built. Not synchronized
  with the board camera's frame clock at all -- call it whenever a rack
  frame is available, on whatever schedule the caller has. Whose turn it
  is, is supplied by the caller (`player_id`), not inferred from vision
  -- there is no game-clock integration here to derive that from.
- **PASS moves can never be detected from vision alone** (nothing changes
  when a player passes) and this module does not attempt to -- that
  requires an external turn-clock signal, out of scope here.
- No cross-camera synchronization exists between a board camera and any
  number of rack cameras -- each is processed independently as its own
  frames arrive. The master plan's "only combine observations once all
  relevant cameras are simultaneously settled" is not implemented.
- Validated against synthetic frame sequences (`tests/unit/test_game_watcher.py`
  and `tests/slow/test_synthetic_full_game.py`, the latter replaying an
  entire real 21-move game's ground truth with a controllable simulated
  classifier accuracy) and against a complete real ~37-minute broadcast
  (`perception/capture/run_watcher.py`, `autoscorer/eval/run_game_eval.py`,
  and `docs/`/README status notes) -- not just short clips.
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

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from autoscorer.api.session import GameSession
from autoscorer.gamelogic.board import BOARD_SIZE, BoardState, Coord, Tile
from autoscorer.gamelogic.dictionary.lexicon import Lexicon
from autoscorer.gamelogic.movedetect.constraint_decoder import CellCandidates, CLASSIFIER_BLANK_LABEL
from autoscorer.gamelogic.movedetect.lexicon_decoder import decode_with_lexicon
from autoscorer.gamelogic.movedetect.placement_search import (
    CandidatePlacement,
    _cluster_cells,
    enumerate_candidate_placements,
)
from autoscorer.gamelogic.movedetect.validator import validate_placement
from autoscorer.gamelogic.movedetect.word_resolver import words_formed
from autoscorer.gamelogic.models import MoveCandidate, MoveProcessingError, MoveType, ScoredMove
from autoscorer.gamelogic.notation import format_square
from autoscorer.gamelogic.pool.bag_engine import PoolInvariantViolation, compute_pool_state
from autoscorer.gamelogic.publish import PendingMove, PublishGateway
from autoscorer.gamelogic.scoring.rules_engine import score_move
from autoscorer.perception.board_reader import read_new_cells_voted, read_rack, rack_observations_to_tiles
from autoscorer.perception.calibration.homography import BoardCalibration, cell_bounds, crop_cell, crop_cell_inset
from autoscorer.perception.classify.blank_heuristic import looks_smooth_like_a_blank
from autoscorer.perception.occupancy.adaptive import AdaptiveOccupancyTracker
from autoscorer.perception.occupancy.detector import (
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_GRADIENT_THRESHOLD,
    ReferenceBoard,
    occupancy_scores,
)
from autoscorer.perception.stillness.detector import (
    DEFAULT_MOTION_THRESHOLD,
    DEFAULT_STILL_FRAME_COUNT,
    StillnessTracker,
)
from training.classify.infer import TileClassifierModel


def _encode_frame_jpeg(frame: Optional[np.ndarray]) -> Optional[bytes]:
    """Best-effort JPEG encode for attaching to a pending move so an
    operator can see what the model actually saw -- never lets an encode
    failure break move detection itself, since the frame is a nice-to-have
    for review, not something anything downstream depends on."""
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None


class WatcherState(str, Enum):
    IDLE_STILL = "IDLE_STILL"
    HANDS_OVER_BOARD = "HANDS_OVER_BOARD"
    BOARD_SETTLED = "BOARD_SETTLED"
    DIFF_COMPUTED = "DIFF_COMPUTED"
    CANDIDATE_VALIDATED = "CANDIDATE_VALIDATED"
    SCORED = "SCORED"
    APPLIED = "APPLIED"
    STALLED = "STALLED"


@dataclass(frozen=True)
class WatcherEvent:
    state: WatcherState
    scored_move: Optional[ScoredMove] = None
    confidence: float = 1.0
    needs_operator: bool = False
    reason: Optional[str] = None
    pending: Optional[PendingMove] = None
    attempted_cells: Tuple[Coord, ...] = ()
    """The cell set a failed attempt tried to commit -- populated on
    `needs_operator` failure paths so a caller (the eval harness, a stall
    report) can name which squares are involved without re-deriving it
    from a log. Empty on any event that isn't reporting a failed attempt."""
    is_stall: bool = False
    """True only for a watchdog event emitted after many consecutive
    observations produced no commit at all -- see `STALL_OBSERVATIONS` in
    the module docstring's stall-handling notes."""


def _rack_multiset(tiles: Sequence[Tile]) -> Counter:
    return Counter((tile.letter, tile.is_blank) for tile in tiles)


# Weight given to each new frame when refreshing the adaptive reference
# (see GameWatcher._refresh_reference_for_still_empty_cells) -- low
# enough that one noisy frame can't singlehandedly redefine what
# "empty" looks like for a cell, high enough that genuine drift across
# many real settled observations gets absorbed in a reasonable number of
# them rather than never.
_REFERENCE_EMA_ALPHA = 0.3

# Never-jam commit search tuning (see the module docstring's "Never-jam
# commit search" bullet). All four are observation counts, not frame
# counts or wall-clock time -- one "observation" is one settled call to
# `observe_board_frame` that had at least one confirmed new cell to
# consider.
FAILURE_QUARANTINE_THRESHOLD = 3
"""A cell that participates in this many failed commit attempts (across
this many distinct observations, at most one increment per cell per
observation) gets quarantined -- excluded from candidate enumeration --
rather than being retried forever."""
QUARANTINE_TTL_OBSERVATIONS = 20
"""How long a quarantined cell stays excluded before it's reconsidered.
Time-limited, not permanent: a real tile that was quarantined for the
wrong reason (e.g. a neighbor's fault, not its own) gets another chance
against a since-advanced board; a genuinely persistent problem simply
re-quarantines cheaply after failing again."""
QUARANTINE_HEAL_DELAY = 2
"""Observations a cell must stay quarantined before the adaptive
reference is allowed to heal it (see
`_refresh_reference_for_still_empty_cells`) -- keeps a cell that was
JUST quarantined (which could still be a real tile mid-confirmation, not
yet given a fair chance) from being instantly baked into the background."""
QUARANTINE_HEAL_ALPHA = 0.1
"""EMA weight used only for a healing (quarantined) cell -- lower than
`_REFERENCE_EMA_ALPHA` since a mistaken heal is a real cost (a cell that
never should have been treated as background), so it should take longer
to matter than ordinary lighting-drift refresh does."""
STALL_OBSERVATIONS = 30
"""Backstop watchdog: if this many observations pass with no commit at
all (even though per-cell quarantine is already excluding known-bad
cells), something is still genuinely stuck -- emit a `STALLED` event and
force-quarantine every confirmed cell with any failure history, rather
than continuing to retry silently forever."""

SOFT_CELL_MIN_CONFIDENCE = 0.7
"""A SOFT cell (see `read_new_cells_voted`) is only usable as an
in-line extension of a HARD run if its own temporal-voted top label also
clears this confidence floor -- majority/all-but-one occupancy support
alone isn't enough. A real full-game run found that a soft-extended
candidate is ranked *ahead* of the plain HARD one (it's longer -- see
the ranking comment below), and phonies being legal means a nonsense
extension can't be rejected on spelling alone; if the extension is real
content and not just an occupancy near-miss, an ordinary letter should
still be visible enough to the classifier that its own confidence says
so. Raised from an initial 0.5 after a full real-game comparison showed
that bar still let through more spurious detections than it recovered
real ones; not yet re-tuned beyond removing that regression's mechanism
-- treat as a starting point, same as the occupancy thresholds."""


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
        adaptive_reference: bool = True,
        lexicon: Optional[Lexicon] = None,
        adaptive_occupancy_tracker: Optional[AdaptiveOccupancyTracker] = None,
    ) -> None:
        """`adaptive_occupancy_tracker`, if given, applies per-cell
        hysteresis and adaptive statistical thresholding (WS3 items 3-4,
        see `perception.occupancy.adaptive`) on top of the existing
        vote-based HARD/SOFT occupancy tiers: a cell only counts as a new
        HARD candidate if BOTH the existing per-window vote AND the
        tracker's own debounced, per-cell-noise-aware decision currently
        agree it's occupied. `None` (the default) disables this entirely
        -- behavior is then byte-for-byte identical to before this
        existed, since only one real venue has ever been tuned and never
        against these newer signals. Only supported with a single
        (non-list) `reference_board`, same restriction as
        `adaptive_reference`.

        `lexicon`, if given, re-ranks decoded readings (see
        `lexicon_decoder.decode_with_lexicon`) -- it can never reject or
        substitute a reading (phonies are legal, scoring plays), only
        prefer one pool-feasible candidate over another when the words
        it forms are real. `None` (the default) decodes by pool
        feasibility and confidence alone, same as before this existed."""
        if session is None and publish_gateway is None:
            raise ValueError(
                "GameWatcher needs either publish_gateway (standalone mode) or "
                "session (delegated mode) -- see the class docstring"
            )

        self.calibration = calibration
        # A single fixed reference photo can't track lighting that keeps
        # drifting over a long (30+ minute) broadcast -- found running
        # this against a complete real game: a screen region far in time
        # from the reference's capture moment developed a modest but
        # persistent false-positive occupancy diff, and no single fixed
        # reference point covers a whole game without the same problem
        # resurfacing somewhere else. When given a single reference image
        # (the common case), it's copied (never mutate a caller's array)
        # and its still-unplayed cells get continuously refreshed from
        # real frames as the game progresses -- see
        # `_refresh_reference_for_still_empty_cells`. Not supported (yet)
        # for the multi-reference case (a venue with a genuinely bimodal
        # empty appearance); adaptive refresh is simply disabled there.
        if isinstance(reference_board, np.ndarray):
            self.reference_board: ReferenceBoard = reference_board.copy()
            self._adaptive_reference = adaptive_reference
        else:
            self.reference_board = reference_board
            self._adaptive_reference = False
            if adaptive_occupancy_tracker is not None:
                raise ValueError(
                    "adaptive_occupancy_tracker needs a single reference_board image, "
                    "not a multi-reference list -- see the class docstring"
                )
        self._adaptive_tracker = adaptive_occupancy_tracker
        self._adaptive_occupied: Dict[Coord, bool] = {}
        self.classifier = classifier
        self.publish_gateway = publish_gateway
        self.session = session
        self.rack_detector = rack_detector
        self.motion_threshold = motion_threshold
        self.still_frame_count = still_frame_count
        self.occupancy_diff_threshold = occupancy_diff_threshold
        self.occupancy_gradient_threshold = occupancy_gradient_threshold
        self.lexicon = lexicon

        self._board = BoardState()
        self._racks: Dict[str, List[Tile]] = {}
        self._turn_number = 0
        self.state = WatcherState.IDLE_STILL
        self._frame_buffer = StillnessTracker(motion_threshold, still_frame_count)
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
        # Per-player rolling rack-frame buffer, gated through the same
        # stillness check the board path uses (see `record_rack`) --
        # keyed by player_id since two rack cameras settle independently.
        self._rack_frame_buffers: Dict[str, StillnessTracker] = {}

        # Never-jam commit search state (see the module docstring and the
        # FAILURE_QUARANTINE_THRESHOLD/etc. constants above).
        self._observation_index: int = 0
        self._cell_failure_count: Counter = Counter()
        self._quarantined: Dict[Coord, int] = {}  # coord -> observation_index its quarantine expires
        self._quarantined_since: Dict[Coord, int] = {}  # coord -> observation_index it was quarantined
        self._observations_since_commit: int = 0
        self._soft_cells: frozenset = frozenset()
        """Cells with partial (not unanimous) occupancy support across the
        most recently observed settled window (see
        `read_new_cells_voted`'s HARD/SOFT tiers) -- refreshed on every
        `observe_board_frame` call, never accumulated across calls. Used
        only as optional in-line extension material for an already-
        confirmed HARD run (`placement_search.enumerate_candidate_placements`'s
        `soft_cells` parameter); a soft cell never becomes part of
        `_confirmed_cells` in its own right, so it can never anchor or
        complete a placement on its own, only extend one."""

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

    def _refresh_reference_for_still_empty_cells(self, rectified_frame: np.ndarray) -> None:
        """Replaces the reference's pixels for every cell that's both
        unplayed in `board_before` AND not currently reading as occupied
        with this frame's -- keeps the empty-board reference tracking
        real lighting drift over a long broadcast instead of staying
        frozen at whatever it looked like at calibration time.

        The occupancy check here matters, not just `board_before`: a cell
        can be genuinely empty in `board_before` (not yet committed) while
        a new tile is ALREADY visibly sitting on it -- mid-placement, or
        confirmed-but-not-yet-applied, or even already applied via a
        session that updates a board this watcher doesn't own. Refreshing
        such a cell from the current frame would silently bake that new
        tile into the reference as "background," making it invisible to
        every future occupancy check. Already-played cells (per
        `board_before`) are left untouched regardless -- their reference
        pixels don't matter, since `board_before.is_empty` already gates
        them out of every future occupancy check. A no-op when adaptive
        refresh is disabled (multi-reference venues).

        Blended (exponential moving average), not a hard replace: a
        single noisy settled frame (a stray shadow, a compression
        artifact) would otherwise get instantly adopted as the new
        "ground truth" for that cell, which can make things worse rather
        than better. Blending a small fraction of each new observation in
        still tracks genuine gradual drift over the many settled
        observations of a real game, while resisting any single bad one.

        A quarantined cell (see `_quarantined_since`) is also eligible for
        refresh once it's been quarantined for at least
        `QUARANTINE_HEAL_DELAY` observations, even if it's currently
        reading as occupied -- this is what breaks the feedback trap
        where a persistently false-positive-occupied cell could otherwise
        never heal (the plain `not occupancy[coord]` gate alone would
        exclude it forever, since it never reads as empty). Healing uses
        a lower alpha (`QUARANTINE_HEAL_ALPHA`) than ordinary drift
        refresh: a mistaken heal is a real cost, so it should take longer
        to matter than genuine lighting drift does.

        Also feeds `self._adaptive_tracker` (if one was given), one
        `diff` score per not-yet-played cell -- this is the one full-
        board scan of the settled moment's last frame every observation
        already does, so hysteresis/adaptive thresholding piggybacks on
        it rather than requiring its own separate pass. A no-op for the
        adaptive tracker specifically whenever `self.board.is_empty(coord)`
        is False, same reasoning as the reference-refresh gate: a played
        cell is never queried again anyway.
        """
        if not self._adaptive_reference and self._adaptive_tracker is None:
            return
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                coord = (row, col)
                current_cell = crop_cell_inset(rectified_frame, row, col)
                reference_cell = crop_cell_inset(self.reference_board, row, col)
                scores = occupancy_scores(current_cell, reference_cell)
                occupied = (
                    scores["diff"] > self.occupancy_diff_threshold
                    or scores["gradient"] > self.occupancy_gradient_threshold
                )

                if self._adaptive_tracker is not None and self.board.is_empty(coord):
                    self._adaptive_occupied[coord] = self._adaptive_tracker.update(coord, scores["diff"])

                if not self._adaptive_reference:
                    continue
                quarantined_since = self._quarantined_since.get(coord)
                healing = (
                    quarantined_since is not None
                    and (self._observation_index - quarantined_since) >= QUARANTINE_HEAL_DELAY
                )
                if self.board.is_empty(coord) and (not occupied or healing):
                    alpha = QUARANTINE_HEAL_ALPHA if healing else _REFERENCE_EMA_ALPHA
                    x1, y1, x2, y2 = cell_bounds(row, col)
                    old = self.reference_board[y1:y2, x1:x2].astype(np.float32)
                    new = rectified_frame[y1:y2, x1:x2].astype(np.float32)
                    blended = (1.0 - alpha) * old + alpha * new
                    self.reference_board[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    def _expire_quarantine(self) -> None:
        expired = [cell for cell, expiry in self._quarantined.items() if self._observation_index >= expiry]
        for cell in expired:
            del self._quarantined[cell]
            del self._quarantined_since[cell]

    def observe_board_frame(self, frame: np.ndarray, player_id: str) -> WatcherEvent:
        """Feed one sampled board-camera frame in. `player_id` is whose
        turn is currently active -- the caller's responsibility to track
        (alternates deterministically in a 2-player game); vision alone
        can't determine this.
        """
        self._frame_buffer.push(frame)

        window = self._frame_buffer.stable_window()
        if window is None:
            moving = self._frame_buffer.last_pair_still is False
            self.state = WatcherState.HANDS_OVER_BOARD if moving else WatcherState.IDLE_STILL
            return WatcherEvent(state=self.state)

        self.state = WatcherState.BOARD_SETTLED
        self._observation_index += 1
        self._expire_quarantine()

        candidates = read_new_cells_voted(
            window, self.calibration, self.reference_board, self.classifier, self.board,
            top_k=len(self.classifier.classes),
            diff_threshold=self.occupancy_diff_threshold,
            gradient_threshold=self.occupancy_gradient_threshold,
        )
        # Refresh using this round's OLD reference for the decision above,
        # then update for next time -- the most recent frame in the
        # window is the closest thing to "current lighting" available.
        self._refresh_reference_for_still_empty_cells(self.calibration.rectify(window[-1]))
        if not candidates:
            # Settled, but nothing new since the last processed turn --
            # this is what makes repeated observation of the same stable
            # moment harmless rather than something callers must guard
            # against themselves.
            self._confirmed_cells = frozenset()
            self._last_candidate_cells = frozenset()
            return WatcherEvent(state=self.state)

        # SOFT cells (partial, not unanimous, occupancy support -- see
        # `read_new_cells_voted`) are deliberately excluded from
        # `current_cells`: they never enter ordinary per-cell confirmation
        # or clustering, only `self._soft_cells`, which
        # `enumerate_candidate_placements` may use to extend an
        # already-confirmed HARD run by one adjacent in-line cell. This
        # keeps every HARD-only code path below (confirmation, clustering,
        # quarantine) byte-for-byte the same as before support tiers
        # existed -- a SOFT cell can add to a placement, never anchor one.
        current_cells = frozenset(cc.coord for cc in candidates if not cc.is_soft)
        if self._adaptive_tracker is not None:
            # Conservative AND, never an OR: this can only REMOVE a cell
            # the vote-based HARD tier already found, when the per-cell
            # hysteresis/adaptive-threshold tracker disagrees -- it can
            # never invent a new candidate the vote-based system didn't
            # already surface. `.get(coord, True)` fails open (keeps the
            # cell) if the tracker somehow has no reading for it yet,
            # since the tracker is only ever a tightening layer here, not
            # the sole source of truth.
            current_cells = frozenset(coord for coord in current_cells if self._adaptive_occupied.get(coord, True))
        # Majority occupancy support (see read_new_cells_voted) alone
        # isn't enough to trust as extension material -- also require the
        # cell's own temporal-voted top label to clear a confidence floor
        # (SOFT_CELL_MIN_CONFIDENCE). A real full-game run showed why:
        # a soft-extended candidate is tried before the plain HARD one
        # (it's longer -- see the ranking comment below), so an
        # unconfident SOFT cell -- more likely occupancy noise than a
        # real tile -- could otherwise hijack the ranking on every single
        # observation, since "phonies are legal" means a nonsense
        # extension can't be rejected on spelling alone.
        self._soft_cells = frozenset(
            cc.coord for cc in candidates
            if cc.is_soft and cc.candidates and cc.candidates[0][1] >= SOFT_CELL_MIN_CONFIDENCE
        )
        newly_confirmed = current_cells & self._last_candidate_cells
        # Intersecting with current_cells means a cell that stops
        # appearing (a marginal one that turned out not to be real, or a
        # transient blip) drops back out on its own -- it never blocks a
        # stable subset from committing, and it isn't permanently baked in
        # either.
        self._confirmed_cells = (self._confirmed_cells | newly_confirmed) & current_cells
        self._last_candidate_cells = current_cells

        # Quarantined cells are excluded from consideration (not from
        # `_confirmed_cells` itself -- see `FAILURE_QUARANTINE_THRESHOLD`'s
        # docstring: quarantine only affects the search below, not the raw
        # perception-confirmation bookkeeping above).
        considerable_confirmed = self._confirmed_cells - frozenset(self._quarantined)

        # Cluster shape comes from the FULL current reading (confirmed or
        # not) -- a cluster is only "ready" once EVERY one of its cells is
        # confirmed. This is the same gate the original single-cluster
        # design used, and it matters for a reason beyond the original
        # motivation: enumerating candidates from the confirmed subset
        # ALONE (dropping this gate) can make a genuinely-still-growing
        # placement look like a clean, non-truncated one -- e.g. if a
        # cell overlaps a coincidentally-identical earlier partial
        # sighting, it can get "confirmed" one observation before its
        # neighbor, and a naive per-candidate readiness check would try
        # to commit it alone. Gating whole clusters, then enumerating
        # candidates only within a cluster that's ENTIRELY confirmed,
        # preserves the "don't act on a still-growing placement"
        # guarantee while still fixing the non-collinear-cluster problem
        # within an already-fully-confirmed cluster.
        #
        # Clustering here deliberately excludes quarantined cells first --
        # otherwise quarantine could never actually free up a good
        # neighbor: a quarantined cell would still glue itself to an
        # adjacent, perfectly fine cell via plain `current_cells`
        # clustering, permanently keeping that cluster un-ready even
        # though we've already given the quarantined cell a fair chance
        # and moved on. A cell that simply hasn't been confirmed YET is
        # different from one that's been tried and quarantined -- the
        # former still needs `current_cells` (not just the confirmed
        # subset) to detect "this cluster is still growing, wait"; the
        # latter has already had its chance and shouldn't keep blocking.
        clusterable_cells = current_cells - frozenset(self._quarantined)
        ready_clusters = [
            cluster for cluster in _cluster_cells(clusterable_cells, self.board)
            if cluster and cluster <= considerable_confirmed
        ]

        # A cluster that's only this small BECAUSE a quarantined neighbor
        # was cut out of it is not the same situation as a cluster that's
        # genuinely just this size -- there was real evidence a tile sits
        # right next to it. Found on the real WESPA broadcast: a tile the
        # classifier kept confidently misreading as BLANK got quarantined
        # after 3 failed observations, and the two neighboring cells it
        # had been gluing together then committed on their own -- cleanly,
        # with no truncation flag, since `cluster_max_size` only ever saw
        # the already-shrunk cluster -- producing a confidently wrong,
        # auto-published 2-cell word where the real move was 3 cells.
        #
        # `quarantined_now` (not `current_cells` alone) is deliberately
        # unioned in below: a quarantined cell's own occupancy signal
        # decays out of `current_cells` within a couple of observations
        # (the adaptive reference starts healing it toward "background"
        # almost immediately, see `QUARANTINE_HEAL_DELAY`) -- confirmed on
        # the real broadcast, where the quarantined cell had already
        # dropped out of `current_cells` entirely by the time its
        # neighbors' cluster next became ready, which silently defeated an
        # earlier version of this check that only looked at
        # `current_cells`. `self._quarantined` itself persists for the
        # cell's full `QUARANTINE_TTL_OBSERVATIONS` regardless of whether
        # the raw signal has since healed away, so re-including it here
        # keeps this check valid for that whole window, not just the
        # couple of observations before healing erases the live evidence.
        quarantined_now = frozenset(self._quarantined)
        unquarantined_clusters = _cluster_cells(current_cells | quarantined_now, self.board)
        clusters_touching_quarantine = frozenset(
            cluster for cluster in ready_clusters
            if any(cluster < full and full & quarantined_now for full in unquarantined_clusters)
        )

        self.state = WatcherState.DIFF_COMPUTED
        cell_candidates_by_coord = {cc.coord: cc for cc in candidates}
        failed_cells_this_observation: set = set()
        first_failure_reason: Optional[str] = None
        first_failure_cells: Tuple[Coord, ...] = ()
        attempted_anything = False

        for cluster in ready_clusters:
            # Rank descending by cell count first, still with no lexicon
            # tiebreak, even though WS2's real lexicon now exists -- cell
            # count must stay primary because the true placement is always
            # a superset of every truncation of itself, phonies included:
            # ranking a valid-but-shorter reading ahead of a longer phony
            # would silently prefer the wrong one (see placement_search.py
            # and the module docstring's phonies-are-legal constraint).
            # This is also what makes a soft-extended candidate (always
            # one cell longer than its un-extended HARD version -- see
            # `_soft_cells` above) get tried FIRST: if the extra cell is a
            # genuine tile, decoding and validation succeed and it commits
            # (always `needs_operator=True`, since `used_soft_cells` forces
            # that below); if the extension is spurious, one of those
            # checks is the only thing standing between it and wrongly
            # winning over the correct, un-extended reading -- the same
            # residual risk WS1 already accepts for truncations, bounded
            # the same way (never auto-published).
            ranked = sorted(
                enumerate_candidate_placements(cluster, self.board, self._soft_cells),
                key=lambda c: (-len(c.cells), sorted(c.cells)),
            )
            shrunk_by_quarantine = cluster in clusters_touching_quarantine
            for candidate in ranked:
                attempted_anything = True
                cell_candidates = [cell_candidates_by_coord[coord] for coord in sorted(candidate.cells)]
                outcome = self._attempt_commit(
                    candidate, cell_candidates, player_id, source_frame=window[-1],
                    shrunk_by_quarantine=shrunk_by_quarantine,
                )
                if isinstance(outcome, WatcherEvent):
                    self._observations_since_commit = 0
                    return outcome

                reason, attempted_cells = outcome
                if first_failure_reason is None:
                    first_failure_reason, first_failure_cells = reason, attempted_cells
                # Attribute the failure to whatever cells the outcome
                # itself named -- usually the whole candidate, but a
                # blank failure names only the actual blank cell(s) (see
                # `_attempt_commit`), so a multi-cell candidate with one
                # bad cell doesn't quarantine its good cells right along
                # with it.
                for cell in attempted_cells:
                    if cell in failed_cells_this_observation:
                        continue
                    failed_cells_this_observation.add(cell)
                    self._cell_failure_count[cell] += 1
                    if self._cell_failure_count[cell] >= FAILURE_QUARANTINE_THRESHOLD:
                        self._quarantined[cell] = self._observation_index + QUARANTINE_TTL_OBSERVATIONS
                        self._quarantined_since[cell] = self._observation_index

        self.state = WatcherState.DIFF_COMPUTED

        if not attempted_anything:
            # No cluster has (yet) had every one of its cells survive a
            # second independent look -- could be a completed turn, or a
            # player mid-placement who just happened to pause, or one
            # cell whose visibility is still marginal. Don't act yet, and
            # don't count this toward the stall watchdog -- nothing was
            # actually attempted and failed, there's just nothing ready.
            return WatcherEvent(state=self.state)

        self._observations_since_commit += 1
        if self._observations_since_commit >= STALL_OBSERVATIONS:
            # A backstop beyond per-cell quarantine: something is still
            # stuck even with known-bad cells already excluded. Force-
            # quarantine every confirmed cell with any failure history so
            # the next observation gets a genuinely fresh look.
            for cell in self._confirmed_cells:
                if self._cell_failure_count.get(cell, 0) > 0 and cell not in self._quarantined:
                    self._quarantined[cell] = self._observation_index + QUARANTINE_TTL_OBSERVATIONS
                    self._quarantined_since[cell] = self._observation_index
            self._observations_since_commit = 0
            self.state = WatcherState.STALLED
            return WatcherEvent(
                state=self.state, needs_operator=True, is_stall=True,
                reason=f"no commit in {STALL_OBSERVATIONS} observations; last failure: {first_failure_reason}",
                attempted_cells=first_failure_cells,
            )

        return WatcherEvent(
            state=self.state, needs_operator=True, reason=first_failure_reason,
            attempted_cells=first_failure_cells,
        )

    def _attempt_commit(
        self, candidate: CandidatePlacement, cell_candidates: List[CellCandidates], player_id: str,
        source_frame: Optional[np.ndarray] = None, shrunk_by_quarantine: bool = False,
    ) -> Union[WatcherEvent, Tuple[str, Tuple[Coord, ...]]]:
        """Tries to commit exactly one candidate placement. Returns a
        `WatcherEvent` if this candidate reached a real outcome (an
        auto-publish, or a genuine operator-pending move) or a `(reason,
        attempted_cells)` tuple if it failed for a reason the caller's
        search loop should record and move past, trying the next
        candidate instead of giving up on the whole observation.
        """
        attempted_cells = tuple(sorted(candidate.cells))
        reading = decode_with_lexicon(cell_candidates, self.board, list(self.racks.values()), lexicon=self.lexicon)[0]
        decoded = reading.labels
        conf_by_coord = {cc.coord: dict(cc.candidates) for cc in cell_candidates}
        # A blank-resolved cell's chosen LETTER (e.g. "T") is usually not
        # itself one of the classifier's own candidate labels for that
        # cell -- the image carries no evidence for which letter a blank
        # is, only that it's a blank at all. Fall back to the classifier's
        # own BLANK confidence in that case; it's what genuinely reflects
        # this cell's uncertainty.
        #
        # Aggregated across cells by GEOMETRIC MEAN, not MIN. Measured
        # against 29 real commits from the WESPA broadcast
        # (confidence_aggregation_experiment.py): a best-threshold split
        # separates correct from wrong commits at 75.9% accuracy under
        # the geometric mean vs. 65.5% under MIN. MIN is a per-length
        # penalty by construction -- a 7-letter word with six confident
        # cells and one shaky one scores no better than a 1-letter play
        # of that same shaky cell alone -- which structurally punishes
        # longer, often-correct words the hardest.
        per_cell_confs = [
            max(conf_by_coord[coord].get(label, conf_by_coord[coord].get(CLASSIFIER_BLANK_LABEL, 0.0)), 1e-6)
            for coord, label in decoded.items()
        ]
        move_confidence = math.exp(sum(math.log(c) for c in per_cell_confs) / len(per_cell_confs))

        blank_coords = tuple(sorted(reading.blank_cells))
        if blank_coords:
            # Attribute this failure only to the actual blank cell(s), not
            # every cell in the candidate -- a multi-cell word with one
            # genuine blank in it (e.g. the real WESPA move "PInOTa.E",
            # 7 cells/2 blanks) would otherwise quarantine its five
            # perfectly good letters right alongside the two blanks,
            # since they all "failed" together as part of the same
            # candidate. Once only the blanks are quarantined, the
            # remaining good cells can be re-tried (as their own, smaller
            # candidates) instead of the whole turn going silent.
            #
            # `decode_with_lexicon` (unlike the old `decode_feasible_reading`)
            # already resolves a blank to a specific, pool-feasible, and
            # (when a lexicon is given) word-consistent letter -- but
            # this project's structural rule stands regardless: a blank's
            # played letter is never trusted enough to auto-publish
            # without an operator confirming it, no matter how the
            # letter was arrived at. Using the resolved letter to pre-fill
            # an operator's review UI is a real future improvement, not
            # done here.
            return (
                "a detected tile is a blank -- letter unknown until an operator confirms what it's played as",
                blank_coords,
            )

        placements = {coord: Tile(letter=label, is_blank=False) for coord, label in decoded.items()}
        try:
            board_after = self.board.with_placements(placements)
        except ValueError as exc:
            return (str(exc), attempted_cells)

        new_cells = list(placements.keys())
        validation = validate_placement(self.board, board_after, new_cells)
        if not validation.ok:
            return (validation.reason, attempted_cells)

        try:
            compute_pool_state(board_after, list(self.racks.values()))
        except PoolInvariantViolation as exc:
            return (f"pool invariant violated: {exc}", attempted_cells)

        # Every check passed -- this reading is genuinely being acted on,
        # so the next observation starts a fresh confirmation cycle.
        self._confirmed_cells = frozenset()
        self._last_candidate_cells = frozenset()

        # A candidate smaller than the biggest one its cluster could have
        # produced (a truncation), or one extended with a soft-support
        # cell, never auto-publishes regardless of confidence -- both are
        # real, measured risks (see placement_search.py's docstring): a
        # truncated word can still pass every geometric/pool check, and so
        # can a contaminated superset. An operator is the correct backstop
        # for the residual neither geometry nor confidence can rule out.
        is_truncated = len(candidate.cells) < candidate.cluster_max_size
        force_operator_reasons: List[str] = []
        if is_truncated or candidate.used_soft_cells:
            force_operator_reasons.append("truncated or soft-supported placement always needs operator review")

        # `cluster_max_size` only ever reflects the cluster AFTER quarantine
        # has already cut cells out of it, so a genuinely-truncated
        # placement can look clean (`is_truncated=False`) once its missing
        # neighbor has been quarantined away -- found on the real WESPA
        # broadcast: a tile the classifier kept confidently misreading as
        # BLANK got quarantined after 3 failures, and the two cells it had
        # been gluing together then committed alone, with no truncation
        # flag, silently dropping a real tile from the word. `shrunk_by_quarantine`
        # (computed by the caller against the FULL current reading, before
        # quarantine exclusion) catches exactly this case.
        if shrunk_by_quarantine:
            force_operator_reasons.append(
                "an adjacent tile was excluded from this placement after repeated failed reads "
                "-- this word may be incomplete"
            )

        # A confident non-blank letter whose tile face is nonetheless
        # visually smooth is a plausible missed blank -- the classifier's
        # own BLANK class is not the only signal for this (see
        # blank_heuristic.py's docstring: measured 2/2 held-out real blank
        # tiles the deployed checkpoint misread as a confident letter).
        # This never overrides the decoded letter or calls the cell blank
        # itself; it only routes an otherwise-confident reading to an
        # operator the same way a genuine blank already does.
        if source_frame is not None:
            rectified = self.calibration.rectify(source_frame)
            smooth_cells = [
                coord for coord in new_cells
                if looks_smooth_like_a_blank(crop_cell(rectified, coord[0], coord[1]))
            ]
            if smooth_cells:
                squares = ", ".join(sorted(format_square(c) for c in smooth_cells))
                force_operator_reasons.append(
                    f"tile at {squares} reads as a confident letter but looks unusually smooth "
                    "-- possible missed blank"
                )

        force_operator = bool(force_operator_reasons)
        force_operator_reason = "; ".join(force_operator_reasons)
        # `publish_confidence` only ever matters to
        # AUTONOMOUS_WITH_CONFIDENCE_FALLBACK -- plain AUTONOMOUS ignores
        # confidence entirely by design (`PublishGateway.should_auto_publish`
        # returns True unconditionally), so forcing operator review for a
        # truncated/soft-extended/possible-missed-blank candidate cannot
        # rely on confidence alone in standalone mode; it bypasses the
        # gateway outright, below, the same way the blank check above
        # already does.
        publish_confidence = 0.0 if force_operator else move_confidence

        if self.session is not None:
            return self._submit_play_via_session(
                decoded, player_id, move_confidence, publish_confidence,
                force_operator, force_operator_reason, source_frame,
            )

        words = words_formed(board_after, new_cells)
        move_score = score_move(board_after, words, new_cells)
        provisional_turn = self._turn_number + 1
        move_candidate = MoveCandidate(
            turn_number=provisional_turn, player_id=player_id, move_type=MoveType.PLAY,
            new_cells=tuple(new_cells),
        )
        scored_move = ScoredMove(candidate=move_candidate, move_score=move_score)
        self.state = WatcherState.SCORED

        if force_operator or not self.publish_gateway.should_auto_publish(publish_confidence):
            pending = PendingMove(
                scored_move=scored_move, board_after=board_after, racks_after={}, confidence=move_confidence,
                source_frame_jpeg=_encode_frame_jpeg(source_frame),
            )
            reason = force_operator_reason if force_operator else "confidence below gateway threshold"
            return WatcherEvent(
                state=self.state, scored_move=scored_move, confidence=move_confidence,
                needs_operator=True, pending=pending, reason=reason,
            )

        self._board = board_after
        self._turn_number = provisional_turn
        self.state = WatcherState.APPLIED
        return WatcherEvent(state=self.state, scored_move=scored_move, confidence=move_confidence)

    def _submit_play_via_session(
        self,
        decoded: Dict[Coord, str],
        player_id: str,
        move_confidence: float,
        publish_confidence: Optional[float] = None,
        force_operator: bool = False,
        force_operator_reason: str = "",
        source_frame: Optional[np.ndarray] = None,
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

        `publish_confidence`, if lower than `move_confidence`, forces the
        session's own gateway to deny auto-publish (a truncated or
        soft-supported candidate) while `move_confidence` still reports the
        real value for diagnostics.
        """
        new_tiles = [(coord, label, False) for coord, label in decoded.items()]
        result = self.session.submit_move(
            player_id, new_tiles=new_tiles,
            confidence=publish_confidence if publish_confidence is not None else move_confidence,
            source_frame_jpeg=_encode_frame_jpeg(source_frame),
        )

        if isinstance(result.outcome, MoveProcessingError):
            return WatcherEvent(
                state=self.state, confidence=move_confidence, needs_operator=True, reason=result.outcome.reason,
            )

        self.state = WatcherState.APPLIED if result.published else WatcherState.SCORED
        turn_number = result.outcome.candidate.turn_number
        pending = None if result.published else self.session.pending.get(turn_number)
        if result.published:
            reason = None
        elif force_operator:
            reason = force_operator_reason
        else:
            reason = "confidence below gateway threshold"
        return WatcherEvent(
            state=self.state, scored_move=result.outcome, confidence=move_confidence,
            needs_operator=not result.published, pending=pending, reason=reason,
        )

    def record_rack(
        self, player_id: str, rack_frame: np.ndarray, detection_threshold: float = 0.3,
    ) -> Optional[WatcherEvent]:
        """Feed one sampled rack-camera frame in for `player_id` (see the
        module docstring's scope notes). Like the board path, a frame
        isn't read the instant it arrives -- it's buffered per player and
        gated through the same stillness check (`stable_window`) the
        board path uses, so a rack caught mid-rearrangement (a hand still
        moving tiles) doesn't get read as if it were settled. Returns None
        while still waiting for a still window, exactly as it does for "no
        change" once reading does happen -- callers already have to
        tolerate a None result meaning "nothing to report from this call,"
        so this doesn't change that contract, only how often a real read
        gets attempted.

        Once settled, the *first* observation for a given `player_id` just
        establishes their starting rack silently (returns None) -- every
        player starts with a full rack, so that's not an event, it's
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

        buffer = self._rack_frame_buffers.setdefault(
            player_id, StillnessTracker(self.motion_threshold, self.still_frame_count)
        )
        buffer.push(rack_frame)

        window = buffer.stable_window()
        if window is None:
            return None
        # All frames in a stable window read the same by construction (the
        # motion gate requires negligible change between every consecutive
        # pair) -- the most recent one is as good as any, and matches the
        # board path's convention of using window[-1] where a single frame
        # is needed.
        settled_frame = window[-1]

        is_first_observation = player_id not in self.racks
        observations = read_rack(settled_frame, self.rack_detector, self.classifier, detection_threshold)
        new_rack = rack_observations_to_tiles(observations)
        previous_rack = self.racks.get(player_id, [])

        if not is_first_observation and _rack_multiset(new_rack) == _rack_multiset(previous_rack):
            return None

        # A rack detector/classifier misread (most often a real letter
        # hallucinated as a BLANK, found running exactly this check
        # against a real game for the first time) can claim more of some
        # letter exists across board+racks than the standard 100-tile set
        # actually has -- if committed silently, this doesn't just make
        # THIS rack wrong, it corrupts `remaining_supply` for every future
        # decode using ANY player's rack, including completely unrelated
        # board turns that have nothing to do with this misread. Reject
        # (never commit, route to operator) any candidate rack that would
        # create an impossible tile-supply state, the same "never silently
        # corrupt downstream state" philosophy `compute_pool_state` itself
        # already documents for board+rack decoding.
        other_racks = [rack for pid, rack in self.racks.items() if pid != player_id]
        try:
            compute_pool_state(self.board, other_racks + [new_rack])
        except PoolInvariantViolation:
            return WatcherEvent(
                state=self.state, confidence=0.0, needs_operator=True,
                reason="rack read would violate the tile-supply pool invariant -- likely a misread "
                       "(e.g. a real letter read as a blank)",
            )

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
