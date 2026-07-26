"""The highest-value new test WS0 adds: reproduces GameWatcher's real
full-game jam behavior in seconds, with no real broadcast video and no
GPU -- so the never-jam properties WS1 exists to deliver can be verified
deterministically instead of by re-running a 20+ minute real video every
time.

Builds the entire real WESPA Game 1 (21 real plays, from the committed
`wespa_word_wars_game1.gcg` fixture) as synthetic board images, using the
same rendering path `tests/unit/test_game_watcher.py` already relies on
(`tests/support/synth_board.py`). Drives `GameWatcher` directly with
in-memory frame arrays rather than an encoded video file -- an actual
video buys nothing here that feeding frame arrays doesn't already give,
and it's simpler and faster; `tests/unit/test_run_watcher.py` already
covers the video-decoding path itself.

The classifier is `NoisyOracleClassifier` (`tests/support/noisy_oracle.py`):
it nearest-neighbor identifies each crop against a registry of everything
actually rendered for this game (robust to the tiny numeric noise
real-ish augmentation introduces), then injects controlled per-cell error
at a chosen accuracy `p`. This is what lets the test simulate the
deployed classifier's real accuracy ceiling without needing real photos.

- At `p=1.0` (perfect classification), any remaining permanent jam is a
  real `GameWatcher` logic bug, independent of classifier accuracy --
  Game 1's own move 19 (`PIOTE?? 14B PInOTa.E`, two blanks) is exactly
  this case, since the blank jam fires regardless of confidence.
- At `p=0.72` (the deployed checkpoint's real measured board-tile
  accuracy), the pipeline is not expected to get every turn right, but it
  must not get permanently STUCK -- this is the never-jam invariant.

Neither assertion here depends on `WatcherEvent.is_stall` (that field
exists, but nothing sets it yet until WS1's watchdog lands) -- stalls are
detected directly, by capping how many settled observations any single
real play is allowed before the test moves on to the next one anyway
(exactly what a real video does: it keeps playing regardless of whether
GameWatcher ever resolves the current board state).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from autoscorer.eval.gcg_truth import TruthTurn, load_truth_turns
from autoscorer.gamelogic.movedetect.game_watcher import GameWatcher
from autoscorer.gamelogic.publish import PublishGateway, PublishMode
from autoscorer.perception.calibration.homography import BoardCalibration
from tests.support.noisy_oracle import NoisyOracleClassifier, build_confusion_distribution
from tests.support.synth_board import blank_board_image, place_tile

pytestmark = pytest.mark.slow

GCG_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "wespa_word_wars_game1.gcg"
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "tile_classifier_v1.pt"
STILL_FRAME_COUNT = 3
MAX_OBSERVATIONS_PER_TURN = 15
IDENTITY_CALIBRATION = BoardCalibration(homography=np.eye(3, dtype=np.float64))


def _disrupt(watcher: GameWatcher, player_id: str, board_shape) -> None:
    """A hand-sized occlusion band -- a single tile-sized change is too
    small to reliably move the whole-frame motion signal (see
    `test_game_watcher.py`'s own `_disrupt`, which established this)."""
    frame = np.zeros(board_shape, dtype=np.uint8)
    frame[:400] = np.random.RandomState(0).randint(0, 255, (400, board_shape[1], 3), dtype=np.uint8)
    watcher.observe_board_frame(frame, player_id=player_id)


def _build_registry_and_boards(
    truth_turns: List[TruthTurn], rng: random.Random,
) -> Tuple[List[Tuple[np.ndarray, str]], List[Tuple[str, np.ndarray]]]:
    """Returns (registry, boards). `registry` is every distinct rendered
    crop as (crop_array, true_label) -- true_label is "BLANK" for a blank
    cell (a blank shows no glyph, so that's the only honest identity a
    classifier could ever assign it; constraint/lexicon decoding resolves
    which letter it was played as separately, same as the real pipeline).
    `boards` is the cumulative board image after each real play, in
    order, paired with which player made that play."""
    board = blank_board_image()
    registry: List[Tuple[np.ndarray, str]] = []
    boards: List[Tuple[str, np.ndarray]] = []

    for turn in truth_turns:
        for coord in sorted(turn.cells):
            row, col = coord
            is_blank = coord in turn.blank_cells
            render_letter = None if is_blank else turn.letters[coord]
            true_label = "BLANK" if is_blank else turn.letters[coord]
            crop = place_tile(board, row, col, render_letter, rng)
            registry.append((crop, true_label))
        boards.append((turn.player, board.copy()))

    return registry, boards


def _run_game(p: float, seed: int) -> Tuple[GameWatcher, List[int]]:
    truth_turns = load_truth_turns(GCG_PATH)
    build_rng = random.Random(42)
    registry, boards = _build_registry_and_boards(truth_turns, build_rng)

    confusion_dist = build_confusion_distribution(CHECKPOINT_PATH, samples_per_class=15, seed=0)
    oracle = NoisyOracleClassifier(registry, confusion_dist, p=p, rng=random.Random(seed))

    gateway = PublishGateway(mode=PublishMode.AUTONOMOUS)
    watcher = GameWatcher(
        calibration=IDENTITY_CALIBRATION,
        reference_board=blank_board_image(),
        classifier=oracle,
        publish_gateway=gateway,
        still_frame_count=STILL_FRAME_COUNT,
    )

    observations_per_turn: List[int] = []
    for player, board_image in boards:
        turn_before = watcher.turn_number
        _disrupt(watcher, player, board_image.shape)
        used = 0
        for _ in range(MAX_OBSERVATIONS_PER_TURN):
            watcher.observe_board_frame(board_image.copy(), player_id=player)
            used += 1
            if watcher.turn_number > turn_before:
                break
        observations_per_turn.append(used)

    return watcher, observations_per_turn


def test_perfect_classification_completes_the_whole_real_game_with_no_jam():
    watcher, observations_per_turn = _run_game(p=1.0, seed=1)
    truth_turns = load_truth_turns(GCG_PATH)

    assert watcher.turn_number == len(truth_turns), (
        f"only {watcher.turn_number}/{len(truth_turns)} real plays committed at perfect "
        f"classification -- any shortfall here is a GameWatcher logic bug, not an accuracy "
        f"ceiling (per-turn observation counts: {observations_per_turn})"
    )
    assert max(observations_per_turn) < MAX_OBSERVATIONS_PER_TURN, (
        "at least one real play never committed within the observation budget -- a jam"
    )


def test_realistic_noise_never_permanently_jams():
    _, observations_per_turn = _run_game(p=0.72, seed=2)

    stuck_turns = sum(1 for n in observations_per_turn if n >= MAX_OBSERVATIONS_PER_TURN)
    assert stuck_turns == 0, (
        f"{stuck_turns} real play(s) never committed within the observation budget at "
        f"p=0.72 -- a permanent jam, not just a missed/wrong turn "
        f"(per-turn observation counts: {observations_per_turn})"
    )
