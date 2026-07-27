"""The rack-detector end-to-end gap `models/README.md` has flagged since
the detector was trained: `read_rack`/`record_rack` exist and work
standalone (see `training/detect/README.md`'s 14/14 real-photo result),
but nothing had ever driven them alongside the real board pipeline for an
actual game, so `GameWatcher.racks` had never been anything but the
`racks=[]` placeholder `decode_with_lexicon`'s pool-feasibility check
falls back to.

Builds a handful of real turns from the real WESPA Game 1 broadcast
(`tests/fixtures/wespa_word_wars_game1.gcg`) as synthetic images, the same
way `test_synthetic_full_game.py` does for the board -- but ALSO renders
each acting player's real rack-before-move (from the `.gcg`'s own `rack`
field, ground truth no different from what `gcg_truth.py` already trusts
for board cells) via `training.synth_render.rack_scene_renderer`, and
feeds both into one real `GameWatcher` using the REAL rack detector
(`models/rack_detector_v1.pth`) and REAL tile classifier
(`models/tile_classifier_v1.pt`) -- not a synthetic oracle -- so this is
an honest measurement of the actual, currently-deployed detector+
classifier pair working together end to end, not a simulation of them.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pytest

from autoscorer.eval.gcg_truth import load_truth_turns
from autoscorer.gamelogic.movedetect.game_watcher import GameWatcher
from autoscorer.gamelogic.publish import PublishGateway, PublishMode
from autoscorer.perception.calibration.homography import BoardCalibration
from tests.support.synth_board import blank_board_image, place_tile
from training.classify.infer import TileClassifierModel
from training.collect.replay_game import read_gcg_moves
from training.detect.visualize_rack_detections import load_model
from training.synth_render.rack_scene_renderer import compress_like_a_real_photo, generate_rack_scene

pytestmark = pytest.mark.slow

GCG_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "wespa_word_wars_game1.gcg"
CLASSIFIER_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "tile_classifier_v1.pt"
RACK_DETECTOR_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "rack_detector_v1.pth"
STILL_FRAME_COUNT = 3
IDENTITY_CALIBRATION = BoardCalibration(homography=np.eye(3, dtype=np.float64))
NUM_TURNS_TO_DRIVE = 4  # keeps real RF-DETR inference cost bounded (~1.3s/rack call)


def _rack_letters_from_gcg_string(rack: str) -> List:
    # A GCG rack string uses '?' for an unplayed blank tile -- represented
    # here the same way `generate_rack_scene`'s `letters` already expects
    # (None means "render a blank"), not the letter it later gets played
    # as, which the rack itself never reveals.
    return [None if ch == "?" else ch for ch in rack]


def _settle_board(watcher, board_image, player_id, max_observations=15):
    # Checks the board's own occupied-cell count rather than
    # `watcher.turn_number` -- `record_rack` correctly flags any rack
    # change (including an entirely normal post-play redraw, which this
    # test deliberately renders every turn) as an EXCHANGE event, since
    # nothing here can distinguish that from a genuine exchange without a
    # turn-clock signal (the same documented limitation as PASS
    # detection) -- so `turn_number` alone conflates real board plays
    # with rack-driven pseudo-turns and can't be used as a per-board-turn
    # counter in this test.
    cells_before = len(list(watcher.board.occupied_cells()))
    frame = np.zeros(board_image.shape, dtype=np.uint8)
    frame[:400] = np.random.RandomState(0).randint(0, 255, (400, board_image.shape[1], 3), dtype=np.uint8)
    watcher.observe_board_frame(frame, player_id=player_id)  # motion, breaks the previous settled window
    for _ in range(max_observations):
        watcher.observe_board_frame(board_image.copy(), player_id=player_id)
        if len(list(watcher.board.occupied_cells())) > cells_before:
            return True
    return False


def _settle_rack(watcher, rack_image, player_id, count=3):
    event = None
    for _ in range(count):
        event = watcher.record_rack(player_id, rack_image.copy())
    return event


def test_rack_detector_and_classifier_work_together_end_to_end_on_a_real_game():
    truth_turns = load_truth_turns(GCG_PATH)
    gcg_moves = read_gcg_moves(GCG_PATH)
    assert len(truth_turns) == len(gcg_moves)

    classifier = TileClassifierModel(CLASSIFIER_PATH, device="cpu")
    rack_detector = load_model(RACK_DETECTOR_PATH)

    gateway = PublishGateway(mode=PublishMode.AUTONOMOUS)
    watcher = GameWatcher(
        calibration=IDENTITY_CALIBRATION,
        reference_board=blank_board_image(),
        classifier=classifier,
        publish_gateway=gateway,
        rack_detector=rack_detector,
        still_frame_count=STILL_FRAME_COUNT,
    )

    render_rng = random.Random(42)
    board = blank_board_image()
    tiles_correct = 0
    tiles_total = 0
    rejections = 0

    for turn, move in zip(truth_turns[:NUM_TURNS_TO_DRIVE], gcg_moves[:NUM_TURNS_TO_DRIVE]):
        true_rack_letters = _rack_letters_from_gcg_string(move.rack)

        # Establish/refresh the acting player's rack from their REAL
        # rack-before-move, through the REAL detector, before the board
        # itself changes -- exactly the order a live broadcast would
        # produce (a player's rack is visible before their tiles land).
        rack_scene = generate_rack_scene(rng=render_rng, letters=true_rack_letters)
        rack_image = np.array(compress_like_a_real_photo(rack_scene.image, render_rng))[:, :, ::-1]  # RGB->BGR
        rack_before = list(watcher.racks.get(turn.player, []))
        event = _settle_rack(watcher, rack_image, turn.player)
        rack_after = list(watcher.racks.get(turn.player, []))

        if event is not None and event.reason and "pool invariant" in event.reason:
            # The detector misread something badly enough this turn's
            # rack read would create an impossible tile-supply state --
            # this MUST be rejected (see GameWatcher.record_rack), not
            # counted as either a correct or incorrect read of THIS
            # turn's rack. The one property this test actually needs
            # from a rejection: it must never corrupt what was already
            # tracked for this player.
            rejections += 1
            assert rack_after == rack_before, "a rejected rack read must never overwrite the last known-good rack"
        else:
            # An honest per-TILE accuracy (not "did the whole 7-tile rack
            # match exactly"), since a single misread letter among 7 is a
            # very different, much better outcome than a fully wrong
            # rack, and this test's job is to measure what actually
            # happens, not assert a number picked in advance. The
            # detector's own single-photo validation (training/detect/
            # README.md) was 14/14 -- these are synthetic renders it
            # wasn't specifically validated against, so some real
            # per-tile error here is expected, not a regression on its
            # own.
            detected_multiset = sorted(("BLANK" if t.is_blank else t.letter) for t in rack_after)
            expected_multiset = sorted("BLANK" if letter is None else letter for letter in true_rack_letters)
            # Multiset overlap: how many of the expected tiles were
            # actually present in what got detected, letter-for-letter.
            overlap = sum((Counter(detected_multiset) & Counter(expected_multiset)).values())
            tiles_correct += overlap
            tiles_total += len(expected_multiset)

        # Now place this turn's real tiles on the board, same as the
        # board-only synthetic test does.
        for coord in sorted(turn.cells):
            row, col = coord
            is_blank = coord in turn.blank_cells
            render_letter = None if is_blank else turn.letters[coord]
            place_tile(board, row, col, render_letter, render_rng)

        committed = _settle_board(watcher, board, turn.player)
        assert committed, (
            f"turn {turn.turn_number} never committed with a real rack detector wired in -- a rack "
            f"read (rejected or not) must never block an unrelated board turn from committing"
        )

    total_cells_placed = sum(len(turn.cells) for turn in truth_turns[:NUM_TURNS_TO_DRIVE])
    assert len(list(watcher.board.occupied_cells())) == total_cells_placed, (
        f"expected {total_cells_placed} occupied cells after {NUM_TURNS_TO_DRIVE} real board turns "
        f"while a real rack detector was active alongside them, got {len(list(watcher.board.occupied_cells()))}"
    )
    print(f"\nrack reads: {rejections} rejected (pool-invariant guard fired), "
          f"{tiles_correct}/{tiles_total} tiles correct on the {tiles_total and tiles_correct/tiles_total:.0%} "
          f"accepted turns")
    assert tiles_total > 0, "every rack read was rejected -- the pool-invariant guard itself may be miscalibrated"
    assert tiles_correct / tiles_total >= 0.75, (
        f"only {tiles_correct}/{tiles_total} rack tiles read correctly on accepted turns -- "
        f"a real regression, not just the occasional synthetic-render misread"
    )
    # The actual point of this test: GameWatcher.racks is no longer the
    # racks=[] placeholder -- real, non-empty racks are now tracked and
    # flow into decode_with_lexicon's pool-feasibility check, and a bad
    # rack read (which real detection will occasionally produce) can no
    # longer corrupt tracked state or block an unrelated board commit.
    assert any(len(rack) > 0 for rack in watcher.racks.values())
