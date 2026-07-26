import random

import numpy as np
import pytest
import torch

import autoscorer.gamelogic.movedetect.game_watcher as gw_module
from autoscorer.api.session import GameSession
from autoscorer.gamelogic.board import CENTER, BoardState, Tile
from autoscorer.gamelogic.models import MoveType
from autoscorer.gamelogic.movedetect.game_watcher import GameWatcher, WatcherState
from autoscorer.gamelogic.publish import PublishGateway, PublishMode
from autoscorer.perception.calibration.homography import BoardCalibration, cell_bounds
from training.classify.infer import TileClassifierModel
from training.classify.train import run_training, save_checkpoint
from training.synth_render.tile_renderer import FINAL_SIZE, augment_tile, render_tile

from tests.support.synth_board import blank_board_image as _blank_board_image
from tests.support.synth_board import place_tile as _place_tile_full

IDENTITY_CALIBRATION = BoardCalibration(homography=np.eye(3, dtype=np.float64))


def _place_tile(board_img: np.ndarray, row: int, col: int, letter, rng: random.Random) -> None:
    _place_tile_full(board_img, row, col, letter, rng)


def _train_tiny_classifier(tmp_path, labels=("A", "N", "T", "BLANK")):
    rng = random.Random(0)
    for label in labels:
        letter = None if label == "BLANK" else label
        class_dir = tmp_path / label
        class_dir.mkdir(parents=True, exist_ok=True)
        base = render_tile(letter, rng=rng)
        for i in range(30):
            augment_tile(base, rng=rng).save(class_dir / f"{i:03d}.png")

    model, result = run_training(tmp_path, epochs=25, batch_size=8, device=torch.device("cpu"), seed=1)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(model, result.classes, checkpoint_path)
    return TileClassifierModel(checkpoint_path, device="cpu")


def _make_watcher(classifier, mode=PublishMode.AUTONOMOUS, confidence_threshold=0.9, rack_detector=None):
    gateway = PublishGateway(mode=mode, confidence_threshold=confidence_threshold)
    return GameWatcher(
        calibration=IDENTITY_CALIBRATION,
        reference_board=_blank_board_image(),
        classifier=classifier,
        publish_gateway=gateway,
        rack_detector=rack_detector,
        still_frame_count=3,
    )


def test_stays_idle_while_frames_keep_moving(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier)
    rng = random.Random(1)

    # A large occlusion covering most of the board (standing in for a
    # hand reaching across it), shifted every frame -- unlike a single
    # tile-sized change, this moves frame_motion_score's whole-image mean
    # well past the threshold, so the board never reads as settled.
    for i in range(4):
        frame = _blank_board_image()
        band_start = i * 100
        frame[band_start:band_start + 400, :] = rng.randint(0, 255)
        event = watcher.observe_board_frame(frame, player_id="p1")

    assert event.state in (WatcherState.IDLE_STILL, WatcherState.HANDS_OVER_BOARD)
    assert event.scored_move is None
    assert watcher.board.is_blank_board()


def test_settled_with_no_change_produces_no_move(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier)
    reference = _blank_board_image()

    events = [watcher.observe_board_frame(reference.copy(), player_id="p1") for _ in range(4)]

    assert events[-1].state == WatcherState.BOARD_SETTLED
    assert all(e.scored_move is None for e in events)
    assert watcher.board.is_blank_board()


def test_full_sequence_places_a_tile_and_auto_publishes(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)
    rng = random.Random(7)

    # First settle on the empty board (establishes board_before).
    empty = _blank_board_image()
    for _ in range(3):
        watcher.observe_board_frame(empty.copy(), player_id="p1")

    # A hand enters (motion) then a new tile is placed at the center --
    # a legal first move.
    watcher.observe_board_frame(_blank_board_image(), player_id="p1")  # arbitrary motion frame stand-in
    placed = _blank_board_image()
    _place_tile(placed, *CENTER, "A", rng)
    # 4, not 3: a placement is only committed once the same set of new
    # cells is confirmed by a SECOND independent settled observation (see
    # GameWatcher.__init__'s _pending_new_cells docstring) -- the 3rd call
    # is the first sighting, the 4th confirms it's stopped growing.
    events = [watcher.observe_board_frame(placed.copy(), player_id="p1") for _ in range(4)]

    final = events[-1]
    assert final.state == WatcherState.APPLIED
    assert final.needs_operator is False
    assert final.scored_move is not None
    assert final.scored_move.candidate.new_cells == (CENTER,)
    assert not watcher.board.is_empty(CENTER)
    assert watcher.board.get(CENTER).letter == "A"
    assert watcher.turn_number == 1

    # Re-observing the same settled state must not re-score the same move.
    repeat = watcher.observe_board_frame(placed.copy(), player_id="p1")
    assert repeat.scored_move is None
    assert watcher.turn_number == 1


def test_low_confidence_routes_to_operator_and_does_not_advance_board(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    # An impossibly strict threshold guarantees every real prediction
    # routes to operator review, regardless of how the tiny classifier
    # happens to score this particular tile.
    watcher = _make_watcher(
        classifier, mode=PublishMode.AUTONOMOUS_WITH_CONFIDENCE_FALLBACK, confidence_threshold=1.01,
    )
    rng = random.Random(7)
    empty = _blank_board_image()
    for _ in range(3):
        watcher.observe_board_frame(empty.copy(), player_id="p1")

    placed = _blank_board_image()
    _place_tile(placed, *CENTER, "A", rng)
    # 4, not 3: a placement is only committed once the same set of new
    # cells is confirmed by a SECOND independent settled observation (see
    # GameWatcher.__init__'s _pending_new_cells docstring) -- the 3rd call
    # is the first sighting, the 4th confirms it's stopped growing.
    events = [watcher.observe_board_frame(placed.copy(), player_id="p1") for _ in range(4)]

    final = events[-1]
    assert final.needs_operator is True
    assert final.pending is not None
    assert watcher.board.is_blank_board()  # not applied -- still awaiting confirmation
    assert watcher.turn_number == 0


def test_detected_blank_always_needs_operator_even_in_autonomous_mode(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)  # would auto-publish anything else
    rng = random.Random(7)
    empty = _blank_board_image()
    for _ in range(3):
        watcher.observe_board_frame(empty.copy(), player_id="p1")

    placed = _blank_board_image()
    _place_tile(placed, *CENTER, None, rng)  # a blank tile
    # 4, not 3: a placement is only committed once the same set of new
    # cells is confirmed by a SECOND independent settled observation (see
    # GameWatcher.__init__'s _pending_new_cells docstring) -- the 3rd call
    # is the first sighting, the 4th confirms it's stopped growing.
    events = [watcher.observe_board_frame(placed.copy(), player_id="p1") for _ in range(4)]

    final = events[-1]
    assert final.needs_operator is True
    assert "blank" in final.reason
    assert watcher.board.is_blank_board()


def test_invalid_placement_routes_to_operator(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)
    rng = random.Random(7)
    empty = _blank_board_image()
    for _ in range(3):
        watcher.observe_board_frame(empty.copy(), player_id="p1")

    # Two new tiles, far apart and disconnected -- not a single contiguous
    # line, and (being the first move) doesn't even cover the center.
    placed = _blank_board_image()
    _place_tile(placed, 0, 0, "A", rng)
    _place_tile(placed, 10, 10, "N", rng)
    # 4, not 3: a placement is only committed once the same set of new
    # cells is confirmed by a SECOND independent settled observation (see
    # GameWatcher.__init__'s _pending_new_cells docstring) -- the 3rd call
    # is the first sighting, the 4th confirms it's stopped growing.
    events = [watcher.observe_board_frame(placed.copy(), player_id="p1") for _ in range(4)]

    final = events[-1]
    assert final.needs_operator is True
    assert final.reason is not None
    assert watcher.board.is_blank_board()


def test_partial_placement_is_not_committed_until_confirmed_stable(tmp_path):
    """Regression test for the exact failure running this against a real,
    complete game found: a player who places part of a multi-tile word,
    then pauses to think (fully satisfying "hands not moving" while
    genuinely mid-turn), must not have that partial placement committed
    as if it were the whole move. This is exactly what split one real
    move ("HUIA") into two separate detected turns, attributed to two
    different players, when GameWatcher was run end-to-end against a full
    real WESPA Word Wars game.
    """
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)
    rng = random.Random(9)

    empty = _blank_board_image()
    _settle(watcher, empty, "p1")

    # Player places 2 of what will eventually be a 3-tile word, then
    # pauses -- 3 identical frames is exactly enough for one settled
    # sighting, not (yet) a confirming second one.
    partial = _blank_board_image()
    _place_tile(partial, 7, 6, "C", rng)
    _place_tile(partial, 7, 7, "A", rng)
    partial_event = _settle(watcher, partial, "p1", count=3)

    assert partial_event.scored_move is None, "a single sighting must never commit on its own"
    assert watcher.board.is_blank_board()
    assert watcher.turn_number == 0

    # Player resumes (hand re-enters the shot, breaking stillness, then
    # leaves again) and places the 3rd tile, completing the real word.
    _disrupt(watcher, "p1")
    full = partial.copy()
    _place_tile(full, 7, 8, "T", rng)
    final = _settle(watcher, full, "p1")

    assert final.state == WatcherState.APPLIED
    assert set(final.scored_move.candidate.new_cells) == {(7, 6), (7, 7), (7, 8)}
    assert watcher.turn_number == 1


def test_a_marginal_cell_that_disappears_does_not_block_a_stable_placement(tmp_path):
    """Regression test for a second real failure found running this
    against a complete real game: after the first fix above (wait for a
    cell to be seen twice), a single marginal/borderline-visibility cell
    that kept flickering in and out of the reading indefinitely could
    still block a perfectly stable *other* cell from ever committing,
    since the old fix compared the whole candidate set for exact
    equality. Confirmation is now tracked per cell instead: a cell that
    stops appearing simply drops out of consideration rather than
    permanently blocking everything else.
    """
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)
    rng = random.Random(15)

    empty = _blank_board_image()
    _settle(watcher, empty, "p1")

    # Reading 1: just the real tile, first sighting.
    stable_only = _blank_board_image()
    _place_tile(stable_only, 7, 7, "A", rng)
    reading1 = _settle(watcher, stable_only, "p1", count=3)
    assert reading1.scored_move is None

    # Reading 2: a second cell flickers into view alongside it (brand new,
    # not yet confirmed).
    _disrupt(watcher, "p1")
    with_marginal = stable_only.copy()
    _place_tile(with_marginal, 7, 8, "N", rng)
    reading2 = _settle(watcher, with_marginal, "p1", count=3)
    assert reading2.scored_move is None

    # Reading 3: the marginal cell is gone again -- the real tile must
    # still commit on its own, not hang forever waiting for (7, 8) to
    # reconfirm.
    _disrupt(watcher, "p1")
    final = _settle(watcher, stable_only, "p1", count=3)

    assert final.state == WatcherState.APPLIED
    assert final.scored_move.candidate.new_cells == ((7, 7),)
    assert watcher.turn_number == 1


def test_reference_adapts_to_lighting_drift_in_still_empty_cells(tmp_path):
    """Regression test for a fifth real issue found running this against
    a complete real game: a single fixed reference photo can't track
    lighting that keeps drifting over a long (30+ minute) broadcast -- a
    screen region far in time from the reference's capture moment
    developed a persistent false-positive occupancy diff, and no single
    fixed reference point covered the whole game without the same
    problem resurfacing elsewhere. GameWatcher now refreshes the
    reference's still-unplayed cells from real frames as the game
    progresses, so gradual drift gets absorbed before it ever crosses the
    occupancy threshold.
    """
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)

    empty = _blank_board_image()
    _settle(watcher, empty, "p1")

    x1, y1, x2, y2 = cell_bounds(3, 3)

    # First increment of lighting drift -- under the default occupancy
    # threshold (25.0) on its own, so this cell stays correctly
    # unoccupied and the reference should refresh to match it.
    drifted_once = empty.copy()
    drifted_once[y1:y2, x1:x2] = np.clip(
        drifted_once[y1:y2, x1:x2].astype(np.int16) + 15, 0, 255
    ).astype(np.uint8)
    _settle(watcher, drifted_once, "p1")
    assert watcher.board.is_blank_board(), "a moderate drift alone must not misread as a placement"

    # A second, equal increment (now +30 from the ORIGINAL reference --
    # enough to spuriously cross the threshold against an un-refreshed
    # reference) should still read as empty, because the reference
    # already caught up to the first increment.
    drifted_twice = empty.copy()
    drifted_twice[y1:y2, x1:x2] = np.clip(
        drifted_twice[y1:y2, x1:x2].astype(np.int16) + 30, 0, 255
    ).astype(np.uint8)
    final = _settle(watcher, drifted_twice, "p1")

    assert final.scored_move is None
    assert watcher.board.is_blank_board(), "incremental drift the reference already tracked must not misread as a placement"


def test_new_cells_hooking_through_an_existing_tile_cluster_together(tmp_path):
    """Regression test for a bug in the clustering fix itself, found
    immediately on the next real-game re-run: clustering only grouped new
    cells that were directly adjacent to EACH OTHER, which splits a single
    real word into two separate clusters whenever it hooks through a
    letter already on the board (exactly the real WESPA move
    "ARB.RIZE", where the "." is a pre-existing tile) -- this fed one
    legitimate 7-cell bingo back as two separate, smaller turns instead of
    the one real turn it was. Two new cells must cluster together if
    they're connected through a straight run of already-occupied cells
    (old or new), not only through other new cells.
    """
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)
    rng = random.Random(23)

    # An existing tile sits in the middle of the word this turn hooks
    # through -- the new cells on either side of it are not adjacent to
    # each other at all.
    watcher._board = BoardState({(7, 7): Tile("R")})

    frame = _blank_board_image()
    for col, letter in [(5, "A"), (6, "B"), (8, "I"), (9, "Z")]:
        _place_tile(frame, 7, col, letter, rng)

    final = _settle(watcher, frame, "p1")

    assert final.state == WatcherState.APPLIED
    assert set(final.scored_move.candidate.new_cells) == {(7, 5), (7, 6), (7, 8), (7, 9)}


def test_when_the_largest_candidate_fails_a_smaller_one_from_the_same_cluster_is_tried(tmp_path):
    """Regression test for the never-jam commit search: a single
    connected cluster can yield more than one candidate placement (see
    `placement_search.py`), and a failure on the first one tried must not
    give up on the whole observation -- the search must fall through to
    the next candidate in the same call.

    An L-shaped cluster -- (0,0),(0,1),(1,1) -- with an existing tile at
    (2,1) yields two same-size candidates: the row-0 run {(0,0),(0,1)}
    and the col-1 run {(0,1),(1,1)}. The row run doesn't connect to
    anything (fails validation); the col run hooks through the existing
    tile at (2,1) (succeeds). The row run sorts first in the deterministic
    tiebreak, so this only passes if the search tries a second candidate
    after the first one fails.
    """
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)
    rng = random.Random(31)

    watcher._board = BoardState({(2, 1): Tile("A")})

    frame = _blank_board_image()
    for row, col, letter in [(0, 0, "A"), (0, 1, "N"), (1, 1, "T")]:
        _place_tile(frame, row, col, letter, rng)

    final = _settle(watcher, frame, "p1")

    assert final.state == WatcherState.APPLIED, f"expected a commit, got {final.reason!r}"
    assert set(final.scored_move.candidate.new_cells) == {(0, 1), (1, 1)}


def test_a_persistently_failing_cluster_never_blocks_an_independent_one(tmp_path):
    """Regression test for the quarantine mechanism's real value: a cell
    that keeps failing (here, a genuine blank tile adjacent to a
    perfectly fine one -- the two always cluster and always get tried
    together, so they fail and eventually quarantine together too, since
    a lone blank has no way to resolve without an operator or a lexicon)
    must never block an entirely UNRELATED, independently valid
    placement elsewhere on the board from committing. Also confirms the
    persistently-failing cell does get quarantined rather than retried
    forever with no bound.
    """
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)
    rng = random.Random(32)

    watcher._board = BoardState({(5, 5): Tile("A"), (10, 10): Tile("A")})

    frame = _blank_board_image()
    _place_tile(frame, 5, 6, "N", rng)  # adjacent to a real blank -- never cleanly resolves
    _place_tile(frame, 5, 7, None, rng)  # a real blank tile, structurally unresolvable alone
    _place_tile(frame, 10, 11, "T", rng)  # entirely independent, cleanly valid placement

    applied = None
    for _ in range(10):
        event = watcher.observe_board_frame(frame.copy(), player_id="p1")
        if event.state == WatcherState.APPLIED:
            applied = event
            break

    assert applied is not None, "the independent placement never committed"
    assert set(applied.scored_move.candidate.new_cells) == {(10, 11)}

    for _ in range(10):
        event = watcher.observe_board_frame(frame.copy(), player_id="p1")
        if (5, 7) in watcher._quarantined:
            break

    assert (5, 7) in watcher._quarantined, "the persistently-failing blank cell was never quarantined"


def test_truncated_candidate_never_auto_publishes_even_at_high_confidence(tmp_path):
    """A candidate smaller than the largest one its cluster could have
    produced is a truncation -- a real, measured risk (see
    `placement_search.py`'s docstring) that must always route to an
    operator, regardless of confidence or publish mode. Tested directly
    against `_attempt_commit` since constructing an end-to-end scenario
    where the full cluster genuinely fails is incidental to what this
    rule actually checks (the size comparison alone).
    """
    from autoscorer.gamelogic.movedetect.constraint_decoder import CellCandidates
    from autoscorer.gamelogic.movedetect.placement_search import CandidatePlacement

    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS, confidence_threshold=0.5)

    candidate = CandidatePlacement(cells=frozenset({CENTER}), cluster_max_size=3)
    cell_candidates = [CellCandidates(coord=CENTER, candidates=[("A", 0.97), ("N", 0.02), ("T", 0.01)])]

    outcome = watcher._attempt_commit(candidate, cell_candidates, "p1")

    assert outcome.needs_operator is True
    assert outcome.confidence > 0.9
    assert watcher.board.is_blank_board(), "a forced-operator candidate must not apply to the board"


def test_watchdog_emits_a_stalled_event_after_repeated_failures(tmp_path, monkeypatch):
    """After enough consecutive failed observations, a `STALLED` watchdog
    event fires as a backstop beyond per-cell quarantine alone -- this is
    what makes a stuck cell visible instead of silently retrying forever
    with nothing in the caller's output to show for it (exactly the
    163-observation real jam this whole redesign targets)."""
    import autoscorer.gamelogic.movedetect.game_watcher as gw_module_local
    monkeypatch.setattr(gw_module_local, "STALL_OBSERVATIONS", 3)

    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)
    rng = random.Random(33)

    frame = _blank_board_image()
    _place_tile(frame, *CENTER, None, rng)  # a lone blank -- structurally unresolvable, first move

    event = None
    for _ in range(8):
        event = watcher.observe_board_frame(frame.copy(), player_id="p1")
        if event.is_stall:
            break

    assert event is not None and event.is_stall is True
    assert event.state == WatcherState.STALLED


def test_a_structurally_valid_but_nonsense_word_still_commits(tmp_path):
    """No lexicon exists yet to check word validity against, and a phony
    (an invalid word) is a completely legal Scrabble play if unchallenged
    -- so this must keep working exactly as-is once one does. "ATN" is
    not a real word; it must still commit and score like any other
    geometrically valid placement."""
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)
    rng = random.Random(34)

    frame = _blank_board_image()
    for col, letter in [(6, "A"), (7, "T"), (8, "N")]:
        _place_tile(frame, 7, col, letter, rng)

    final = _settle(watcher, frame, "p1")

    assert final.state == WatcherState.APPLIED
    assert set(final.scored_move.candidate.new_cells) == {(7, 6), (7, 7), (7, 8)}


def test_two_disconnected_confirmed_clusters_commit_separately_not_merged(tmp_path):
    """Regression test for a third real failure found running this
    against a complete real game: board_before can genuinely have more
    than one turn's worth of unaccounted-for tiles new at once (a slow
    multi-tile word still being placed in one part of the board while an
    unrelated, already-stable cell elsewhere is also new). Both cells
    could pass per-cell confirmation in the same observation, but
    treating them as ONE combined placement fails validation forever
    ("new tiles must lie in a single row or column") since they're
    nowhere near each other -- confirmed cells must be clustered by
    adjacency first, and only one cluster acted on per turn.
    """
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)
    rng = random.Random(21)

    # Two pre-existing anchors, far apart -- stands in for a real board
    # with tiles from earlier turns scattered across it.
    watcher._board = BoardState({(7, 7): Tile("A"), (0, 5): Tile("N")})

    # Two new cells, each legally adjacent to its own anchor but nowhere
    # near each other -- exactly the "two different turns' worth of new
    # tiles" scenario.
    frame = _blank_board_image()
    _place_tile(frame, 7, 8, "T", rng)
    _place_tile(frame, 0, 6, "T", rng)

    # 4 identical calls: settles on the 3rd (first sighting of both
    # cells), confirms on the 4th -- both clusters confirmed together,
    # exactly the scenario that used to fail.
    events = [watcher.observe_board_frame(frame.copy(), player_id="p1") for _ in range(4)]
    final = events[-1]

    assert final.state == WatcherState.APPLIED, "one of the two legal clusters must commit, not fail validation"
    assert len(final.scored_move.candidate.new_cells) == 1, "clusters must not be merged into one scattered placement"
    committed_cell = final.scored_move.candidate.new_cells[0]
    assert committed_cell in {(7, 8), (0, 6)}

    # The other cluster's cell is still sitting there, unapplied -- one
    # more settled look should pick it up as its own, separate turn.
    other_cell = (0, 6) if committed_cell == (7, 8) else (7, 8)
    assert watcher.board.is_empty(other_cell)

    more_events = [watcher.observe_board_frame(frame.copy(), player_id="p2") for _ in range(2)]
    second = more_events[-1]
    assert second.state == WatcherState.APPLIED
    assert second.scored_move.candidate.new_cells == (other_cell,)
    assert watcher.turn_number == 2


class _FakeRackDetector:
    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, image, threshold=0.3):
        sv = pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")
        return sv.Detections(
            xyxy=np.array(self._boxes, dtype=np.float64),
            confidence=np.full(len(self._boxes), 0.95),
            class_id=np.zeros(len(self._boxes), dtype=int),
        )


def _rack_image_with_tiles(letters, rng, gap=10):
    tile_size = FINAL_SIZE
    width = len(letters) * (tile_size + gap) + gap
    height = tile_size + 2 * gap
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (200, 200, 200)

    boxes = []
    for i, letter in enumerate(letters):
        tile = augment_tile(render_tile(letter, rng=rng), rng=rng)
        tile_arr = np.array(tile)[:, :, ::-1]
        x1 = gap + i * (tile_size + gap)
        y1 = gap
        x2, y2 = x1 + tile_size, y1 + tile_size
        image[y1:y2, x1:x2] = tile_arr
        boxes.append((x1, y1, x2, y2))
    return image, boxes


def _settle_rack(watcher, image, player_id, count=3):
    """Feeds the same rack frame in repeatedly until the rack-specific
    stillness gate (mirrors the board path's `_settle`) reports it
    settled -- identical consecutive frames read as zero motion, so this
    always reaches a real (non-None-due-to-still-settling) result."""
    event = None
    for _ in range(count):
        event = watcher.record_rack(player_id, image.copy())
    return event


def test_record_rack_first_observation_is_silent(tmp_path):
    pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")
    classifier = _train_tiny_classifier(tmp_path)
    rng = random.Random(3)
    rack_image, boxes = _rack_image_with_tiles(["A", "N", "T"], rng)
    watcher = _make_watcher(classifier, rack_detector=_FakeRackDetector(boxes))

    event = _settle_rack(watcher, rack_image, "p1")

    assert event is None
    assert [t.letter for t in watcher.racks["p1"]] == ["A", "N", "T"]


def test_record_rack_mid_rearrangement_is_not_read_as_settled(tmp_path):
    """A hand still moving frame-to-frame over the rack (simulated here
    with a large hand-sized occlusion alternating in and out, the same
    technique the board path's own stillness test uses -- a rack-sized
    version of a single differing tile turned out too small a change to
    reliably register on the whole-frame motion signal) must never reach
    a real rack read -- exactly the board path's HANDS_OVER_BOARD-style
    protection, applied to racks."""
    pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")
    classifier = _train_tiny_classifier(tmp_path)
    rng = random.Random(3)
    image, boxes = _rack_image_with_tiles(["A", "N", "T"], rng)
    occluded = image.copy()
    occluded[:, : image.shape[1] // 2] = 255  # a hand-sized patch over half the rack
    watcher = _make_watcher(classifier, rack_detector=_FakeRackDetector(boxes))

    for i in range(6):
        event = watcher.record_rack("p1", image.copy() if i % 2 == 0 else occluded.copy())
        assert event is None

    assert "p1" not in watcher.racks


def test_record_rack_reports_exchange_on_genuine_change(tmp_path):
    pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")
    classifier = _train_tiny_classifier(tmp_path)
    rng = random.Random(3)
    first_image, first_boxes = _rack_image_with_tiles(["A", "N", "T"], rng)
    watcher = _make_watcher(classifier, rack_detector=_FakeRackDetector(first_boxes))
    _settle_rack(watcher, first_image, "p1")  # establishes the starting rack

    second_image, second_boxes = _rack_image_with_tiles(["N", "A", "N"], rng)
    watcher.rack_detector = _FakeRackDetector(second_boxes)
    event = _settle_rack(watcher, second_image, "p1")

    assert event is not None
    assert event.state == WatcherState.APPLIED
    assert event.scored_move.candidate.move_type.value == "EXCHANGE"
    assert [t.letter for t in watcher.racks["p1"]] == ["N", "A", "N"]


def test_record_rack_reordering_only_is_not_reported_as_a_change(tmp_path):
    pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")
    classifier = _train_tiny_classifier(tmp_path)
    rng = random.Random(3)
    first_image, first_boxes = _rack_image_with_tiles(["A", "N", "T"], rng)
    watcher = _make_watcher(classifier, rack_detector=_FakeRackDetector(first_boxes))
    _settle_rack(watcher, first_image, "p1")

    # Same multiset, different order -- a player rearranging their own
    # tiles, not an exchange.
    reordered_image, reordered_boxes = _rack_image_with_tiles(["T", "A", "N"], rng)
    watcher.rack_detector = _FakeRackDetector(reordered_boxes)
    event = _settle_rack(watcher, reordered_image, "p1")

    assert event is None


def test_record_rack_without_detector_raises(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, rack_detector=None)
    with pytest.raises(ValueError):
        watcher.record_rack("p1", np.zeros((150, 900, 3), dtype=np.uint8))


def _settle(watcher, frame, player_id, count=4):
    event = None
    for _ in range(count):
        event = watcher.observe_board_frame(frame.copy(), player_id=player_id)
    return event


def _disrupt(watcher, player_id):
    # A large occlusion (stands in for a hand reaching across the board)
    # breaks the trailing settled window without itself needing to be
    # part of any later "settled" run -- see test_stays_idle... above for
    # why a single-tile change alone isn't enough motion signal.
    frame = _blank_board_image()
    frame[200:600, :] = np.random.randint(0, 255)
    watcher.observe_board_frame(frame, player_id=player_id)


def test_multi_turn_synthetic_game_sequences_correctly(tmp_path):
    """The integration proof the audit called for: a full run of the real
    state machine (stillness gate -> temporal-voted read -> constraint
    decode -> validate -> resolve words -> score -> publish gate -> apply)
    across *multiple sequential turns*, not a single already-known move.

    Honest about what this does and doesn't prove: every frame here is
    synthetically rendered (no real broadcast video exists locally to
    drive this against -- see game_watcher.py's module docstring). This
    proves the sequencing and multi-turn state accumulation are wired
    correctly; it does not substitute for running this against real
    continuous footage, which is the natural next validation step.

    Runs in AUTONOMOUS mode deliberately, isolating "does the sequencing
    and cross-turn state accumulation work" from "does the confidence
    gate work" (separately covered by
    test_low_confidence_routes_to_operator_and_does_not_advance_board) --
    a turn that isn't applied would leave board_before stale for every
    subsequent turn, compounding into placements that fail validation for
    reasons that have nothing to do with what this test is checking.
    """
    classifier = _train_tiny_classifier(tmp_path)
    watcher = _make_watcher(classifier, mode=PublishMode.AUTONOMOUS)

    _settle(watcher, _blank_board_image(), "p1")  # establish the empty board

    # Turn 1 (p1): A at center -- the only legal first move shape here.
    frame = _blank_board_image()
    _place_tile(frame, 7, 7, "A", random.Random(1))
    _disrupt(watcher, "p1")
    turn1 = _settle(watcher, frame, "p1")

    # Turn 2 (p2): N at (7,8) -- extends the existing run to "AN".
    _place_tile(frame, 7, 8, "N", random.Random(2))
    _disrupt(watcher, "p2")
    turn2 = _settle(watcher, frame, "p2")

    # Turn 3 (p1): T at (7,9) -- extends to "ANT".
    _place_tile(frame, 7, 9, "T", random.Random(3))
    _disrupt(watcher, "p1")
    turn3 = _settle(watcher, frame, "p1")

    # Turn 4 (p2): N below the T at (8,9) -- a perpendicular cross-word.
    _place_tile(frame, 8, 9, "N", random.Random(4))
    _disrupt(watcher, "p2")
    turn4 = _settle(watcher, frame, "p2")

    turns = [turn1, turn2, turn3, turn4]
    # AUTONOMOUS mode always applies (see the docstring for why), so this
    # isn't measuring gateway behavior -- it's an FYI readout of what a
    # 0.9-threshold confidence-fallback gateway *would* have done with
    # these same reads, the same M3/M4-style question WS5 asked, just
    # computed here instead of actually gating on it.
    would_clear_gate = sum(1 for t in turns if t.confidence >= 0.9)
    print(
        f"\n{len(turns)} synthetic turns applied; {would_clear_gate} would have "
        f"cleared a 0.9-confidence gate (informational only -- mode is AUTONOMOUS)"
    )

    # Every turn should have produced *some* outcome (not silently vanish),
    # and every turn's coordinates should exactly match what was actually
    # placed -- the real assertion that state carried forward correctly
    # turn-to-turn, not just that events fired.
    assert all(t.scored_move is not None for t in turns), "every turn should be detected as some move"
    assert all(not t.needs_operator for t in turns), "AUTONOMOUS mode should never route to operator"
    assert turn1.scored_move.candidate.new_cells == ((7, 7),)
    assert turn2.scored_move.candidate.new_cells == ((7, 8),)
    assert turn3.scored_move.candidate.new_cells == ((7, 9),)
    assert turn4.scored_move.candidate.new_cells == ((8, 9),)

    # The board actually accumulated across turns, in order -- not just
    # four independent single-turn reads.
    assert watcher.board.get((7, 7)).letter == "A"
    assert watcher.board.get((7, 8)).letter == "N"
    assert watcher.board.get((7, 9)).letter == "T"
    assert watcher.board.get((8, 9)).letter == "N"
    assert watcher.turn_number == 4


# --- Delegated mode: GameWatcher driving a live GameSession -----------------

def _make_session_watcher(classifier, session, rack_detector=None):
    return GameWatcher(
        calibration=IDENTITY_CALIBRATION,
        reference_board=_blank_board_image(),
        classifier=classifier,
        session=session,
        rack_detector=rack_detector,
        still_frame_count=3,
    )


def test_requires_either_publish_gateway_or_session(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    with pytest.raises(ValueError):
        GameWatcher(
            calibration=IDENTITY_CALIBRATION, reference_board=_blank_board_image(), classifier=classifier,
        )


def test_delegated_mode_auto_publish_applies_to_the_session_not_a_separate_board(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    session = GameSession(mode=PublishMode.AUTONOMOUS)
    watcher = _make_session_watcher(classifier, session)
    rng = random.Random(7)

    empty = _blank_board_image()
    for _ in range(3):
        watcher.observe_board_frame(empty.copy(), player_id="p1")

    placed = _blank_board_image()
    _place_tile(placed, *CENTER, "A", rng)
    # 4, not 3: a placement is only committed once the same set of new
    # cells is confirmed by a SECOND independent settled observation (see
    # GameWatcher.__init__'s _pending_new_cells docstring) -- the 3rd call
    # is the first sighting, the 4th confirms it's stopped growing.
    events = [watcher.observe_board_frame(placed.copy(), player_id="p1") for _ in range(4)]

    final = events[-1]
    assert final.needs_operator is False
    assert final.scored_move.candidate.new_cells == (CENTER,)

    # The move landed in the SESSION's game state -- not some parallel
    # board only the watcher knows about. watcher.board is a live view
    # onto the same object, not a copy.
    assert session.game_state.board.get(CENTER).letter == "A"
    assert watcher.board is session.game_state.board
    assert session.game_state.scores["p1"] == final.scored_move.move_score.total
    assert watcher.turn_number == 1


def test_delegated_mode_low_confidence_becomes_a_real_pending_move_in_the_session(tmp_path):
    classifier = _train_tiny_classifier(tmp_path)
    # AUTONOMOUS_WITH_CONFIDENCE_FALLBACK at an impossible threshold guarantees
    # this real prediction needs operator review, regardless of its own score.
    session = GameSession(mode=PublishMode.AUTONOMOUS_WITH_CONFIDENCE_FALLBACK)
    session.gateway.confidence_threshold = 1.01
    watcher = _make_session_watcher(classifier, session)
    rng = random.Random(7)

    empty = _blank_board_image()
    for _ in range(3):
        watcher.observe_board_frame(empty.copy(), player_id="p1")

    placed = _blank_board_image()
    _place_tile(placed, *CENTER, "A", rng)
    # 4, not 3: a placement is only committed once the same set of new
    # cells is confirmed by a SECOND independent settled observation (see
    # GameWatcher.__init__'s _pending_new_cells docstring) -- the 3rd call
    # is the first sighting, the 4th confirms it's stopped growing.
    events = [watcher.observe_board_frame(placed.copy(), player_id="p1") for _ in range(4)]

    final = events[-1]
    assert final.needs_operator is True
    assert final.pending is not None
    assert session.game_state.board.is_blank_board()  # not applied yet

    # It's a REAL pending move in the session -- the same operator-review
    # queue a manually-typed low-confidence move would land in, approvable
    # through the same session.decide() a human operator's "approve"
    # click already calls.
    pending_list = session.list_pending()
    assert len(pending_list) == 1
    assert pending_list[0].scored_move.candidate.new_cells == (CENTER,)

    turn_number = final.scored_move.candidate.turn_number
    applied = session.decide(turn_number, "approve")
    assert applied is True
    assert session.game_state.board.get(CENTER).letter == "A"


def test_delegated_mode_record_rack_first_observation_seeds_session_racks_silently(tmp_path):
    pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")
    classifier = _train_tiny_classifier(tmp_path)
    session = GameSession(mode=PublishMode.AUTONOMOUS)
    rng = random.Random(3)
    rack_image, boxes = _rack_image_with_tiles(["A", "N", "T"], rng)
    watcher = _make_session_watcher(classifier, session, rack_detector=_FakeRackDetector(boxes))

    event = _settle_rack(watcher, rack_image, "p1")

    assert event is None
    assert [t.letter for t in session.game_state.racks["p1"]] == ["A", "N", "T"]
    assert session.game_state.history == []  # establishing a starting rack isn't a turn


def test_delegated_mode_record_rack_exchange_goes_through_session_submit_move(tmp_path):
    pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")
    classifier = _train_tiny_classifier(tmp_path)
    session = GameSession(mode=PublishMode.AUTONOMOUS)
    rng = random.Random(3)
    first_image, first_boxes = _rack_image_with_tiles(["A", "N", "T"], rng)
    watcher = _make_session_watcher(classifier, session, rack_detector=_FakeRackDetector(first_boxes))
    _settle_rack(watcher, first_image, "p1")

    second_image, second_boxes = _rack_image_with_tiles(["N", "A", "N"], rng)
    watcher.rack_detector = _FakeRackDetector(second_boxes)
    event = _settle_rack(watcher, second_image, "p1")

    assert event is not None
    assert event.needs_operator is False
    assert event.scored_move.candidate.move_type == MoveType.EXCHANGE
    assert [t.letter for t in session.game_state.racks["p1"]] == ["N", "A", "N"]
    assert len(session.game_state.history) == 1
