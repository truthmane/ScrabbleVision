import random
import threading

import cv2
import numpy as np
import pytest
import torch

from autoscorer.api.session import GameSession
from autoscorer.gamelogic.board import CENTER
from autoscorer.gamelogic.models import MoveType
from autoscorer.gamelogic.publish import PublishMode
from autoscorer.perception.calibration.homography import CANONICAL_SIZE
from autoscorer.perception.calibration.venue_profile import VenueProfile, save_venue_profile
from autoscorer.perception.capture.run_watcher import format_event, run_watcher_on_video
from training.classify.infer import TileClassifierModel
from training.classify.train import run_training, save_checkpoint
from training.synth_render.tile_renderer import augment_tile, render_tile

from tests.support.synth_board import blank_board_image as _blank_board_image
from tests.support.synth_board import place_tile as _place_tile_full

cv2_module = pytest.importorskip("cv2")

# Corners chosen so calibrate_from_corners produces an (approximately)
# identity mapping for a video already shot at canonical resolution --
# avoids conflating "does rectification work" (covered by
# test_calibration.py) with "does the runner wire things together".
IDENTITY_CORNERS = ((0.0, 0.0), (CANONICAL_SIZE, 0.0), (CANONICAL_SIZE, CANONICAL_SIZE), (0.0, CANONICAL_SIZE))


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
    return checkpoint_path


def _write_video(path, frames, fps=10.0) -> None:
    size = (frames[0].shape[1], frames[0].shape[0])
    # Lossless-ish codec: mp4v distorts small glyph details enough to risk
    # misclassifying a 60x60 synthetic tile crop after compression; FFV1
    # keeps this a fair test of the *wiring*, not of compression robustness
    # (a separate, already-covered concern -- see the real-footage work in
    # game_watcher.py's docstring).
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_format_event_returns_none_for_no_op_events():
    from autoscorer.gamelogic.movedetect.game_watcher import WatcherEvent, WatcherState
    event = WatcherEvent(state=WatcherState.BOARD_SETTLED)
    assert format_event(event, "p1") is None


def test_run_watcher_on_video_detects_a_play_and_alternates_players(tmp_path):
    checkpoint_path = _train_tiny_classifier(tmp_path)

    reference = _blank_board_image()
    rng = random.Random(7)

    still_frame_count = 3
    frames = [reference.copy() for _ in range(still_frame_count)]  # settle on empty board

    placed = reference.copy()
    _place_tile(placed, *CENTER, "A", rng)
    frames.append(reference.copy())  # a motion stand-in frame (differs from both settled states)
    frames.append(np.zeros_like(reference))
    # +1: a placement is only committed once the same set of new cells is
    # confirmed by a SECOND independent settled observation (see
    # GameWatcher's _pending_new_cells) -- still_frame_count identical
    # frames only reaches the first sighting, one more confirms it.
    frames += [placed.copy() for _ in range(still_frame_count + 1)]  # settle on the new placement, twice

    video_path = tmp_path / "test_game.mkv"
    _write_video(video_path, frames)

    reference_path = tmp_path / "reference.png"
    cv2.imwrite(str(reference_path), reference)
    profile = VenueProfile(
        name="test_venue", corners=IDENTITY_CORNERS, still_frame_count=still_frame_count,
        reference_board_path=str(reference_path),
    )
    save_venue_profile(profile, directory=tmp_path)

    # run_watcher_on_video loads the profile by name via the default venues
    # directory -- point it at our tmp_path by monkeypatching the module's
    # lookup instead of relying on a directory argument it doesn't expose.
    import autoscorer.perception.capture.run_watcher as run_watcher_module
    original_loader = run_watcher_module.load_venue_profile
    run_watcher_module.load_venue_profile = lambda name: original_loader(name, directory=tmp_path)
    try:
        events = run_watcher_on_video(
            video_path, "test_venue", checkpoint_path, "Alice", "Bob",
            sample_fps=None, mode=PublishMode.AUTONOMOUS,
        )
    finally:
        run_watcher_module.load_venue_profile = original_loader

    assert len(events) == 1
    event = events[0]
    assert event.scored_move.candidate.move_type == MoveType.PLAY
    assert event.scored_move.candidate.new_cells == (CENTER,)
    assert not event.needs_operator


def test_run_watcher_on_video_pauses_while_a_move_is_pending(tmp_path):
    """Regression test for a real bug found on a live run: with nothing
    deciding a pending candidate, the loop kept observing the still-settled
    board and re-detected + re-submitted the *same* real move as a new
    duplicate PendingMove every subsequent stable observation (one bingo
    piled up 9 duplicate entries in practice). In delegated (session) mode
    the loop must instead block -- not observe any further frames -- while
    `session.pending` is non-empty, and resume only once the operator
    decides."""
    checkpoint_path = _train_tiny_classifier(tmp_path)
    reference = _blank_board_image()
    rng = random.Random(11)

    still_frame_count = 3
    frames = [reference.copy() for _ in range(still_frame_count)]

    placed = reference.copy()
    _place_tile(placed, *CENTER, "A", rng)
    frames.append(reference.copy())  # motion stand-in
    frames.append(np.zeros_like(reference))
    # Many extra settled observations of the *same* placement after the
    # first commit -- previously each one re-triggered a duplicate commit.
    frames += [placed.copy() for _ in range(still_frame_count + 10)]

    video_path = tmp_path / "pending_pause.mkv"
    _write_video(video_path, frames)

    reference_path = tmp_path / "reference.png"
    cv2.imwrite(str(reference_path), reference)
    profile = VenueProfile(
        name="pause_test_venue", corners=IDENTITY_CORNERS, still_frame_count=still_frame_count,
        reference_board_path=str(reference_path),
    )
    save_venue_profile(profile, directory=tmp_path)

    import autoscorer.perception.capture.run_watcher as run_watcher_module
    original_loader = run_watcher_module.load_venue_profile
    run_watcher_module.load_venue_profile = lambda name: original_loader(name, directory=tmp_path)

    session = GameSession(mode=PublishMode.MANUAL)

    def run() -> None:
        run_watcher_on_video(
            video_path, "pause_test_venue", checkpoint_path, "Alice", "Bob",
            sample_fps=None, session=session,
        )

    thread = threading.Thread(target=run, daemon=True)
    try:
        thread.start()
        thread.join(timeout=5.0)
        assert thread.is_alive()  # blocked waiting on the pending decision, not finished

        pending = session.list_pending()
        assert len(pending) == 1  # not duplicated across the many repeated settled frames

        session.decide(pending[0].scored_move.candidate.turn_number, "approve")
        thread.join(timeout=15.0)
        assert not thread.is_alive()
    finally:
        run_watcher_module.load_venue_profile = original_loader

    assert session.list_pending() == []


def test_run_watcher_on_video_resolves_lexicon_from_venue_profile_or_override(tmp_path):
    checkpoint_path = _train_tiny_classifier(tmp_path)
    reference = _blank_board_image()
    frames = [reference.copy() for _ in range(4)]
    video_path = tmp_path / "empty.mkv"
    _write_video(video_path, frames)
    reference_path = tmp_path / "reference.png"
    cv2.imwrite(str(reference_path), reference)
    profile = VenueProfile(
        name="lexicon_test_venue", corners=IDENTITY_CORNERS, still_frame_count=3,
        reference_board_path=str(reference_path), lexicon=None,
    )
    save_venue_profile(profile, directory=tmp_path)

    import autoscorer.perception.capture.run_watcher as run_watcher_module
    original_loader = run_watcher_module.load_venue_profile
    original_load_lexicon = run_watcher_module.load_lexicon
    calls = []

    def _spy_load_lexicon(name):
        calls.append(name)
        return original_load_lexicon()  # always the real, always-loadable vendored default

    run_watcher_module.load_venue_profile = lambda name: original_loader(name, directory=tmp_path)
    run_watcher_module.load_lexicon = _spy_load_lexicon
    try:
        run_watcher_on_video(
            video_path, "lexicon_test_venue", checkpoint_path, "Alice", "Bob",
            sample_fps=None, mode=PublishMode.AUTONOMOUS,
        )
        assert calls == [None]  # no override, profile.lexicon is None -> vendored default

        calls.clear()
        run_watcher_on_video(
            video_path, "lexicon_test_venue", checkpoint_path, "Alice", "Bob",
            sample_fps=None, mode=PublishMode.AUTONOMOUS, lexicon_name="some_override",
        )
        assert calls == ["some_override"]
    finally:
        run_watcher_module.load_venue_profile = original_loader
        run_watcher_module.load_lexicon = original_load_lexicon


def test_run_watcher_on_video_derives_still_frame_count_from_still_seconds_and_sample_fps(tmp_path):
    """Regression test for the real bug this whole mechanism exists to
    prevent: a venue profile's raw still_frame_count has no fixed
    real-world meaning without knowing the rate it was calibrated
    against, and a CLI defaulting to a different sample_fps than a
    profile was tuned at silently changes what "settled" means -- this
    produced a full evening of what looked exactly like a code
    regression before being traced back to the actual cause. With
    still_seconds set, the frame count actually used must be derived
    from the real sample_fps a run passes in, not the profile's raw
    (and now purely informational) still_frame_count field.
    """
    checkpoint_path = _train_tiny_classifier(tmp_path)
    reference = _blank_board_image()
    frames = [reference.copy() for _ in range(4)]
    video_path = tmp_path / "empty.mkv"
    _write_video(video_path, frames)
    reference_path = tmp_path / "reference.png"
    cv2.imwrite(str(reference_path), reference)
    profile = VenueProfile(
        name="timed_test_venue", corners=IDENTITY_CORNERS,
        still_frame_count=999,  # must be ignored once still_seconds is set
        still_seconds=2.5, reference_board_path=str(reference_path),
    )
    save_venue_profile(profile, directory=tmp_path)

    import autoscorer.perception.capture.run_watcher as run_watcher_module
    original_loader = run_watcher_module.load_venue_profile
    original_game_watcher = run_watcher_module.GameWatcher
    captured = {}

    class _SpyGameWatcher(original_game_watcher):
        def __init__(self, *args, **kwargs):
            captured["still_frame_count"] = kwargs.get("still_frame_count")
            super().__init__(*args, **kwargs)

    run_watcher_module.load_venue_profile = lambda name: original_loader(name, directory=tmp_path)
    run_watcher_module.GameWatcher = _SpyGameWatcher
    try:
        run_watcher_on_video(
            video_path, "timed_test_venue", checkpoint_path, "Alice", "Bob",
            sample_fps=2.0, mode=PublishMode.AUTONOMOUS,
        )
        assert captured["still_frame_count"] == 5  # 2.5s * 2.0fps, not the raw 999
    finally:
        run_watcher_module.load_venue_profile = original_loader
        run_watcher_module.GameWatcher = original_game_watcher


def test_format_event_includes_score_and_status_for_a_play():
    from autoscorer.gamelogic.models import MoveCandidate, ScoredMove
    from autoscorer.gamelogic.movedetect.game_watcher import WatcherEvent, WatcherState
    from autoscorer.gamelogic.scoring.rules_engine import MoveScore, WordScore

    candidate = MoveCandidate(turn_number=1, player_id="Alice", move_type=MoveType.PLAY, new_cells=(CENTER,))
    move_score = MoveScore(words=[WordScore(cells=[CENTER], text="A", score=2)], is_bingo=False, total=2)
    event = WatcherEvent(
        state=WatcherState.APPLIED,
        scored_move=ScoredMove(candidate=candidate, move_score=move_score),
        confidence=0.95,
    )
    line = format_event(event, "Alice")
    assert "Alice" in line
    assert "score=2" in line
    assert "AUTO-PUBLISHED" in line
