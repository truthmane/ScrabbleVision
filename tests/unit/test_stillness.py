import time

import numpy as np

from autoscorer.perception.stillness.detector import (
    StillnessTracker,
    frame_motion_score,
    is_settled,
    stable_window,
)

FRAME_SIZE = 100


def _plain_frame() -> np.ndarray:
    return np.full((FRAME_SIZE, FRAME_SIZE, 3), 200, dtype=np.uint8)


def _frame_with_moving_hand(position: int) -> np.ndarray:
    """A large dark block (a stand-in for a hand/tile covering a
    realistic fraction of the frame -- real hand-over-board footage
    scored ~38 on this signal, see detector.py's calibration note) at a
    different position each frame, simulating something physically
    moving over the board."""
    frame = _plain_frame()
    frame[20:80, position:position + 40] = 30
    return frame


def test_identical_frames_score_zero_motion():
    a = _plain_frame()
    b = _plain_frame()
    assert frame_motion_score(a, b) == 0.0


def test_a_moving_hand_scores_high_motion():
    a = _frame_with_moving_hand(10)
    b = _frame_with_moving_hand(50)
    assert frame_motion_score(a, b) > 5.0


def test_is_settled_true_once_enough_still_frames_seen():
    frames = [_plain_frame() for _ in range(5)]
    assert is_settled(frames, required_still_frames=5)


def test_is_settled_false_with_too_few_frames():
    frames = [_plain_frame() for _ in range(3)]
    assert not is_settled(frames, required_still_frames=5)


def test_is_settled_false_while_a_hand_is_still_moving_within_the_window():
    # The last frame in the window still differs from its predecessor --
    # a hand is mid-motion, board not safe to read yet.
    frames = [_plain_frame() for _ in range(4)] + [_frame_with_moving_hand(50)]
    assert not is_settled(frames, required_still_frames=5)


def test_is_settled_true_once_a_hand_has_left_the_trailing_window():
    # Motion happened earlier (a hand moved through), but the most recent
    # `required_still_frames` frames are all stable again -- the gate
    # should recover immediately, not wait for some unrelated cooldown.
    moving = [_frame_with_moving_hand(p) for p in (10, 20, 30)]
    settled_again = [_plain_frame() for _ in range(5)]
    frames = moving + settled_again
    assert is_settled(frames, required_still_frames=5)


def test_stable_window_returns_none_when_not_settled():
    frames = [_plain_frame() for _ in range(2)]
    assert stable_window(frames, required_still_frames=5) is None


def test_stable_window_returns_the_trailing_frames_when_settled():
    frames = [_frame_with_moving_hand(5)] + [_plain_frame() for _ in range(5)]
    window = stable_window(frames, required_still_frames=5)
    assert window is not None
    assert len(window) == 5
    assert all(np.array_equal(f, _plain_frame()) for f in window)


def _push_all(tracker: StillnessTracker, frames):
    for frame in frames:
        tracker.push(frame)
    return tracker


def test_stillness_tracker_matches_is_settled_and_stable_window():
    frames = [_frame_with_moving_hand(p) for p in (10, 20, 30)] + [_plain_frame() for _ in range(5)]
    tracker = _push_all(StillnessTracker(required_still_frames=5), frames)

    assert tracker.is_settled() == is_settled(frames, required_still_frames=5)
    expected_window = stable_window(frames, required_still_frames=5)
    actual_window = tracker.stable_window()
    assert len(actual_window) == len(expected_window)
    assert all(np.array_equal(a, b) for a, b in zip(actual_window, expected_window))


def test_stillness_tracker_false_with_too_few_frames():
    tracker = _push_all(StillnessTracker(required_still_frames=5), [_plain_frame() for _ in range(3)])
    assert not tracker.is_settled()
    assert tracker.stable_window() is None


def test_stillness_tracker_recovers_once_hand_leaves_trailing_window():
    moving = [_frame_with_moving_hand(p) for p in (10, 20, 30)]
    settled_again = [_plain_frame() for _ in range(5)]
    tracker = _push_all(StillnessTracker(required_still_frames=5), moving + settled_again)
    assert tracker.is_settled()


def test_stillness_tracker_last_pair_still_tracks_the_newest_transition():
    tracker = StillnessTracker(required_still_frames=5)
    assert tracker.last_pair_still is None

    tracker.push(_plain_frame())
    assert tracker.last_pair_still is None  # only one frame seen so far

    tracker.push(_plain_frame())
    assert tracker.last_pair_still is True

    tracker.push(_frame_with_moving_hand(50))
    assert tracker.last_pair_still is False

    tracker.push(_frame_with_moving_hand(10))  # hand still moving, but motion recorded again
    assert tracker.last_pair_still is False


def test_stillness_tracker_incremental_pushes_dont_blow_up_with_a_large_window():
    # Regression test for the WS3 sample_fps bug: a venue calibrated at a
    # low sample_fps but replayed at a much higher one derives a large
    # `required_still_frames` window (see VenueProfile.
    # effective_still_frame_count). The naive is_settled/stable_window
    # approach recomputes the whole window's pairwise motion scores on
    # every single call, so per-frame cost scales with window size --
    # this proves the incremental tracker doesn't.
    required_still_frames = 200
    tracker = StillnessTracker(required_still_frames=required_still_frames)
    frame = _plain_frame()

    # Warm up to a full, settled window first.
    for _ in range(required_still_frames):
        tracker.push(frame)
    assert tracker.is_settled()

    start = time.perf_counter()
    for _ in range(2000):
        tracker.push(frame)
        assert tracker.is_settled()
    elapsed = time.perf_counter() - start

    # At O(window) per call this would be ~2000 * 200 = 400k score calls;
    # incrementally it's exactly 2000. Generous bound to avoid flakiness
    # while still catching an accidental regression to the naive approach.
    assert elapsed < 5.0
