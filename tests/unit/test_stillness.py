import numpy as np

from autoscorer.perception.stillness.detector import frame_motion_score, is_settled, stable_window

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
