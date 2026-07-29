import numpy as np

from autoscorer.perception.clock.clock_reader import (
    ClockRegions,
    DEFAULT_RED_DOMINANCE_THRESHOLD,
    detect_active_side,
)

REGIONS = ClockRegions(left=(0, 0, 10, 10), right=(20, 0, 30, 10))


def _solid_bgr(b: int, g: int, r: int, size: int = 40) -> np.ndarray:
    frame = np.zeros((10, size, 3), dtype=np.uint8)
    frame[..., 0] = b
    frame[..., 1] = g
    frame[..., 2] = r
    return frame


def _frame_with(left_bgr, right_bgr) -> np.ndarray:
    frame = np.zeros((10, 30, 3), dtype=np.uint8)
    frame[0:10, 0:10] = left_bgr
    frame[0:10, 20:30] = right_bgr
    return frame


def test_left_red_right_white_reads_as_left_active():
    # Real measured values: red box ~R=140,G=71,B=72; white box ~R=148,G=148,B=147.
    frame = _frame_with(left_bgr=(72, 71, 140), right_bgr=(147, 148, 148))
    assert detect_active_side(frame, REGIONS) == "left"


def test_right_red_left_white_reads_as_right_active():
    frame = _frame_with(left_bgr=(147, 148, 148), right_bgr=(53, 53, 103))
    assert detect_active_side(frame, REGIONS) == "right"


def test_both_white_is_indeterminate():
    frame = _frame_with(left_bgr=(148, 148, 148), right_bgr=(148, 148, 148))
    assert detect_active_side(frame, REGIONS) is None


def test_both_red_is_indeterminate_not_a_guess():
    """A transition frame (mid color-swap) must never be resolved by
    picking one side arbitrarily -- the caller relies on `None` meaning
    "no new information," not "guess.\""""
    frame = _frame_with(left_bgr=(72, 71, 140), right_bgr=(53, 53, 103))
    assert detect_active_side(frame, REGIONS) is None


def test_neither_red_is_indeterminate():
    frame = _frame_with(left_bgr=(20, 20, 20), right_bgr=(20, 20, 20))
    assert detect_active_side(frame, REGIONS) is None


def test_borderline_dominance_respects_the_threshold():
    # R - max(G, B) == exactly the default threshold -- must NOT count as red
    # (a strict ">" comparison, not ">="), keeping the boundary unambiguous.
    r = 100.0
    g = b = r - DEFAULT_RED_DOMINANCE_THRESHOLD
    frame = _frame_with(left_bgr=(b, g, r), right_bgr=(148, 148, 148))
    assert detect_active_side(frame, REGIONS) is None


def test_custom_threshold_is_respected():
    # A weak red tint that clears a low custom threshold but not the default.
    frame = _frame_with(left_bgr=(140, 135, 150), right_bgr=(148, 148, 148))
    assert detect_active_side(frame, REGIONS) is None
    assert detect_active_side(frame, REGIONS, red_dominance_threshold=5.0) == "left"
