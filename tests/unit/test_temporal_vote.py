import pytest

from autoscorer.gamelogic.movedetect.temporal_vote import temporal_vote


def test_unanimous_frames_keep_their_confidence():
    frames = [
        [("A", 0.9), ("N", 0.1)],
        [("A", 0.8), ("N", 0.2)],
        [("A", 0.95), ("N", 0.05)],
    ]
    result = temporal_vote(frames)
    assert result[0][0] == "A"
    assert result[0][1] == pytest.approx((0.9 + 0.8 + 0.95) / 3)


def test_minority_misread_is_outvoted():
    # Four frames confidently say "E"; one frame, presumably caught at a
    # bad angle or mid-glare, confidently says "F" instead. The whole
    # point of voting is that this single dissenting frame shouldn't win.
    frames = [
        [("E", 0.85), ("F", 0.10)],
        [("E", 0.80), ("F", 0.15)],
        [("F", 0.90), ("E", 0.05)],  # the minority misread
        [("E", 0.88), ("F", 0.08)],
        [("E", 0.82), ("F", 0.12)],
    ]
    result = temporal_vote(frames)
    winner, confidence = result[0]
    assert winner == "E"
    # E's average should clear a normal 0.9 gate even though no single
    # frame's own confidence for "F" ever mattered once outvoted.
    assert confidence == pytest.approx((0.85 + 0.80 + 0.05 + 0.88 + 0.82) / 5)


def test_labels_missing_from_some_frames_are_treated_as_zero_there():
    # Frame 2 doesn't mention "Z" at all (e.g. a truncated top-k list) --
    # it should count as 0 confidence for that frame, not be excluded
    # from the average's denominator.
    frames = [
        [("Z", 0.7), ("I", 0.2)],
        [("I", 0.6), ("N", 0.3)],
    ]
    result = temporal_vote(frames)
    by_label = dict(result)
    assert by_label["Z"] == pytest.approx(0.7 / 2)
    assert by_label["I"] == pytest.approx((0.2 + 0.6) / 2)


def test_single_frame_is_returned_unchanged_in_ranked_order():
    frames = [[("Q", 0.4), ("O", 0.9), ("D", 0.1)]]
    result = temporal_vote(frames)
    assert result == [("O", 0.9), ("Q", 0.4), ("D", 0.1)]


def test_requires_at_least_one_frame():
    with pytest.raises(ValueError):
        temporal_vote([])
