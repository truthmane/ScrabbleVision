from autoscorer.eval.alignment import align_turns, jaccard


def _kinds(ops):
    return [(o.kind, o.detected_indices, o.truth_indices) for o in ops]


def test_jaccard_identical_sets_is_one():
    a = frozenset({(0, 0), (0, 1)})
    assert jaccard(a, a) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert jaccard(frozenset({(0, 0)}), frozenset({(5, 5)})) == 0.0


def test_jaccard_both_empty_is_one():
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_clean_one_to_one_sequence_matches_every_turn():
    d = [frozenset({(0, 0)}), frozenset({(0, 1), (0, 2)}), frozenset({(1, 1)})]
    t = [frozenset({(0, 0)}), frozenset({(0, 1), (0, 2)}), frozenset({(1, 1)})]
    ops = align_turns(d, t)
    assert _kinds(ops) == [
        ("MATCH", (0,), (0,)),
        ("MATCH", (1,), (1,)),
        ("MATCH", (2,), (2,)),
    ]


def test_a_real_move_with_no_detection_is_reported_missed():
    d = [frozenset({(0, 0)})]
    t = [frozenset({(0, 0)}), frozenset({(1, 1), (1, 2)})]
    ops = align_turns(d, t)
    assert _kinds(ops) == [("MATCH", (0,), (0,)), ("MISSED", (), (1,))]


def test_a_detection_with_no_real_move_is_reported_spurious():
    d = [frozenset({(0, 0)}), frozenset({(5, 5)})]
    t = [frozenset({(0, 0)})]
    ops = align_turns(d, t)
    assert _kinds(ops) == [("MATCH", (0,), (0,)), ("SPURIOUS", (1,), ())]


def test_one_detection_spanning_two_real_moves_is_a_merge():
    d = [frozenset({(0, 0), (0, 1), (0, 2), (0, 3)})]
    t = [frozenset({(0, 0), (0, 1)}), frozenset({(0, 2), (0, 3)})]
    ops = align_turns(d, t)
    assert _kinds(ops) == [("MERGE", (0,), (0, 1))]


def test_one_real_move_split_across_two_detections_is_a_split():
    # a truncated commit, then the dropped cell committing separately.
    d = [frozenset({(0, 0), (0, 1), (0, 2)}), frozenset({(0, 3)})]
    t = [frozenset({(0, 0), (0, 1), (0, 2), (0, 3)})]
    ops = align_turns(d, t)
    assert _kinds(ops) == [("SPLIT", (0, 1), (0,))]


def test_low_jaccard_pair_is_never_forced_to_match():
    d = [frozenset({(10, 10)})]
    t = [frozenset({(0, 0)})]
    ops = align_turns(d, t)
    assert _kinds(ops) == [("MISSED", (), (0,)), ("SPURIOUS", (0,), ())]


def test_merge_requires_every_member_to_actually_overlap_the_detection():
    # truth[1] shares nothing at all with the detection -- it must be
    # reported as a separate missed move, not folded into a merge just
    # because the union's Jaccard looks acceptable.
    d = [frozenset({(0, 0)})]
    t = [frozenset({(0, 0)}), frozenset({(9, 9), (9, 8)})]
    ops = align_turns(d, t)
    assert _kinds(ops) == [("MATCH", (0,), (0,)), ("MISSED", (), (1,))]


def test_split_requires_every_member_to_actually_overlap_the_truth_turn():
    d = [frozenset({(0, 0)}), frozenset({(9, 9)})]
    t = [frozenset({(0, 0)})]
    ops = align_turns(d, t)
    assert _kinds(ops) == [("MATCH", (0,), (0,)), ("SPURIOUS", (1,), ())]


def test_empty_detected_sequence_reports_everything_missed():
    t = [frozenset({(0, 0)}), frozenset({(1, 1)})]
    ops = align_turns([], t)
    assert _kinds(ops) == [("MISSED", (), (0,)), ("MISSED", (), (1,))]


def test_empty_truth_sequence_reports_everything_spurious():
    d = [frozenset({(0, 0)}), frozenset({(1, 1)})]
    ops = align_turns(d, [])
    assert _kinds(ops) == [("SPURIOUS", (0,), ()), ("SPURIOUS", (1,), ())]
