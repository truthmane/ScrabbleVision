from autoscorer.eval.gcg_truth import TruthTurn
from autoscorer.eval.metrics import DetectedTurn, StallInfo, build_report


def _truth(turn_number, player, cells, letters, blank_cells=frozenset(), score=0, cumulative=0, position="8G"):
    return TruthTurn(
        turn_number=turn_number, player=player, cells=frozenset(cells), letters=dict(letters),
        blank_cells=frozenset(blank_cells), position=position, turn_score=score, cumulative_score=cumulative,
    )


def _detected(frame, player, cells, letters, blank_cells=frozenset(), score=0, needs_operator=False):
    return DetectedTurn(
        frame_index=frame, player=player, cells=frozenset(cells), letters=dict(letters),
        blank_cells=frozenset(blank_cells), score=score, needs_operator=needs_operator,
    )


def test_clean_game_reports_no_divergence_and_perfect_metrics():
    truth = [
        _truth(1, "A", [(7, 6), (7, 7), (7, 8)], {(7, 6): "Q", (7, 7): "A", (7, 8): "T"}, score=24, cumulative=24),
        _truth(2, "B", [(8, 4), (8, 5)], {(8, 4): "H", (8, 5): "I"}, score=10, cumulative=10),
    ]
    detected = [
        _detected(10, "A", [(7, 6), (7, 7), (7, 8)], {(7, 6): "Q", (7, 7): "A", (7, 8): "T"}, score=24),
        _detected(20, "B", [(8, 4), (8, 5)], {(8, 4): "H", (8, 5): "I"}, score=10),
    ]
    report = build_report(detected, truth)

    assert report.real_plays == 2
    assert report.matched_1to1 == 2
    assert report.missed == 0
    assert report.spurious == 0
    assert report.first_divergence_index is None
    assert report.exact_score_matches == 2
    assert report.cell_f1_micro == 1.0
    assert report.letter_accuracy == 1.0
    assert report.final_cumulative_drift == {"A": 0, "B": 0}


def test_first_divergence_index_names_the_first_broken_truth_turn():
    truth = [
        _truth(1, "A", [(0, 0)], {(0, 0): "Q"}, score=5, cumulative=5),
        _truth(2, "B", [(0, 1)], {(0, 1): "I"}, score=3, cumulative=8),
        _truth(3, "A", [(0, 2)], {(0, 2): "T"}, score=2, cumulative=10),
    ]
    detected = [
        _detected(1, "A", [(0, 0)], {(0, 0): "Q"}, score=5),
        # turn 2's score is wrong -- this is the first divergence, even
        # though its cells are otherwise correct.
        _detected(2, "B", [(0, 1)], {(0, 1): "I"}, score=99),
        _detected(3, "A", [(0, 2)], {(0, 2): "T"}, score=2),
    ]
    report = build_report(detected, truth)
    assert report.first_divergence_index == 2


def test_a_missed_real_move_counts_as_divergence_and_reduces_matches():
    truth = [
        _truth(1, "A", [(0, 0)], {(0, 0): "Q"}, score=5, cumulative=5),
        _truth(2, "B", [(0, 1)], {(0, 1): "I"}, score=3, cumulative=8),
    ]
    detected = [_detected(1, "A", [(0, 0)], {(0, 0): "Q"}, score=5)]
    report = build_report(detected, truth)
    assert report.matched_1to1 == 1
    assert report.missed == 1
    assert report.first_divergence_index == 2


def test_letter_accuracy_and_blank_recovery_tracked_separately():
    truth = [_truth(1, "A", [(0, 0), (0, 1)], {(0, 0): "Q", (0, 1): "I"}, blank_cells=[(0, 1)], score=10, cumulative=10)]
    # (0,0) read correctly, (0,1) -- the blank -- decoded wrong.
    detected = [_detected(1, "A", [(0, 0), (0, 1)], {(0, 0): "Q", (0, 1): "X"}, score=10)]
    report = build_report(detected, truth)
    assert report.letter_total == 2
    assert report.letter_correct == 1
    assert report.blank_total == 1
    assert report.blank_recovered == 0


def test_stalls_are_carried_through_with_longest_stall_computed():
    stalls = [
        StallInfo(start_frame=10, length=5, reason="pool invariant violated", attempted_cells=frozenset({(0, 0)})),
        StallInfo(start_frame=100, length=163, reason="new tiles must lie in a single row or column",
                   attempted_cells=frozenset({(1, 1), (2, 1)})),
    ]
    report = build_report([], [], stalls=stalls)
    assert report.longest_stall == 163
    assert len(report.stalls) == 2


def test_operator_routed_fraction_computed_over_detected_turns():
    truth = [_truth(1, "A", [(0, 0)], {(0, 0): "Q"}, score=5, cumulative=5)]
    detected = [
        _detected(1, "A", [(0, 0)], {(0, 0): "Q"}, score=5, needs_operator=True),
        _detected(2, "A", [(5, 5)], {(5, 5): "Z"}, score=1, needs_operator=False),
    ]
    report = build_report(detected, truth)
    assert report.operator_routed_fraction == 0.5


def test_to_json_dict_round_trips_key_fields_and_formats_squares():
    stalls = [StallInfo(start_frame=1, length=2, reason="x", attempted_cells=frozenset({(7, 7)}))]
    report = build_report([], [], stalls=stalls)
    data = report.to_json_dict()
    assert data["longest_stall"] == 2
    assert data["stalls"][0]["attempted_squares"] == ["H8"]
