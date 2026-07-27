from autoscorer.perception.occupancy.adaptive import AdaptiveOccupancyTracker


def test_transition_to_occupied_is_immediate_not_delayed():
    tracker = AdaptiveOccupancyTracker(base_threshold=10.0)
    coord = (7, 7)

    # A handful of quiet, empty-looking observations first.
    for _ in range(5):
        assert tracker.update(coord, 2.0) is False

    # A real tile appears -- must be trusted on this very observation,
    # never delayed by hysteresis (only the empty->occupied direction is
    # eager; only occupied->empty is damped).
    assert tracker.update(coord, 50.0) is True


def test_hysteresis_requires_consecutive_observations_below_the_flip_band():
    tracker = AdaptiveOccupancyTracker(
        base_threshold=10.0, hysteresis_ratio=0.8, hysteresis_observations=2,
    )
    coord = (7, 7)
    tracker.update(coord, 2.0)  # establish "empty" baseline
    assert tracker.update(coord, 50.0) is True  # now occupied

    # Diff drops back under the raw threshold (10.0) but not under the
    # flip band (0.8 * 10.0 = 8.0) -- must NOT flip back yet.
    assert tracker.update(coord, 9.0) is True

    # Now genuinely below the flip band, but only once -- still occupied.
    assert tracker.update(coord, 3.0) is True

    # A second consecutive observation below the flip band -- now flips.
    assert tracker.update(coord, 3.0) is False


def test_hysteresis_resets_the_consecutive_counter_on_a_single_bounce_back_above_the_band():
    tracker = AdaptiveOccupancyTracker(
        base_threshold=10.0, hysteresis_ratio=0.8, hysteresis_observations=2,
    )
    coord = (7, 7)
    tracker.update(coord, 2.0)
    tracker.update(coord, 50.0)  # occupied

    assert tracker.update(coord, 3.0) is True  # 1st below-band observation
    # Bounces back above the flip band before a 2nd consecutive one --
    # the streak must reset, not carry over.
    assert tracker.update(coord, 9.0) is True
    assert tracker.update(coord, 3.0) is True  # this is only the 1st again
    assert tracker.update(coord, 3.0) is False  # now the 2nd consecutive -- flips


def test_adaptive_threshold_learns_a_higher_noise_floor_than_the_base_default():
    # This cell's own genuine background noise runs higher/noisier than
    # the global base_threshold would assume -- the adaptive EMA should
    # raise its effective threshold so ordinary noise here doesn't trip
    # it, exactly the "different cells, different noise floors" case the
    # real WESPA RAGBOLT/noise-overlap measurement motivated.
    tracker = AdaptiveOccupancyTracker(
        base_threshold=10.0, warmup_samples=5, std_multiplier=6.0, floor=1.0,
    )
    coord = (3, 3)
    # Feed enough "empty" samples (mean ~8, std ~1) to clear warm-up --
    # the learned threshold (mean + 6*std) ends up ~13-14, well above
    # base_threshold (10.0).
    noisy_empty_readings = [7.0, 9.0, 6.5, 9.5, 7.5, 8.5, 7.0, 9.0]
    for reading in noisy_empty_readings:
        tracker.update(coord, reading)

    # A reading of 11 would have tripped the base_threshold (10.0) as a
    # false positive, but should NOT trip the learned, noise-aware one.
    assert tracker.update(coord, 11.0) is False
    # A reading well above the learned threshold must still trip it.
    assert tracker.update(coord, 20.0) is True


def test_adaptive_threshold_uses_base_threshold_during_warmup():
    tracker = AdaptiveOccupancyTracker(base_threshold=10.0, warmup_samples=5)
    coord = (3, 3)
    # Only 2 samples so far -- still in warm-up, must fall back to the
    # base threshold rather than trusting a barely-established estimate.
    tracker.update(coord, 2.0)
    tracker.update(coord, 2.0)

    assert tracker.update(coord, 15.0) is True  # above base_threshold (10.0)


def test_floor_prevents_an_unrealistically_sensitive_threshold():
    # A cell that has read almost perfectly quiet (near-zero variance) so
    # far must not become so sensitive that ordinary future noise trips
    # it -- the floor caps how low the learned threshold can go. Uses two
    # separately-warmed-up trackers rather than chaining both readings
    # onto one, since a reading under the floor still updates the EMA
    # (the cell never became occupied), which would otherwise shift the
    # second assertion's own threshold.
    def _warmed_up_tracker():
        tracker = AdaptiveOccupancyTracker(
            base_threshold=10.0, warmup_samples=3, std_multiplier=6.0, floor=5.0,
        )
        coord = (3, 3)
        for _ in range(4):
            tracker.update(coord, 0.1)  # near-perfectly quiet
        assert tracker._threshold_for(tracker._state[coord]) == 5.0, "learned mean+6*std should be under the floor"
        return tracker, coord

    tracker_a, coord = _warmed_up_tracker()
    # Under the floor (5.0) -- must NOT trip occupancy.
    assert tracker_a.update(coord, 4.0) is False

    tracker_b, coord = _warmed_up_tracker()
    # Above the floor -- must trip occupancy.
    assert tracker_b.update(coord, 6.0) is True


def test_cells_are_tracked_independently():
    tracker = AdaptiveOccupancyTracker(base_threshold=10.0)
    tracker.update((0, 0), 2.0)
    tracker.update((0, 0), 50.0)  # (0,0) becomes occupied

    # A totally separate cell must be unaffected.
    assert tracker.update((1, 1), 2.0) is False


def test_reset_forgets_a_cells_state():
    tracker = AdaptiveOccupancyTracker(base_threshold=10.0)
    coord = (5, 5)
    tracker.update(coord, 2.0)
    tracker.update(coord, 50.0)
    assert tracker.update(coord, 50.0) is True

    tracker.reset(coord)

    # Back to a clean slate -- a quiet reading right after reset must not
    # be considered occupied.
    assert tracker.update(coord, 2.0) is False
