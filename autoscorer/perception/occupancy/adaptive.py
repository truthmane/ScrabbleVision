"""Per-cell hysteresis and adaptive statistical thresholding on top of the
raw `diff` occupancy signal (`occupancy/detector.py`'s `occupancy_scores`)
-- WS3 items 3-4 from `docs/classifier-accuracy-plan.md`.

Both address the same underlying problem from a different angle: a single
global `diff_threshold` can't simultaneously fit a thin-glyph tile sitting
right next to the noise floor (the real WESPA "RAGBOLT" case: diff 38.96
against a 38.0 threshold) and a cell whose own background noise happens to
run higher than another cell's -- these are different cells with
different noise floors, which one fixed number can't separate.

- **Hysteresis**: once a cell is considered occupied, its diff score must
  drop comfortably *below* the occupied threshold (not just cross back
  under it) for several consecutive observations before it's trusted as
  empty again -- directly attacks the same flicker that forced
  `GameWatcher`'s per-cell confirmation mechanism to exist in the first
  place, but at the raw-signal source rather than the higher-level
  confirmed-cells bookkeeping.
- **Adaptive threshold**: while a cell reads empty, an exponential moving
  average of its own `diff` score (mean and variance) tracks that cell's
  own noise floor; the effective threshold becomes
  `max(floor, mean + std_multiplier * std)` instead of one fixed global
  number, with a warm-up period (too few samples to trust the estimate
  yet) falling back to `base_threshold`, and a mandatory floor so a cell
  with an unusually quiet noise floor can't become too sensitive.

Deliberately a standalone, stateful class rather than baked into
`detector.py`'s stateless functions: both hysteresis and the adaptive EMA
are inherently cross-observation state that only a caller processing a
whole game (like `GameWatcher`) can own -- `detector.py` stays a pure,
one-shot-per-call module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from autoscorer.gamelogic.board import Coord
from autoscorer.perception.occupancy.detector import DEFAULT_DIFF_THRESHOLD

DEFAULT_HYSTERESIS_RATIO = 0.8
"""Once occupied, diff must drop below `hysteresis_ratio * threshold` --
not just under `threshold` itself -- before a cell is even eligible to
flip back to empty."""
DEFAULT_HYSTERESIS_OBSERVATIONS = 2
"""How many consecutive observations diff must stay below the flip-back
band before a cell actually flips back to empty."""
DEFAULT_EMA_ALPHA = 0.2
DEFAULT_WARMUP_SAMPLES = 5
"""Below this many empty-cell samples, the adaptive EMA estimate isn't
trusted yet -- `base_threshold` is used instead."""
DEFAULT_STD_MULTIPLIER = 6.0


@dataclass
class _CellState:
    occupied: bool = False
    consecutive_below_flip: int = 0
    ema_mean: Optional[float] = None
    ema_var: Optional[float] = None
    sample_count: int = 0


class AdaptiveOccupancyTracker:
    """Call `update(coord, diff_score)` once per settled observation for
    every not-yet-played cell (per `board_before.is_empty`), with that
    cell's current `diff` occupancy score (see `occupancy_scores`).
    Returns the debounced, adaptively-thresholded occupied decision for
    that cell this observation. Entirely per-cell and order-independent
    across cells within one observation; state only accumulates across
    successive calls for the SAME coord.
    """

    def __init__(
        self,
        base_threshold: float = DEFAULT_DIFF_THRESHOLD,
        hysteresis_ratio: float = DEFAULT_HYSTERESIS_RATIO,
        hysteresis_observations: int = DEFAULT_HYSTERESIS_OBSERVATIONS,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        warmup_samples: int = DEFAULT_WARMUP_SAMPLES,
        std_multiplier: float = DEFAULT_STD_MULTIPLIER,
        floor: Optional[float] = None,
    ) -> None:
        self.base_threshold = base_threshold
        self.hysteresis_ratio = hysteresis_ratio
        self.hysteresis_observations = hysteresis_observations
        self.ema_alpha = ema_alpha
        self.warmup_samples = warmup_samples
        self.std_multiplier = std_multiplier
        # A mandatory floor independent of the learned noise floor -- an
        # unusually quiet cell (near-zero observed variance so far) must
        # never become so sensitive that ordinary future noise trips it;
        # half of base_threshold is a conservative default that's still
        # well below where a real tile scores.
        self.floor = floor if floor is not None else base_threshold * 0.5
        self._state: Dict[Coord, _CellState] = {}

    def _threshold_for(self, state: _CellState) -> float:
        if state.sample_count < self.warmup_samples or state.ema_mean is None or state.ema_var is None:
            return self.base_threshold
        std = math.sqrt(max(state.ema_var, 0.0))
        return max(self.floor, state.ema_mean + self.std_multiplier * std)

    def update(self, coord: Coord, diff_score: float) -> bool:
        state = self._state.setdefault(coord, _CellState())
        threshold = self._threshold_for(state)

        if not state.occupied:
            # Only feed the "what does empty look like" EMA while the
            # cell isn't currently considered occupied -- folding an
            # occupied reading into the noise-floor estimate would poison
            # it for every future observation of this same cell.
            if state.ema_mean is None:
                state.ema_mean = diff_score
                state.ema_var = 0.0
            else:
                delta = diff_score - state.ema_mean
                incr = self.ema_alpha * delta
                state.ema_mean += incr
                state.ema_var = (1.0 - self.ema_alpha) * (state.ema_var + delta * incr)
            state.sample_count += 1

            if diff_score > threshold:
                # Eager, un-hysteresis'd transition into occupied -- a
                # genuinely appearing new tile must never be delayed by
                # this layer, only a transition BACK to empty is damped.
                state.occupied = True
                state.consecutive_below_flip = 0
        else:
            flip_band = threshold * self.hysteresis_ratio
            if diff_score < flip_band:
                state.consecutive_below_flip += 1
                if state.consecutive_below_flip >= self.hysteresis_observations:
                    state.occupied = False
                    state.consecutive_below_flip = 0
            else:
                state.consecutive_below_flip = 0

        return state.occupied

    def reset(self, coord: Coord) -> None:
        """Forgets a cell's tracked state entirely. Not required for
        correctness (a cell that commits as a real tile is never queried
        again, since `board_before.is_empty` already gates every future
        occupancy check upstream of this tracker) -- provided so a caller
        that wants to bound memory, or deliberately give a healed/
        quarantined cell a clean slate, can."""
        self._state.pop(coord, None)
