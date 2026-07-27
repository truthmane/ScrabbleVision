"""Decides when the board is settled enough to trust a read at all --
the "stillness/occlusion gate" from the master architecture plan's
move-detection state machine (Phase 5). A camera watching a live board
sees hands moving tiles, players reaching across, and the moment of
placement itself; reading any of those frames as a stable final state is
how a half-completed placement or a hand blocking a cell gets misread as
a tile (or a tile misread as absent). Occupancy detection, classification,
and temporal voting are all only meaningful once this gate says the board
has actually stopped changing -- voting assumes its input frames all show
the same real state, and this is what decides that.

Pure classical CV (frame-to-frame diff), same philosophy as
occupancy/detector.py: cheap, debuggable, no ML needed for a problem
this mechanically simple. Per-venue calibration of the motion threshold
is expected, same as occupancy's thresholds -- lighting flicker and
camera sensor noise set a venue-specific noise floor.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Sequence

import cv2
import numpy as np

# Calibrated against one real venue's footage (the same NASPA 2026 finals
# broadcast used throughout docs/classifier-accuracy-plan.md's WS3):
# consecutive frames of a genuinely stable board scored 0.4-8 on this
# signal (JPEG/H.264 compression noise, not real motion), while a frame
# where hands entered the shot scored 37.8 -- comfortably separated, but
# still expect to retune per venue, same as occupancy/detector.py's
# thresholds. Not yet validated against a live (non-broadcast-compressed)
# camera feed, which should be *less* noisy than this, not more.
DEFAULT_MOTION_THRESHOLD = 10.0
DEFAULT_STILL_FRAME_COUNT = 5


def frame_motion_score(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Mean absolute pixel difference between two consecutive frames --
    near-zero for a genuinely static scene, large whenever anything (a
    hand, a tile, the lighting) moved between them."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(gray_a - gray_b).mean())


def is_settled(
    recent_frames: Sequence[np.ndarray],
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    required_still_frames: int = DEFAULT_STILL_FRAME_COUNT,
) -> bool:
    """True once the most recent `required_still_frames` frames show
    negligible motion between every consecutive pair -- i.e. nothing has
    moved for that whole window. Earlier motion outside the window
    (a hand that has since left, an old placement) doesn't matter; only
    the trailing window does, so this naturally recovers the instant a
    hand clears the board rather than waiting for some fixed cooldown.

    Returns False (never claims settled) if fewer than
    `required_still_frames` frames have been supplied yet -- a short
    history is exactly as untrustworthy as a moving one here.
    """
    if len(recent_frames) < required_still_frames:
        return False

    window = recent_frames[-required_still_frames:]
    return all(
        frame_motion_score(a, b) <= motion_threshold
        for a, b in zip(window, window[1:])
    )


def stable_window(
    recent_frames: Sequence[np.ndarray],
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
    required_still_frames: int = DEFAULT_STILL_FRAME_COUNT,
) -> Optional[List[np.ndarray]]:
    """The trailing settled window of frames, ready to hand to
    `board_reader.read_new_cells_voted` -- or `None` if the board isn't
    currently settled, in which case a caller should keep waiting rather
    than read anything."""
    if not is_settled(recent_frames, motion_threshold, required_still_frames):
        return None
    return list(recent_frames[-required_still_frames:])


class StillnessTracker:
    """Incremental equivalent of calling `is_settled`/`stable_window`
    against a hand-maintained buffer on every new frame.

    `is_settled`/`stable_window` recompute all `required_still_frames - 1`
    pairwise `frame_motion_score` calls from scratch every time they're
    called, even though a caller that appends one frame per call (every
    real caller -- see `GameWatcher`) already had all but one of those
    pairs computed on the *previous* call. That's fine at the small window
    this was originally tuned at, but `VenueProfile.
    effective_still_frame_count` now derives `required_still_frames` from
    `still_seconds * sample_fps`, so a venue calibrated at a low
    `sample_fps` and replayed at a higher one gets a much larger window
    (e.g. 50 frames instead of 5) -- and since per-call cost scales with
    window size too, that's quadratic in the frame stream's length,
    measured at ~100x slower on a real video eval.

    This caches the still/not-still verdict for each consecutive pair the
    first time it's computed, so `push` costs exactly one new
    `frame_motion_score` call regardless of window size.

    `motion_threshold` and `required_still_frames` are fixed for the
    tracker's lifetime -- matches how every real caller uses them (set
    once, never reassigned); make a new tracker rather than mutating one
    mid-stream.
    """

    def __init__(
        self,
        motion_threshold: float = DEFAULT_MOTION_THRESHOLD,
        required_still_frames: int = DEFAULT_STILL_FRAME_COUNT,
    ):
        self.motion_threshold = motion_threshold
        self.required_still_frames = required_still_frames
        self._frames: Deque[np.ndarray] = deque(maxlen=required_still_frames)
        self._pair_still: Deque[bool] = deque(maxlen=max(required_still_frames - 1, 0))

    def push(self, frame: np.ndarray) -> None:
        """Feed one new frame in, updating the cached pairwise verdicts."""
        if self._frames:
            still = frame_motion_score(self._frames[-1], frame) <= self.motion_threshold
            self._pair_still.append(still)
        self._frames.append(frame)

    def is_settled(self) -> bool:
        return (
            len(self._frames) == self.required_still_frames
            and len(self._pair_still) == self.required_still_frames - 1
            and all(self._pair_still)
        )

    def stable_window(self) -> Optional[List[np.ndarray]]:
        if not self.is_settled():
            return None
        return list(self._frames)

    @property
    def last_pair_still(self) -> Optional[bool]:
        """Whether the two most recently pushed frames were within
        `motion_threshold` of each other, or `None` if fewer than two
        frames have been pushed yet."""
        return self._pair_still[-1] if self._pair_still else None

    def __len__(self) -> int:
        return len(self._frames)
