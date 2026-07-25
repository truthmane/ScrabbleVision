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

from typing import List, Optional, Sequence

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
