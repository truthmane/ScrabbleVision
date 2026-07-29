"""Detects which player's on-screen chess clock is currently ticking --
an independent, authoritative turn-boundary signal every real broadcast
this project has touched already carries, orthogonal to board-pixel
stillness.

**Why this exists.** `GameWatcher`'s move detection previously had exactly
one signal for "is this move finished": has the board stopped changing for
long enough. That is fundamentally ambiguous whenever a player places a
multi-tile word one letter at a time with real thinking pauses between
tiles -- a long enough pause between two tiles looks pixel-for-pixel
identical to "the move is over," and no amount of tuning the stillness
threshold or the quarantine/healing machinery fixes that, because the
board camera alone cannot see the one thing that actually marks a turn's
end: the player hitting their clock. Found live-testing the retrained
checkpoint against two different real games (WESPA "DJIN" and "HUIA"):
both fragmented into multiple partial-word candidates an operator had to
manually reject before the real, complete word ever got proposed --
board-stillness timing, not a classifier or logic defect.

**The signal.** This broadcast's overlay renders the active player's
remaining-time box with a solid red background and the inactive player's
box white/black -- confirmed against real frames from two different games
in the same broadcast series (WESPA "Games 1-7"): the same two fixed
pixel regions read `R - max(G, B)` around +50 to +70 for whichever side is
currently on the clock, and within a couple of points of zero for the
other -- a huge, trivially-separable margin, no OCR or ML needed. A color
switch between the two regions is the authoritative "a turn just ended"
event: a player doesn't stop their clock until they're genuinely done
placing tiles, so this signal, unlike stillness, is never fooled by a
mid-word thinking pause -- if anything it lags slightly behind the last
tile physically landing (the player has to reach over and press it),
which just means it is a conservative, not an eager, signal.

**Scope.** This module only answers "which side (if either) currently
reads as active" for one frame -- it has no memory of previous frames and
no opinion about *when* a switch should unblock anything. `GameWatcher`
owns tracking a side-to-side transition and deciding what to do with it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Box = Tuple[int, int, int, int]
"""(x1, y1, x2, y2) pixel bounds in the RAW (unrectified) camera frame --
the clock overlay lives outside the board's homography entirely, so these
are plain image-array slice bounds, never passed through
`BoardCalibration.rectify`."""

DEFAULT_RED_DOMINANCE_THRESHOLD = 20.0
"""Measured against real frames from two different games: genuinely
active (red) boxes scored +48 to +68 on `R - max(G, B)`; genuinely
inactive (white/black) boxes scored between -1.5 and +0 -- 20.0 sits
with wide margin on both sides of that real gap, not tuned to a knife
edge."""


@dataclass(frozen=True)
class ClockRegions:
    """Both players' clock-box pixel bounds for one venue. `left`/`right`
    are screen-position labels only (this venue always renders one player
    on each side of the board) -- `GameWatcher` never needs to know which
    named player is which side, only whether the active side changed."""

    left: Box
    right: Box


def _mean_bgr(crop: np.ndarray) -> Tuple[float, float, float]:
    b = float(crop[..., 0].mean())
    g = float(crop[..., 1].mean())
    r = float(crop[..., 2].mean())
    return b, g, r


def _is_red(crop: np.ndarray, threshold: float) -> bool:
    if crop.size == 0:
        return False
    b, g, r = _mean_bgr(crop)
    return (r - max(g, b)) > threshold


def detect_active_side(
    frame: np.ndarray,
    regions: ClockRegions,
    red_dominance_threshold: float = DEFAULT_RED_DOMINANCE_THRESHOLD,
) -> Optional[str]:
    """Returns `"left"` or `"right"` -- whichever clock box currently
    reads as the active-player red highlight -- or `None` if neither does
    (a genuine steady state this module has no signal for, e.g. a graphic
    transition) or, implausibly, both do (a transition frame mid-switch).
    Callers should treat `None` as "no new information this frame," never
    as an error or as "no one is active."

    `frame` is expected in OpenCV's BGR channel order, matching every
    other raw-frame consumer in this pipeline (`VideoFrameSource`,
    `BoardCalibration.rectify`).
    """
    lx1, ly1, lx2, ly2 = regions.left
    rx1, ry1, rx2, ry2 = regions.right
    left_crop = frame[ly1:ly2, lx1:lx2]
    right_crop = frame[ry1:ry2, rx1:rx2]
    left_red = _is_red(left_crop, red_dominance_threshold)
    right_red = _is_red(right_crop, red_dominance_threshold)
    if left_red and not right_red:
        return "left"
    if right_red and not left_red:
        return "right"
    return None
