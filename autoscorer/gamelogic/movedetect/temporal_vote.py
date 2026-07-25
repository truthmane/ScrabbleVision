"""Combines a cell's classifier readings across multiple frames of the
*same* stable board state into a single, more reliable reading -- the M2
lever from docs/classifier-accuracy-plan.md ("per-tile after temporal
voting over >=5 stable frames"). A single frame can catch a tile at a bad
angle, mid-glare, or blurred; if that frame's misread is a minority across
several looks at the same physical tile, it shouldn't survive.

Pure logic over (label, confidence) lists -- no CV, no camera concepts.
Mirrors the classic "mode over the last N frames" approach (jheidel's
scrabble-opencv, cited in the master plan's literature review), but keeps
the full averaged distribution rather than a bare majority count, which is
what lets a confidence threshold still mean something after voting.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def temporal_vote(per_frame_candidates: Sequence[Sequence[Tuple[str, float]]]) -> List[Tuple[str, float]]:
    """`per_frame_candidates` is one (label, confidence) list per frame,
    covering the same physical cell (typically a full per-class
    distribution, e.g. `TileClassifierModel.predict_topk(image, k=num_classes)`
    for each frame). Returns a single combined list in the same shape,
    most-likely label first, by averaging each label's confidence across
    frames -- a label only a minority of frames guessed still gets pulled
    down by the frames that disagreed, exactly the outvoting effect this
    exists for.

    Frames need not agree on which labels appear in their top-k; a label
    missing from a given frame's list contributes 0 for that frame (not
    "missing data"), so passing truncated top-k lists still behaves
    sensibly, just less precisely than passing a full distribution.
    """
    if not per_frame_candidates:
        raise ValueError("temporal_vote needs at least one frame's candidates")

    totals: Dict[str, float] = {}
    for frame_candidates in per_frame_candidates:
        for label, confidence in frame_candidates:
            totals[label] = totals.get(label, 0.0) + confidence

    n = len(per_frame_candidates)
    averaged = [(label, total / n) for label, total in totals.items()]
    averaged.sort(key=lambda pair: -pair[1])
    return averaged
