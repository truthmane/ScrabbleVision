"""Aligns a sequence of detected turns against a sequence of real ("truth")
turns, by their cell sets.

This is not a zip and not a greedy nearest-match: detection can skip a real
move entirely (a stall), commit a blob spanning more than one real move
(merge), or split one real move across more than one commit (a truncated
commit followed by the dropped cell committing separately, or vice versa).
A wrong pairing here silently invalidates every metric downstream, so this
module does the alignment as a monotone sequence-alignment problem
(Needleman-Wunsch) rather than by hand-rolled heuristics -- both sequences
are temporally ordered and tiles are only ever added, so monotonicity is a
safe assumption; it isn't a safe assumption that nothing skips, merges, or
splits.

Pure logic: no cv2, no torch, no board/game imports beyond `Coord`'s type
alias -- must stay importable and testable anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Tuple

from autoscorer.gamelogic.board import Coord

_INF = float("inf")


def jaccard(a: FrozenSet[Coord], b: FrozenSet[Coord]) -> float:
    """|a n b| / |a u b|, defined as 1.0 for two empty sets (vacuously
    identical) rather than raising a ZeroDivisionError."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


@dataclass(frozen=True)
class AlignmentOp:
    """One edit in the alignment. `detected_indices`/`truth_indices` are
    0-indexed positions into the sequences passed to `align_turns`.
    `similarity` is the Jaccard score of the (possibly unioned) cell sets
    involved -- 0.0 for a pure skip (MISSED/SPURIOUS), since there is
    nothing on the other side to compare against."""
    kind: str  # "MATCH" | "MERGE" | "SPLIT" | "MISSED" | "SPURIOUS"
    detected_indices: Tuple[int, ...]
    truth_indices: Tuple[int, ...]
    similarity: float


def _union(sets: Sequence[FrozenSet[Coord]]) -> FrozenSet[Coord]:
    result: FrozenSet[Coord] = frozenset()
    for s in sets:
        result = result | s
    return result


def align_turns(
    detected: Sequence[FrozenSet[Coord]],
    truth: Sequence[FrozenSet[Coord]],
    max_merge: int = 3,
    max_split: int = 3,
    skip_cost: float = 1.0,
    merge_penalty: float = 0.25,
    split_penalty: float = 0.25,
    min_jaccard: float = 0.10,
) -> List[AlignmentOp]:
    """Monotone alignment of `detected` cell-set sequence against `truth`
    cell-set sequence. Edit operations, each scored by `1 - jaccard(...)`
    plus a fixed penalty where applicable:

    - MATCH:    one detected turn <-> one truth turn.
    - MERGE:    one detected turn <-> `w` (2..max_merge) consecutive truth
                turns unioned together (a blob commit spanning >1 real move).
    - SPLIT:    `w` (2..max_split) consecutive detected turns <-> one truth
                turn (a truncated commit, then the rest committing later).
    - MISSED:   a truth turn with no corresponding detection at all.
    - SPURIOUS: a detected turn with no corresponding real move at all.

    `min_jaccard` is a hard gate: any MATCH/MERGE/SPLIT candidate scoring
    below it is never considered, regardless of cost -- otherwise two
    unrelated turns can get paired just to dodge two skip costs. Complexity
    is O(N*M*(max_merge+max_split)), fine for N, M in the tens.
    """
    n, m = len(detected), len(truth)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: List[List[Optional[Tuple[str, int]]]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost[i][0] = cost[i - 1][0] + skip_cost
        back[i][0] = ("SPURIOUS", 1)
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + skip_cost
        back[0][j] = ("MISSED", 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_cost = _INF
            best_back: Optional[Tuple[str, int]] = None

            sim = jaccard(detected[i - 1], truth[j - 1])
            if sim >= min_jaccard:
                c = cost[i - 1][j - 1] + (1 - sim)
                if c < best_cost:
                    best_cost, best_back = c, ("MATCH", 1)

            c = cost[i - 1][j] + skip_cost
            if c < best_cost:
                best_cost, best_back = c, ("SPURIOUS", 1)

            c = cost[i][j - 1] + skip_cost
            if c < best_cost:
                best_cost, best_back = c, ("MISSED", 1)

            for w in range(2, max_merge + 1):
                if j - w < 0:
                    break
                truth_members = truth[j - w:j]
                # Every real move folded into this merge must itself
                # genuinely overlap the detected blob -- otherwise a
                # union's Jaccard can look deceptively high purely
                # because ONE member is a full match, while another
                # member shares nothing with the detection at all (that
                # second member is just a separate missed move, not part
                # of this merge).
                if any(not (detected[i - 1] & tm) for tm in truth_members):
                    continue
                sim = jaccard(detected[i - 1], _union(truth_members))
                if sim < min_jaccard:
                    continue
                c = cost[i - 1][j - w] + (1 - sim) + merge_penalty
                if c < best_cost:
                    best_cost, best_back = c, ("MERGE", w)

            for w in range(2, max_split + 1):
                if i - w < 0:
                    break
                det_members = detected[i - w:i]
                if any(not (dm & truth[j - 1]) for dm in det_members):
                    continue
                sim = jaccard(_union(det_members), truth[j - 1])
                if sim < min_jaccard:
                    continue
                c = cost[i - w][j - 1] + (1 - sim) + split_penalty
                if c < best_cost:
                    best_cost, best_back = c, ("SPLIT", w)

            cost[i][j] = best_cost
            back[i][j] = best_back

    ops: List[AlignmentOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        kind, w = back[i][j]  # type: ignore[misc]
        if kind == "MATCH":
            ops.append(AlignmentOp("MATCH", (i - 1,), (j - 1,), jaccard(detected[i - 1], truth[j - 1])))
            i, j = i - 1, j - 1
        elif kind == "SPURIOUS":
            ops.append(AlignmentOp("SPURIOUS", (i - 1,), (), 0.0))
            i -= 1
        elif kind == "MISSED":
            ops.append(AlignmentOp("MISSED", (), (j - 1,), 0.0))
            j -= 1
        elif kind == "MERGE":
            truth_idx = tuple(range(j - w, j))
            truth_union = _union(truth[j - w:j])
            ops.append(AlignmentOp("MERGE", (i - 1,), truth_idx, jaccard(detected[i - 1], truth_union)))
            i, j = i - 1, j - w
        else:  # SPLIT
            det_idx = tuple(range(i - w, i))
            det_union = _union(detected[i - w:i])
            ops.append(AlignmentOp("SPLIT", det_idx, (j - 1,), jaccard(det_union, truth[j - 1])))
            i, j = i - w, j - 1

    ops.reverse()
    return ops
