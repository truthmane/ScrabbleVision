"""Enumerates legal candidate placements from a set of confirmed new
cells -- the piece that replaces "pick the first connected cluster,
commit or fail forever" with a real search over what could actually be
played.

Two real, measured bugs motivated this module rather than a guess:

1. `_cluster_cells` groups new cells by connectivity (including bridging
   through existing occupied tiles, so a hook play like "ARB.RIZE" -- new
   cells on either side of an existing letter -- clusters as one group).
   But connectivity is not collinearity: an L-shaped or staircase group of
   adjacent cells clusters as ONE connected group even though no straight
   line could ever cover it, and `validate_placement` can only ever reject
   it with `'new tiles must lie in a single row or column'`. A real
   full-game run hit exactly this: **163 consecutive settled observations,
   every single one rejected with that identical reason**, because the
   watcher had no other candidate to try once it picked that cluster.
2. Once `board_before` has accumulated more than one real turn's worth of
   unaccounted-for tiles (which happens the moment ANY turn fails to
   commit -- including via the separate blank-detection jam), a single
   connected-through-occupied-cells cluster can span cells from multiple,
   entirely unrelated real plays. Grouping by connectivity alone cannot
   tell those apart.

The fix here is to make every candidate this module returns collinear
and contiguous-through-the-board **by construction**, so those two
`validate_placement` rejection reasons become structurally unreachable.
Only genuine game rules (connectivity to existing tiles, first-move-
covers-center) can still reject a candidate this module produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Set

from autoscorer.gamelogic.board import BoardState, Coord, Tile
from autoscorer.gamelogic.movedetect.word_resolver import Axis, run_through


def _cluster_cells(cells: FrozenSet[Coord], board_before: BoardState) -> List[FrozenSet[Coord]]:
    """Groups new-cell coordinates into clusters connected through
    contiguous occupied cells (new or already on `board_before`) -- the
    same notion of "connected" `word_resolver.run_through`/
    `validate_placement` use, computed before any of these cells have
    actually been placed. A cluster is a raw grouping signal for
    `enumerate_candidate_placements` below, not itself a claim that the
    cluster is a legal (collinear) placement -- see this module's
    docstring for why that distinction matters.
    """
    remaining = set(cells)
    clusters: List[FrozenSet[Coord]] = []
    while remaining:
        seed = next(iter(remaining))
        cluster = {seed}
        frontier = [seed]
        remaining.discard(seed)
        while frontier:
            r, c = frontier.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                # Walk through a run of already-occupied cells (old tiles
                # this play hooks through) to find the next new cell in
                # this direction, if any.
                while (nr, nc) not in cells and board_before.get((nr, nc)) is not None:
                    nr, nc = nr + dr, nc + dc
                if (nr, nc) in remaining:
                    remaining.discard((nr, nc))
                    cluster.add((nr, nc))
                    frontier.append((nr, nc))
        clusters.append(frozenset(cluster))
    return clusters


@dataclass(frozen=True)
class CandidatePlacement:
    cells: FrozenSet[Coord]
    cluster_max_size: int
    """Size of the largest candidate drawn from the same raw cluster as
    this one. A candidate smaller than its cluster's max is a strict
    subset of some larger alternative reading of the same raw data (a
    truncation) -- the commit loop in `game_watcher.py` uses this to
    decide whether a candidate is safe to auto-publish."""
    used_soft_cells: bool = False
    """True if this candidate was extended with a cell that only had
    partial (not unanimous) occupancy support across the settled window
    -- see the `soft_cells` parameter below. Always `needs_operator=True`
    when true; never inferred from confidence alone."""


def _runs_along_axis(cluster: FrozenSet[Coord], hypothetical: BoardState, axis: Axis) -> List[FrozenSet[Coord]]:
    """Every maximal contiguous-through-the-board run along `axis` that
    can be built from `cluster`'s own cells, allowing the run to bridge
    through cells already on `board_before` (already reflected in
    `hypothetical`, which has every cluster cell provisionally placed).
    A cluster with more than one disjoint run in the same row/column
    (separated by a genuine gap, not a bridge) yields one entry per run.
    """
    lines: Set[int] = {r for r, _ in cluster} if axis == "row" else {c for _, c in cluster}
    seen: Set[Coord] = set()
    runs: List[FrozenSet[Coord]] = []
    for line in lines:
        line_cells = sorted(coord for coord in cluster if (coord[0] == line if axis == "row" else coord[1] == line))
        for coord in line_cells:
            if coord in seen:
                continue
            full_run = run_through(hypothetical, coord[0], coord[1], axis)
            run_new_cells = frozenset(rc for rc in full_run if rc in cluster)
            seen |= run_new_cells
            if run_new_cells:
                runs.append(run_new_cells)
    return runs


def _soft_extended_candidates(
    hard_candidates: List[CandidatePlacement], soft_cells: FrozenSet[Coord], board_before: BoardState,
) -> List[CandidatePlacement]:
    """Extends each multi-cell hard candidate by at most one adjacent
    in-line soft cell at either end -- a single new cell has no
    established direction to extend along, so singletons are left alone.
    Additive: these are extra candidates alongside the hard ones, never a
    replacement for them."""
    extended: List[CandidatePlacement] = []
    for candidate in hard_candidates:
        cells = sorted(candidate.cells)
        if len(cells) < 2:
            continue
        rows = {r for r, _ in cells}
        axis: Axis = "row" if len(rows) == 1 else "col"
        first, last = cells[0], cells[-1]
        if axis == "row":
            before, after = (first[0], first[1] - 1), (last[0], last[1] + 1)
        else:
            before, after = (first[0] - 1, first[1]), (last[0] + 1, last[1])
        for extra in (before, after):
            if extra in soft_cells and board_before.is_empty(extra):
                new_cells = frozenset(candidate.cells | {extra})
                extended.append(CandidatePlacement(
                    cells=new_cells,
                    cluster_max_size=max(candidate.cluster_max_size, len(new_cells)),
                    used_soft_cells=True,
                ))
    return extended


def enumerate_candidate_placements(
    confirmed_cells: FrozenSet[Coord],
    board_before: BoardState,
    soft_cells: FrozenSet[Coord] = frozenset(),
) -> List[CandidatePlacement]:
    """Every collinear, contiguous-through-the-board candidate placement
    that could be built from `confirmed_cells` -- grouped first by
    connectivity (`_cluster_cells`), then split into maximal per-row and
    per-column runs within each group. Assumes every cell in
    `confirmed_cells` is genuinely empty in `board_before` (the caller's
    contract, same as `_cluster_cells` already assumed).

    Deliberately `O(|confirmed_cells|)`: at most two runs per cell (one
    row, one column), never all subsets -- a real placement's cells all
    belong to the same one or two runs regardless of how large the
    overall confirmed set has grown.
    """
    if not confirmed_cells:
        return []

    candidates: List[CandidatePlacement] = []
    for cluster in _cluster_cells(confirmed_cells, board_before):
        if not cluster:
            continue
        hypothetical = board_before.with_placements(
            {coord: Tile(letter="A", is_blank=False) for coord in cluster}
        )
        raw_runs = _runs_along_axis(cluster, hypothetical, "row") + _runs_along_axis(cluster, hypothetical, "col")

        unique_runs: List[FrozenSet[Coord]] = []
        seen_sets: Set[FrozenSet[Coord]] = set()
        for run in raw_runs:
            if run not in seen_sets:
                seen_sets.add(run)
                unique_runs.append(run)

        cluster_max_size = max((len(run) for run in unique_runs), default=0)
        candidates.extend(
            CandidatePlacement(cells=run, cluster_max_size=cluster_max_size) for run in unique_runs
        )

    if soft_cells:
        candidates.extend(_soft_extended_candidates(candidates, soft_cells, board_before))

    return candidates
