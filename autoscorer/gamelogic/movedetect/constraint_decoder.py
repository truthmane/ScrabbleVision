"""Corrects locally-plausible-but-globally-impossible classifier readings
using a constraint that's already fully built and tested: the standard
100-tile distribution. A third Q, or a fifth E beyond what's already
visible on the board and both racks, is not a possible game state -- so
when the classifier's top guess for some cell would require more of a
letter than remains unaccounted for, this falls back to the next
most-likely candidate that IS still possible, rather than trusting a
locally-confident but globally-impossible reading.

Pure logic on top of gamelogic.pool.bag_engine -- no new ML, no new data.
This is the "constraint decoding" piece of WS4 in
docs/classifier-accuracy-plan.md; the other piece (a dictionary-scored
beam over low-margin cells) needs a real word list wired into
gamelogic/dictionary/lookup.py and is left for when that exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from autoscorer.gamelogic.board import BLANK, STANDARD_DISTRIBUTION, BoardState, Coord, Tile
from autoscorer.gamelogic.pool.bag_engine import _tile_counts

# The classifier's class label for a blank tile ("BLANK", see
# training/synth_render/tile_renderer.py's TILE_CLASSES) is spelled
# differently from board.py's pool-accounting sentinel (BLANK = "_") --
# this bridges the two vocabularies at the one place they need to meet.
CLASSIFIER_BLANK_LABEL = "BLANK"


@dataclass(frozen=True)
class CellCandidates:
    coord: Coord
    candidates: List[Tuple[str, float]]  # (label, confidence), most likely first
    is_soft: bool = False
    """True if this cell's occupancy support was partial (not unanimous)
    across a temporal-voting window -- see
    `board_reader.read_new_cells_voted`'s HARD/SOFT tiers. Always False for
    a single-frame read (`read_new_cells`/`read_board`, no window to
    disagree across) or for a HARD cell (every frame in the window agreed).
    `GameWatcher` uses this to keep soft cells out of ordinary confirmation/
    clustering and route them through the soft-extension path in
    `placement_search.py` instead."""


def _pool_key(label: str) -> str:
    return BLANK if label == CLASSIFIER_BLANK_LABEL else label


def remaining_supply(board: BoardState, racks: Sequence[Sequence[Tile]]) -> Dict[str, int]:
    """How many of each letter (or blank) are not yet visibly accounted for
    on the board or in any rack -- i.e. still somewhere in the bag, or
    among the very tiles about to be decoded for this move.
    """
    used = _tile_counts(board.all_tiles())
    for rack in racks:
        used.update(_tile_counts(rack))
    return {key: max(0, count - used.get(key, 0)) for key, count in STANDARD_DISTRIBUTION.items()}


def decode_feasible_reading(
    cell_candidates: Sequence[CellCandidates],
    board_before: BoardState,
    racks: Sequence[Sequence[Tile]],
) -> Dict[Coord, str]:
    """Assigns each cell its highest-confidence candidate label that the
    remaining tile supply can still support, most-confident cell first (so
    a genuinely scarce letter goes to whichever cell the classifier is
    surest about, not whichever happens to be processed first). Falls back
    to the raw top-1 label if none of a cell's candidates are feasible --
    that's a sign of a deeper misread for the operator to catch, not
    something to silently paper over by inventing a reading.
    """
    available = remaining_supply(board_before, racks)
    assignment: Dict[Coord, str] = {}

    ordered = sorted(cell_candidates, key=lambda cc: -cc.candidates[0][1])
    for cc in ordered:
        chosen = None
        for label, _confidence in cc.candidates:
            key = _pool_key(label)
            if available.get(key, 0) > 0:
                chosen = label
                available[key] -= 1
                break
        assignment[cc.coord] = chosen if chosen is not None else cc.candidates[0][0]

    return assignment
