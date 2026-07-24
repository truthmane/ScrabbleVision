"""Exact tile-pool (bag) tracking.

Because both the board and every player's rack are visible to the camera
system, the bag's contents can be *derived* exactly -- tile-for-tile, not
just a count -- rather than tracked incrementally:

    bag = standard_distribution - tiles_on_board - tiles_on_all_racks

This is deliberately stateless: recomputing from the current authoritative
board/rack state each turn means an operator correction anywhere (a misread
letter, a reclassified blank) automatically fixes the bag figure on the next
recomputation, with no separate reconciliation step that could itself drift
out of sync.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from autoscorer.gamelogic.board import STANDARD_DISTRIBUTION, TOTAL_TILE_COUNT, BoardState, Tile


class PoolInvariantViolation(ValueError):
    """Raised when board+rack tiles claim more of some letter (or blank)
    than exist in the standard set -- this always indicates a misread
    somewhere upstream (CV or operator entry), never a legal game state.
    """


@dataclass(frozen=True)
class PoolState:
    bag: Counter
    rack_counts: list  # Counter per rack, in the same order as the input racks

    @property
    def bag_count(self) -> int:
        return sum(self.bag.values())


def _tile_counts(tiles) -> Counter:
    counts: Counter = Counter()
    for tile in tiles:
        counts[tile.pool_key] += 1
    return counts


def compute_pool_state(board: BoardState, racks: Sequence[Sequence[Tile]]) -> PoolState:
    """Derive bag contents from the current board and rack states.

    Raises PoolInvariantViolation if the observed tiles exceed the standard
    supply for any letter/blank -- callers (the state machine / operator UI)
    should surface this as a "please review this turn's reads" signal rather
    than letting it silently corrupt downstream state.
    """
    board_counts = _tile_counts(board.all_tiles())
    rack_counts_list = [_tile_counts(rack) for rack in racks]

    used = Counter(board_counts)
    for rc in rack_counts_list:
        used.update(rc)

    for key, count in used.items():
        supply = STANDARD_DISTRIBUTION.get(key, 0)
        if count > supply:
            raise PoolInvariantViolation(
                f"observed {count} of {key!r} across board+racks, but the standard set only has {supply}"
            )

    bag = Counter(STANDARD_DISTRIBUTION)
    bag.subtract(used)
    bag = Counter({k: v for k, v in bag.items() if v > 0})

    return PoolState(bag=bag, rack_counts=rack_counts_list)


def check_total_invariant(pool_state: PoolState, board: BoardState) -> bool:
    """`bag + all racks + board == 100` should always hold. Returns False
    (rather than raising) since this is a coarser, second-line sanity check
    meant to be surfaced as an operator warning, not a hard failure."""
    total = pool_state.bag_count + sum(sum(rc.values()) for rc in pool_state.rack_counts)
    total += sum(1 for _ in board.all_tiles())
    return total == TOTAL_TILE_COUNT
