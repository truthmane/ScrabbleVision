"""Board model: the standard 15x15 Scrabble board, tile values, and the
standard English tile distribution. Pure data + a minimal BoardState
container -- no CV, no I/O.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

BOARD_SIZE = 15
CENTER = (7, 7)

Coord = Tuple[int, int]

BLANK = "_"  # sentinel used in distributions/pool bookkeeping for a blank tile


@dataclass(frozen=True)
class Tile:
    """A single physical tile.

    For a non-blank tile, `letter` is always set. For a blank tile,
    `letter` is `None` while it sits unplayed in a rack or bag, and is set
    to whatever letter it was *played as* once placed on the board; either
    way `is_blank` stays True so scoring (worth 0) and pool bookkeeping
    (counted against the blank supply, never the letter's supply) are
    unaffected by which letter it represents.
    """
    letter: Optional[str]
    is_blank: bool = False

    def __post_init__(self) -> None:
        if self.letter is None:
            if not self.is_blank:
                raise ValueError("only a blank tile may have letter=None (unplayed)")
            return
        if len(self.letter) != 1 or not self.letter.isalpha():
            raise ValueError(f"Tile letter must be a single alphabetic character, got {self.letter!r}")
        object.__setattr__(self, "letter", self.letter.upper())

    @property
    def pool_key(self) -> str:
        """Which bucket this tile counts against in the standard distribution."""
        return BLANK if self.is_blank else self.letter


LETTER_VALUES: Dict[str, int] = {
    "A": 1, "B": 3, "C": 3, "D": 2, "E": 1, "F": 4, "G": 2, "H": 4, "I": 1,
    "J": 8, "K": 5, "L": 1, "M": 3, "N": 1, "O": 1, "P": 3, "Q": 10, "R": 1,
    "S": 1, "T": 1, "U": 1, "V": 4, "W": 4, "X": 8, "Y": 4, "Z": 10,
}

# Standard English Scrabble set: 100 tiles total, including 2 blanks.
STANDARD_DISTRIBUTION: Counter = Counter({
    "A": 9, "B": 2, "C": 2, "D": 4, "E": 12, "F": 2, "G": 3, "H": 2, "I": 9,
    "J": 1, "K": 1, "L": 4, "M": 2, "N": 6, "O": 8, "P": 2, "Q": 1, "R": 6,
    "S": 4, "T": 6, "U": 4, "V": 2, "W": 2, "X": 1, "Y": 2, "Z": 1,
    BLANK: 2,
})
assert sum(STANDARD_DISTRIBUTION.values()) == 100

TOTAL_TILE_COUNT = sum(STANDARD_DISTRIBUTION.values())

# Premium square layout for the standard 15x15 board, one 2-char code per
# cell (space-separated to avoid off-by-one errors versus coordinate lists).
# '..' = no premium. Symmetric in all 4 quadrants; center (7,7) is the
# double-word "star" square.
_ROW_STRINGS = [
    "3W .. .. 2L .. .. .. 3W .. .. .. 2L .. .. 3W",
    ".. 2W .. .. .. 3L .. .. .. 3L .. .. .. 2W ..",
    ".. .. 2W .. .. .. 2L .. 2L .. .. .. 2W .. ..",
    "2L .. .. 2W .. .. .. 2L .. .. .. 2W .. .. 2L",
    ".. .. .. .. 2W .. .. .. .. .. 2W .. .. .. ..",
    ".. 3L .. .. .. 3L .. .. .. 3L .. .. .. 3L ..",
    ".. .. 2L .. .. .. 2L .. 2L .. .. .. 2L .. ..",
    "3W .. .. 2L .. .. .. 2W .. .. .. 2L .. .. 3W",
    ".. .. 2L .. .. .. 2L .. 2L .. .. .. 2L .. ..",
    ".. 3L .. .. .. 3L .. .. .. 3L .. .. .. 3L ..",
    ".. .. .. .. 2W .. .. .. .. .. 2W .. .. .. ..",
    "2L .. .. 2W .. .. .. 2L .. .. .. 2W .. .. 2L",
    ".. .. 2W .. .. .. 2L .. 2L .. .. .. 2W .. ..",
    ".. 2W .. .. .. 3L .. .. .. 3L .. .. .. 2W ..",
    "3W .. .. 2L .. .. .. 3W .. .. .. 2L .. .. 3W",
]


def _build_premium_squares() -> Dict[Coord, str]:
    squares: Dict[Coord, str] = {}
    for r, row in enumerate(_ROW_STRINGS):
        codes = row.split(" ")
        assert len(codes) == BOARD_SIZE, f"row {r} has {len(codes)} cells, expected {BOARD_SIZE}"
        for c, code in enumerate(codes):
            if code != "..":
                squares[(r, c)] = code
    return squares


PREMIUM_SQUARES: Dict[Coord, str] = _build_premium_squares()

# Sanity-check the layout matches the well-known official counts.
_counts = Counter(PREMIUM_SQUARES.values())
assert _counts["3W"] == 8, _counts
assert _counts["3L"] == 12, _counts
assert _counts["2L"] == 24, _counts
# 16 diagonal double-word squares + the center star = 17.
assert _counts["2W"] == 17, _counts
assert PREMIUM_SQUARES[CENTER] == "2W"


class BoardState:
    """Mutable 15x15 grid of Optional[Tile], keyed by (row, col)."""

    def __init__(self, cells: Optional[Dict[Coord, Tile]] = None) -> None:
        self._cells: Dict[Coord, Tile] = dict(cells) if cells else {}

    @staticmethod
    def _check_bounds(coord: Coord) -> None:
        r, c = coord
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            raise ValueError(f"Coordinate {coord} is outside the 15x15 board")

    def get(self, coord: Coord) -> Optional[Tile]:
        return self._cells.get(coord)

    def set(self, coord: Coord, tile: Tile) -> None:
        self._check_bounds(coord)
        self._cells[coord] = tile

    def is_empty(self, coord: Coord) -> bool:
        return self.get(coord) is None

    def with_placements(self, placements: Dict[Coord, Tile]) -> "BoardState":
        """Return a new BoardState with the given cells filled in, leaving
        this instance untouched. Raises if any target cell is already occupied
        or if a tile has no assigned letter (an unplayed blank).
        """
        for coord, tile in placements.items():
            self._check_bounds(coord)
            if not self.is_empty(coord):
                raise ValueError(f"Cell {coord} is already occupied; cannot place a tile there")
            if tile.letter is None:
                raise ValueError(f"Tile at {coord} is a blank with no letter assigned; a blank must be played as a specific letter")
        merged = dict(self._cells)
        merged.update(placements)
        return BoardState(merged)

    def without_cells(self, coords: Iterable[Coord]) -> "BoardState":
        """Return a new BoardState with the given cells cleared, leaving
        this instance untouched. The inverse of `with_placements`, used to
        undo a committed move."""
        remaining = dict(self._cells)
        for coord in coords:
            remaining.pop(coord, None)
        return BoardState(remaining)

    def occupied_cells(self) -> Iterable[Coord]:
        return self._cells.keys()

    def all_tiles(self) -> Iterable[Tile]:
        return self._cells.values()

    def is_blank_board(self) -> bool:
        return not self._cells

    def copy(self) -> "BoardState":
        return BoardState(dict(self._cells))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BoardState) and self._cells == other._cells

    def __repr__(self) -> str:
        return f"BoardState({len(self._cells)} tiles placed)"
