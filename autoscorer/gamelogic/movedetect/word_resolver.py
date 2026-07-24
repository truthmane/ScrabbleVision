"""Derives the word(s) formed by a set of newly-placed tiles.

Given the board *after* placement, a single tile-run lookup primitive
(`run_through`) is enough to build both the main word and every
perpendicular cross-word a placement forms -- including the edge case of a
single new tile that simultaneously completes a horizontal and a vertical
word. This is pure combinatorics over board coordinates; no CV or game
history is involved.
"""
from __future__ import annotations

from typing import List, Literal, Sequence

from autoscorer.gamelogic.board import BOARD_SIZE, BoardState, Coord

Axis = Literal["row", "col"]


def run_through(board: BoardState, row: int, col: int, axis: Axis) -> List[Coord]:
    """The maximal contiguous run of occupied cells along `axis` that
    includes (row, col). Assumes (row, col) itself is occupied in `board`.
    """
    if axis == "row":
        c = col
        while c - 1 >= 0 and not board.is_empty((row, c - 1)):
            c -= 1
        start = c
        c = col
        while c + 1 < BOARD_SIZE and not board.is_empty((row, c + 1)):
            c += 1
        end = c
        return [(row, cc) for cc in range(start, end + 1)]
    else:
        r = row
        while r - 1 >= 0 and not board.is_empty((r - 1, col)):
            r -= 1
        start = r
        r = row
        while r + 1 < BOARD_SIZE and not board.is_empty((r + 1, col)):
            r += 1
        end = r
        return [(r_, col) for r_ in range(start, end + 1)]


def words_formed(board_after: BoardState, new_cells: Sequence[Coord]) -> List[List[Coord]]:
    """All words formed by this placement: the main word plus any
    perpendicular cross-words, each as an ordered list of board coordinates.

    Assumes `new_cells` has already passed placement validation (single line,
    contiguous). Order of returned words is not significant to callers.
    """
    if not new_cells:
        return []

    new_list = list(new_cells)
    words: List[List[Coord]] = []

    if len(new_list) > 1:
        rows = {r for r, _ in new_list}
        axis: Axis = "row" if len(rows) == 1 else "col"
        cross_axis: Axis = "col" if axis == "row" else "row"

        anchor_r, anchor_c = new_list[0]
        words.append(run_through(board_after, anchor_r, anchor_c, axis))

        for (r, c) in new_list:
            cross = run_through(board_after, r, c, cross_axis)
            if len(cross) > 1:
                words.append(cross)
    else:
        r, c = new_list[0]
        horizontal = run_through(board_after, r, c, "row")
        vertical = run_through(board_after, r, c, "col")
        if len(horizontal) > 1:
            words.append(horizontal)
        if len(vertical) > 1:
            words.append(vertical)
        if not words:
            # Isolated single tile: only legal as the very first move
            # (a one-letter opening word covering the center square).
            words.append([(r, c)])

    return words


def word_text(board_after: BoardState, word_cells: Sequence[Coord]) -> str:
    """The letters spelled out by a resolved word, in board order."""
    letters = []
    for coord in word_cells:
        tile = board_after.get(coord)
        assert tile is not None, f"word_text called with unoccupied cell {coord}"
        letters.append(tile.letter)
    return "".join(letters)
