"""Validates that a set of newly-placed tiles forms a legal Scrabble move:
a single line, contiguous with any pre-existing tiles (or the center square
on the first move), and connected to the rest of the board thereafter.

Pure logic over board coordinates -- no CV, no confidence scores. Takes the
board *before* the move and the coordinates the perception/operator layer
claims are newly occupied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from autoscorer.gamelogic.board import CENTER, BoardState, Coord, format_square
from autoscorer.gamelogic.movedetect.word_resolver import run_through


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: Optional[str] = None

    @staticmethod
    def success() -> "ValidationResult":
        return ValidationResult(True)

    @staticmethod
    def failure(reason: str) -> "ValidationResult":
        return ValidationResult(False, reason)


def validate_placement(
    board_before: BoardState,
    board_after: BoardState,
    new_cells: Sequence[Coord],
) -> ValidationResult:
    """Validate a placement. `board_after` must equal `board_before` with
    exactly `new_cells` filled in (and nothing else changed) -- callers
    build it via `board_before.with_placements(...)`.
    """
    if not new_cells:
        return ValidationResult.failure("no new tiles placed")

    for coord in new_cells:
        if not board_before.is_empty(coord):
            return ValidationResult.failure(f"cell {format_square(coord)} was already occupied before this move")

    is_first_move = board_before.is_blank_board()

    # Axis is ambiguous for a single new tile (a lone tile has no "line" of
    # its own); gap-checking via the main run is only meaningful when there's
    # more than one new tile.
    axis = None
    if len(new_cells) > 1:
        rows = {r for r, _ in new_cells}
        cols = {c for _, c in new_cells}
        if len(rows) > 1 and len(cols) > 1:
            return ValidationResult.failure("new tiles must lie in a single row or column")
        axis = "row" if len(rows) == 1 else "col"

    anchor_r, anchor_c = new_cells[0]
    main_run = run_through(board_after, anchor_r, anchor_c, axis) if axis is not None else None

    if main_run is not None and not set(new_cells).issubset(set(main_run)):
        # run_through only ever extends through occupied cells, so a gap in
        # the requested placement would silently produce a shorter run that
        # excludes some of new_cells -- that's how we detect it here.
        return ValidationResult.failure("new tiles are not contiguous with each other/existing tiles")

    if is_first_move:
        if CENTER not in new_cells:
            return ValidationResult.failure("the first move must cover the center square")
        return ValidationResult.success()

    # Not the first move: at least one new tile must connect to a
    # pre-existing tile, either in-line (the main run includes an old tile)
    # or perpendicularly (a cross-word run of length > 1).
    connected = False
    if main_run is not None:
        if len(main_run) > len(new_cells):
            connected = True
        cross_axis = "col" if axis == "row" else "row"
        for (r, c) in new_cells:
            if len(run_through(board_after, r, c, cross_axis)) > 1:
                connected = True
                break
    else:
        r, c = new_cells[0]
        if len(run_through(board_after, r, c, "row")) > 1 or len(run_through(board_after, r, c, "col")) > 1:
            connected = True

    if not connected:
        return ValidationResult.failure("placement is not connected to any existing tile")

    return ValidationResult.success()
