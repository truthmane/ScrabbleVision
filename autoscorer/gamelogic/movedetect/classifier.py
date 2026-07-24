"""Classifies a turn as a tile placement, an exchange, or a pass by
diffing board and rack state -- no camera-specific concepts here, just
before/after state comparison.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Sequence

from autoscorer.gamelogic.board import BoardState, Coord, Tile
from autoscorer.gamelogic.models import MoveType


def diff_new_cells(board_before: BoardState, board_after: BoardState) -> List[Coord]:
    """Cells occupied in `board_after` but empty in `board_before`."""
    return [coord for coord in board_after.occupied_cells() if board_before.is_empty(coord)]


def _rack_signature(racks: Sequence[Sequence[Tile]]) -> List[Counter]:
    return [Counter((tile.letter, tile.is_blank) for tile in rack) for rack in racks]


def racks_changed(racks_before: Sequence[Sequence[Tile]], racks_after: Sequence[Sequence[Tile]]) -> bool:
    return _rack_signature(racks_before) != _rack_signature(racks_after)


def classify_move(
    board_before: BoardState,
    board_after: BoardState,
    racks_before: Sequence[Sequence[Tile]],
    racks_after: Sequence[Sequence[Tile]],
) -> MoveType:
    if diff_new_cells(board_before, board_after):
        return MoveType.PLAY
    if racks_changed(racks_before, racks_after):
        return MoveType.EXCHANGE
    return MoveType.PASS
