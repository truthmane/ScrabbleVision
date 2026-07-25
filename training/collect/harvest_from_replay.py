"""Auto-labels every occupied board cell in a rectified real photo, given
the `BoardState` reconstructed by replaying an official .gcg game record
(see `replay_game.py`). This is the payoff of WS3: no human reads a
crop table cell by cell -- the ticker/GCG replay already knows exactly
which letter (and whether it's a blank) sits in every occupied cell, so
labeling is just "crop this cell, name the file after what we already
know is there."

Usage:
    harvest_from_replay.py rectified_board.jpg game.gcg 13 out_dir/ prefix

Crops every occupied cell of the board state as it stood right after
move number 13 (1-indexed, matching replay_game's turn numbering) from
`rectified_board.jpg` (already homography-rectified to the canonical
900x900 grid -- see `perception/calibration/homography.py`), and writes
one `{LETTER-or-BLANK}_{prefix}_{row}_{col}.png` file per occupied cell.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from autoscorer.gamelogic.board import BoardState
from autoscorer.perception.calibration.homography import crop_cell
from training.collect.replay_game import read_gcg_moves, replay_gcg_game

# The classifier's 27 classes are A-Z plus this literal label (see
# models/README.md and constraint_decoder.CLASSIFIER_BLANK_LABEL) -- distinct
# from board.py's internal "_" bag/pool bookkeeping sentinel, which would
# make for an unreadable filename and wouldn't match the training pipeline's
# class-folder naming.
BLANK_LABEL = "BLANK"


def harvest_board_cells(rectified_path: Path, board: BoardState, out_dir: Path, prefix: str) -> int:
    rectified = cv2.imread(str(rectified_path))
    if rectified is None:
        raise ValueError(f"could not read image at {rectified_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for (row, col) in board.occupied_cells():
        tile = board.get((row, col))
        label = BLANK_LABEL if tile.is_blank else tile.letter
        crop = crop_cell(rectified, row, col)
        cv2.imwrite(str(out_dir / f"{label}_{prefix}_{row}_{col}.png"), crop)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rectified_board", type=Path)
    parser.add_argument("gcg_file", type=Path)
    parser.add_argument("move_number", type=int, help="1-indexed move number; board state is read as of right after this move")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("prefix")
    args = parser.parse_args()

    moves = read_gcg_moves(args.gcg_file)
    results = replay_gcg_game(moves)
    board = results[args.move_number - 1].board_after

    count = harvest_board_cells(args.rectified_board, board, args.out_dir, args.prefix)
    print(f"wrote {count} labeled board-cell crops")


if __name__ == "__main__":
    main()
