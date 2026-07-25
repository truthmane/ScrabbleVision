"""The end-to-end game-replay metric WS5 (docs/classifier-accuracy-plan.md)
calls for: run the REAL classifier and REAL constraint decoder against a
harvested real photo's known new-tile cells for one turn, and check whether
the resulting word/score matches what the game's official .gcg record says
actually happened -- M3 ("per-move, after constraint decoding") and a
per-move operator-routing decision (the M4 side of the same coin).

This deliberately does NOT run occupancy detection to *find* the new
cells -- that's a separately-tested concern (see tests/unit/test_occupancy.py)
and mixing it in here would conflate two different failure modes. Instead
the new-cell coordinates come from the .gcg replay itself (ground truth),
isolating exactly the question this metric is for: given the right cells,
does classification + constraint decoding read them correctly?

Usage (the rectified photo must already exist locally -- these can't be
committed to the repo, see models/README.md's copyright note):
    eval_harvested_moves.py rectified.jpg game.gcg 22 checkpoint.pt
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2

from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.movedetect.constraint_decoder import CellCandidates, decode_feasible_reading
from autoscorer.gamelogic.movedetect.word_resolver import words_formed
from autoscorer.gamelogic.notation import NewTilePlacement, resolve_new_tiles_gcg
from autoscorer.gamelogic.scoring.rules_engine import score_move
from autoscorer.perception.calibration.homography import crop_cell
from training.classify.infer import TileClassifierModel
from training.collect.replay_game import read_gcg_moves

CONFIDENCE_THRESHOLD = 0.9


@dataclass(frozen=True)
class MoveEvalResult:
    move_number: int
    has_blank: bool
    raw_top1_correct: int
    decoded_correct: int
    cell_count: int
    would_route_to_operator: bool  # any blank, or any raw confidence below threshold
    move_score_matches: Optional[bool]  # None if not auto-scoreable (a decoded BLANK)
    true_score: int
    computed_score: Optional[int]


def _board_before_and_placements(gcg_path: Path, move_number: int) -> tuple:
    """Replays every move strictly before `move_number` to get the exact
    board state going into it, then resolves that move's own new-tile
    placements -- both straight from the official record, independent of
    any photo. `move_number` is 1-indexed, matching replay_game.py."""
    moves = read_gcg_moves(gcg_path)
    board_before = BoardState()
    for move in moves[: move_number - 1]:
        placements = resolve_new_tiles_gcg(move)
        board_before = board_before.with_placements(
            {p.coord: Tile(p.letter, p.is_blank) for p in placements}
        )
    this_move = moves[move_number - 1]
    placements: List[NewTilePlacement] = resolve_new_tiles_gcg(this_move)
    return board_before, placements, this_move.turn_score


def evaluate_move(
    rectified_frame_path: Path,
    gcg_path: Path,
    move_number: int,
    classifier: TileClassifierModel,
) -> MoveEvalResult:
    rectified = cv2.imread(str(rectified_frame_path))
    if rectified is None:
        raise ValueError(f"could not read image at {rectified_frame_path}")

    board_before, placements, true_score = _board_before_and_placements(gcg_path, move_number)

    cell_candidates: List[CellCandidates] = []
    raw_top1 = {}
    for placement in placements:
        crop = crop_cell(rectified, *placement.coord)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        candidates = classifier.predict_topk(crop_rgb, k=3)
        cell_candidates.append(CellCandidates(coord=placement.coord, candidates=candidates))
        raw_top1[placement.coord] = candidates[0]

    decoded = decode_feasible_reading(cell_candidates, board_before, racks=[])

    true_by_coord = {p.coord: p for p in placements}
    raw_correct = sum(
        1 for coord, p in true_by_coord.items()
        if raw_top1[coord][0] == ("BLANK" if p.is_blank else p.letter)
    )
    decoded_correct = sum(
        1 for coord, p in true_by_coord.items()
        if decoded[coord] == ("BLANK" if p.is_blank else p.letter)
    )

    has_blank_true = any(p.is_blank for p in placements)
    has_blank_decoded = any(label == "BLANK" for label in decoded.values())
    would_route = has_blank_decoded or any(conf < CONFIDENCE_THRESHOLD for _, conf in raw_top1.values())

    move_score_matches: Optional[bool] = None
    computed_score: Optional[int] = None
    if not has_blank_decoded:
        # Safe to actually place and score -- board.py refuses a letter-less
        # blank on the board by construction, so this branch is only
        # reachable when every decoded cell has a concrete letter.
        decoded_placements = {
            p.coord: Tile(decoded[p.coord], is_blank=p.is_blank) for p in placements
        }
        board_after = board_before.with_placements(decoded_placements)
        new_cells = list(decoded_placements.keys())
        words = words_formed(board_after, new_cells)
        computed_score = score_move(board_after, words, new_cells).total
        move_score_matches = computed_score == true_score

    return MoveEvalResult(
        move_number=move_number,
        has_blank=has_blank_true,
        raw_top1_correct=raw_correct,
        decoded_correct=decoded_correct,
        cell_count=len(placements),
        would_route_to_operator=would_route,
        move_score_matches=move_score_matches,
        true_score=true_score,
        computed_score=computed_score,
    )


def print_result(label: str, result: MoveEvalResult) -> None:
    outcome = "ROUTED_TO_OPERATOR" if result.would_route_to_operator else "AUTO_PUBLISHED"
    score_note = (
        "n/a (routed)" if result.move_score_matches is None
        else ("MATCH" if result.move_score_matches else f"MISMATCH ({result.computed_score} != {result.true_score})")
    )
    danger = " <-- WRONG MOVE WOULD HAVE AUTO-PUBLISHED" if (
        not result.would_route_to_operator and result.move_score_matches is False
    ) else ""
    print(
        f"{label} move {result.move_number}: "
        f"{result.decoded_correct}/{result.cell_count} cells correct after decoding "
        f"(raw top-1: {result.raw_top1_correct}/{result.cell_count}), "
        f"blank={result.has_blank}, {outcome}, score={score_note}{danger}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rectified_frame", type=Path)
    parser.add_argument("gcg_file", type=Path)
    parser.add_argument("move_number", type=int)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    classifier = TileClassifierModel(args.checkpoint, device="cpu")
    result = evaluate_move(args.rectified_frame, args.gcg_file, args.move_number, classifier)
    print_result(args.rectified_frame.name, result)


if __name__ == "__main__":
    main()
