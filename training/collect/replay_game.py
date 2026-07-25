"""Replays a transcribed game (ticker lines, one per move) through the real
`GameSession`/rules engine, in the exact same way a live broadcast would
feed it moves -- except the "camera" here is a human reading the ticker
text off the video instead of a photo of the board.

This is WS3 from `docs/classifier-accuracy-plan.md`: reconstructing exact
board state (including which letter every blank was played as) at every
turn of a real game without reading a single pixel, so frames harvested
from that game's video can be auto-labeled from the replayed state instead
of a human reading an axis-labeled crop table cell by cell. It also
doubles as a free end-to-end regression test of the scoring engine against
real official tournament scores -- nothing has exercised that before.

Input file format: one ticker line per move (see `notation.py` for the
exact grammar), blank lines and lines starting with '#' ignored, e.g.:

    Krafchick, Joey H8 QUIXOTIC 92 92
    Reinke, Thomas J7 TaSKING 86 86
    Krafchick, Joey E2 VODU(N) 18 110

Usage:
    replay_game.py moves.txt
    replay_game.py moves.txt --board-at 5   # print board state after move 5
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

from autoscorer.api.session import GameSession
from autoscorer.gamelogic.board import BOARD_SIZE, BoardState
from autoscorer.gamelogic.models import MoveProcessingError
from autoscorer.gamelogic.notation import (
    GcgMove,
    TranscribedMove,
    parse_gcg_line,
    parse_ticker_line,
    resolve_new_tiles,
    resolve_new_tiles_gcg,
)
from autoscorer.gamelogic.publish import PublishMode


@dataclass(frozen=True)
class ReplayedTurn:
    move: Union[TranscribedMove, GcgMove]
    board_after: BoardState
    computed_score: Optional[int]  # None if the line failed to process at all
    score_matches: bool
    cumulative_matches: bool
    error: Optional[str] = None


def read_moves(path: Path) -> List[TranscribedMove]:
    moves = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            moves.append(parse_ticker_line(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return moves


def read_gcg_moves(path: Path) -> List[GcgMove]:
    """Extracts every standard play line from an official .gcg game
    record, silently skipping lines our rules engine doesn't model
    (passes, exchanges, challenges, the end-of-game rack bonus/penalty
    line) -- see `notation.parse_gcg_line`."""
    moves = []
    for raw in path.read_text().splitlines():
        move = parse_gcg_line(raw)
        if move is not None:
            moves.append(move)
    return moves


def replay_game(moves: Sequence[TranscribedMove]) -> List[ReplayedTurn]:
    """Feeds each move through a real `GameSession` in submission order,
    same as a live pipeline would. Rack contents are never known from
    ticker text alone, so every submission passes no rack info -- harmless
    here because every ticker line is a completed play (the board always
    changes), which is all `classify_move` needs to route it as PLAY.
    """
    session = GameSession(mode=PublishMode.AUTONOMOUS)
    results: List[ReplayedTurn] = []

    for move in moves:
        placements = resolve_new_tiles(move)
        new_tiles = [(p.coord, p.letter, p.is_blank) for p in placements]
        submission = session.submit_move(move.player, new_tiles=new_tiles)

        if isinstance(submission.outcome, MoveProcessingError):
            results.append(ReplayedTurn(
                move=move, board_after=session.game_state.board, computed_score=None,
                score_matches=False, cumulative_matches=False, error=submission.outcome.reason,
            ))
            continue

        move_score = submission.outcome.move_score
        computed_total = move_score.total if move_score is not None else 0
        cumulative = session.game_state.scores.get(move.player, 0)
        results.append(ReplayedTurn(
            move=move,
            board_after=session.game_state.board,
            computed_score=computed_total,
            score_matches=computed_total == move.turn_score,
            cumulative_matches=cumulative == move.cumulative_score,
        ))

    return results


def replay_gcg_game(moves: Sequence[GcgMove]) -> List[ReplayedTurn]:
    """Same as `replay_game`, for moves parsed from an official .gcg
    record instead of a hand-transcribed ticker line."""
    session = GameSession(mode=PublishMode.AUTONOMOUS)
    results: List[ReplayedTurn] = []

    for move in moves:
        placements = resolve_new_tiles_gcg(move)
        new_tiles = [(p.coord, p.letter, p.is_blank) for p in placements]
        submission = session.submit_move(move.player, new_tiles=new_tiles)

        if isinstance(submission.outcome, MoveProcessingError):
            results.append(ReplayedTurn(
                move=move, board_after=session.game_state.board, computed_score=None,
                score_matches=False, cumulative_matches=False, error=submission.outcome.reason,
            ))
            continue

        move_score = submission.outcome.move_score
        computed_total = move_score.total if move_score is not None else 0
        cumulative = session.game_state.scores.get(move.player, 0)
        results.append(ReplayedTurn(
            move=move,
            board_after=session.game_state.board,
            computed_score=computed_total,
            score_matches=computed_total == move.turn_score,
            cumulative_matches=cumulative == move.cumulative_score,
        ))

    return results


def render_board(board: BoardState) -> str:
    lines = []
    for row in range(BOARD_SIZE):
        cells = []
        for col in range(BOARD_SIZE):
            tile = board.get((row, col))
            cells.append("." if tile is None else tile.letter)
        lines.append(" ".join(cells))
    return "\n".join(lines)


def print_report(results: Sequence[ReplayedTurn]) -> None:
    mismatches = 0
    for i, turn in enumerate(results, start=1):
        if turn.error is not None:
            mismatches += 1
            print(f"[{i:3d}] {turn.move.player:<20} {turn.move.position:<5} {turn.move.word:<12} ERROR: {turn.error}")
            continue
        ok = "OK" if turn.score_matches and turn.cumulative_matches else "MISMATCH"
        if ok == "MISMATCH":
            mismatches += 1
        print(
            f"[{i:3d}] {turn.move.player:<20} {turn.move.position:<5} {turn.move.word:<12} "
            f"computed={turn.computed_score:<4} ticker={turn.move.turn_score:<4} "
            f"cumulative_ticker={turn.move.cumulative_score:<4} -> {ok}"
        )
    print(f"\n{len(results) - mismatches}/{len(results)} moves matched the ticker exactly.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("moves_file", type=Path, help="a ticker-line .txt transcript, or an official .gcg game record")
    parser.add_argument("--board-at", type=int, default=None, help="print the board state after this move number (1-indexed)")
    args = parser.parse_args()

    if args.moves_file.suffix == ".gcg":
        results = replay_gcg_game(read_gcg_moves(args.moves_file))
    else:
        results = replay_game(read_moves(args.moves_file))
    print_report(results)

    if args.board_at is not None:
        turn = results[args.board_at - 1]
        print(f"\nBoard state after move {args.board_at}:\n")
        print(render_board(turn.board_after))


if __name__ == "__main__":
    main()
