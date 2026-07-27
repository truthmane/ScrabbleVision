"""CLI: run the pipeline against a real broadcast video and compare
detected turns to the real GCG record, producing the metrics
`autoscorer.eval.metrics.GameEvalReport` defines.

The video itself can never be committed (copyright), but the baseline
JSON this produces is -- that's the whole point: today's numbers become
a contract every later workstream must improve against, not just a
one-off printout nobody can reproduce later.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from autoscorer.eval.gcg_truth import load_truth_turns
from autoscorer.eval.metrics import DetectedTurn, GameEvalReport, Provenance, StallInfo, build_report
from autoscorer.gamelogic.board import BoardState, Coord, Tile
from autoscorer.gamelogic.dictionary.lexicon import load_lexicon
from autoscorer.gamelogic.models import MoveType
from autoscorer.gamelogic.publish import PublishMode
from autoscorer.perception.capture.run_watcher import PUBLISH_MODES, run_watcher_on_video
from training.collect.replay_game import read_gcg_moves, replay_gcg_game


def _letters_from_move_score(move_score, new_cells) -> Dict[Coord, str]:
    """Every new cell's decoded letter, read off the scored word(s) --
    `move_score.words[0]` is the main word and by construction covers
    every new cell of a valid placement; any cross-words in the rest of
    the list just confirm the same letter at their intersection cell."""
    letters: Dict[Coord, str] = {}
    for word in move_score.words:
        for coord, ch in zip(word.cells, word.text):
            if coord in new_cells:
                letters[coord] = ch
    return letters


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent, text=True,
        ).strip()
    except Exception:
        return None


def run_and_evaluate(
    video_path: Path,
    gcg_path: Path,
    venue_name: str,
    classifier_path: Path,
    player1_id: str,
    player2_id: str,
    sample_fps: Optional[float] = 0.2,
    mode: PublishMode = PublishMode.AUTONOMOUS,
    confidence_threshold: float = 0.9,
    max_frames: Optional[int] = None,
    device: str = "cpu",
    lexicon_name: Optional[str] = None,
) -> GameEvalReport:
    truth_turns = load_truth_turns(gcg_path)

    detected: List[DetectedTurn] = []
    stalls: List[StallInfo] = []
    last_frame_index = [0]
    # A "stall" here is any maximal run of consecutive settled
    # observations that all failed with the exact same (reason, attempted
    # cells) -- exactly the symptom a permanently jammed watcher produces
    # (163 consecutive identical failures in the real run that motivated
    # this whole evaluator). A commit, a plain settle/no-op, or a change
    # in reason/cells all end the current run.
    stall_state = {"reason": None, "cells": None, "start": 0, "length": 0}

    def _flush_stall() -> None:
        if stall_state["length"] > 0:
            stalls.append(StallInfo(
                start_frame=stall_state["start"], length=stall_state["length"],
                reason=stall_state["reason"], attempted_cells=stall_state["cells"],
            ))
        stall_state["length"] = 0

    def on_frame_event(i: int, event) -> None:
        last_frame_index[0] = i
        if event.needs_operator and event.scored_move is None and event.reason is not None:
            cells = frozenset(event.attempted_cells)
            if event.reason == stall_state["reason"] and cells == stall_state["cells"]:
                stall_state["length"] += 1
            else:
                _flush_stall()
                stall_state.update(reason=event.reason, cells=cells, start=i, length=1)
        else:
            _flush_stall()

    def on_event(event) -> None:
        candidate = event.scored_move.candidate
        if candidate.move_type != MoveType.PLAY:
            return
        new_cells = set(candidate.new_cells)
        letters = (
            _letters_from_move_score(event.scored_move.move_score, new_cells)
            if event.scored_move.move_score is not None else {}
        )
        detected.append(DetectedTurn(
            frame_index=last_frame_index[0],
            player=candidate.player_id,
            cells=frozenset(candidate.new_cells),
            letters=letters,
            blank_cells=frozenset(candidate.blank_cells),
            score=event.scored_move.move_score.total if event.scored_move.move_score is not None else None,
            needs_operator=event.needs_operator,
        ))

    t0 = time.time()
    run_watcher_on_video(
        video_path, venue_name, classifier_path, player1_id, player2_id,
        sample_fps=sample_fps, mode=mode, confidence_threshold=confidence_threshold,
        max_frames=max_frames, device=device, on_event=on_event, on_frame_event=on_frame_event,
        lexicon_name=lexicon_name,
    )
    _flush_stall()
    wall_clock = time.time() - t0

    # The final board a live deployment would actually have, reconstructed
    # from only the AUTO-PUBLISHED (non-operator-pending) detected turns --
    # exactly what `GameWatcher._board` holds in standalone mode.
    final_board = BoardState()
    for dt in detected:
        if dt.needs_operator:
            continue
        placements = {
            coord: Tile(letter=label, is_blank=(coord in dt.blank_cells))
            for coord, label in dt.letters.items()
        }
        final_board = final_board.with_placements(placements)

    truth_replayed = replay_gcg_game(read_gcg_moves(gcg_path))
    truth_final_board = truth_replayed[-1].board_after if truth_replayed else BoardState()

    resolved_lexicon = load_lexicon(lexicon_name)
    provenance = Provenance(
        lexicon_name=resolved_lexicon.name,
        lexicon_word_count=resolved_lexicon.word_count,
        classifier_checkpoint=str(classifier_path),
        venue=venue_name,
        git_sha=_git_sha(),
        wall_clock_s=wall_clock,
        sample_fps=sample_fps,
        publish_mode=mode.value if hasattr(mode, "value") else str(mode),
    )

    return build_report(
        detected, truth_turns, stalls=stalls, final_board=final_board,
        truth_final_board=truth_final_board, provenance=provenance,
    )


def check_for_regressions(report: GameEvalReport, baseline: dict) -> List[str]:
    """Fails only on a real regression against a committed baseline --
    an improvement never fails this. Deliberately loose (a handful of
    headline fields, not every metric) since many fields can move in
    either direction between honest runs without either being wrong."""
    violations = []
    data = report.to_json_dict()
    # A baseline captured at a different sampling rate is not comparable
    # AT ALL -- the stillness gate is a frame count, so its wall-clock
    # meaning scales with sample_fps, and the same code+video behaves
    # completely differently across rates (see Provenance.sample_fps).
    # Refuse loudly instead of reporting phantom regressions/improvements.
    baseline_fps = (baseline.get("provenance") or {}).get("sample_fps")
    report_fps = (data.get("provenance") or {}).get("sample_fps")
    if baseline_fps is not None and report_fps is not None and baseline_fps != report_fps:
        return [
            f"NOT COMPARABLE: baseline was captured at sample_fps={baseline_fps}, this run at "
            f"{report_fps} -- re-run with --sample-fps {baseline_fps} (or regenerate the baseline)"
        ]
    if data["detected_turns"] < baseline["detected_turns"]:
        violations.append(f"detected_turns regressed: {data['detected_turns']} < {baseline['detected_turns']}")
    baseline_divergence = baseline["first_divergence_index"] if baseline["first_divergence_index"] is not None else 10**9
    report_divergence = data["first_divergence_index"] if data["first_divergence_index"] is not None else 10**9
    if report_divergence < baseline_divergence:
        violations.append(
            f"first_divergence_index regressed: {data['first_divergence_index']} < {baseline['first_divergence_index']}"
        )
    if data["longest_stall"] > baseline["longest_stall"]:
        violations.append(f"longest_stall regressed: {data['longest_stall']} > {baseline['longest_stall']}")
    if data["cell_f1_micro"] < baseline["cell_f1_micro"] - 0.01:
        violations.append(f"cell_f1_micro regressed: {data['cell_f1_micro']:.3f} < {baseline['cell_f1_micro']:.3f}")
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--gcg", required=True, type=Path)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--classifier", type=Path, default=Path("models/tile_classifier_v1.pt"))
    parser.add_argument("--player1", required=True)
    parser.add_argument("--player2", required=True)
    parser.add_argument("--sample-fps", type=float, default=0.2, help="MUST match the rate the venue profile's stillness gate was tuned at (still_frame_count is a frame count, so its meaning in seconds scales with this) -- 0.2 is what every wespa_word_wars calibration run used; recorded in provenance either way")
    parser.add_argument("--mode", choices=list(PUBLISH_MODES), default="autonomous")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lexicon", default=None, help="name/path resolved via dictionary.lexicon.load_lexicon; defaults to the venue profile's own lexicon, or the vendored default")
    parser.add_argument("--baseline", type=Path, default=None, help="a previously committed report JSON to compare against")
    parser.add_argument("--json-out", type=Path, default=None, help="write the full report as JSON here")
    args = parser.parse_args()

    report = run_and_evaluate(
        args.video_path, args.gcg, args.venue, args.classifier, args.player1, args.player2,
        sample_fps=args.sample_fps, mode=PUBLISH_MODES[args.mode], confidence_threshold=args.confidence_threshold,
        max_frames=args.max_frames, device=args.device, lexicon_name=args.lexicon,
    )
    print(report.summary())

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report.to_json_dict(), indent=2) + "\n")

    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        violations = check_for_regressions(report, baseline)
        if violations:
            print("\nREGRESSIONS:")
            for v in violations:
                print(f"  - {v}")
            raise SystemExit(1)
        print("\nNo regressions vs baseline.")


if __name__ == "__main__":
    main()
