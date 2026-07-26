"""Aggregate accuracy/health metrics for one game's detection run against
its real GCG ground truth, built on `alignment.align_turns`.

Every metric here is reported separately and never collapsed into one
number -- `turns_detected`, `first_divergence_index`, and
`exact_score_matches` answer different questions, and provenance travels
with every report because none of these numbers are comparable across
runs without knowing what produced them (checkpoint, lexicon, venue).

Pure logic: no cv2, no torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence

from autoscorer.eval.alignment import align_turns
from autoscorer.eval.gcg_truth import TruthTurn
from autoscorer.gamelogic.board import BoardState, Coord
from autoscorer.gamelogic.notation import format_square


@dataclass(frozen=True)
class DetectedTurn:
    """One committed (or pending) move from a `GameWatcher` run, in the
    shape the evaluator needs -- deliberately not `WatcherEvent` itself,
    so this module has no cv2/torch import chain through `game_watcher.py`.
    """
    frame_index: int
    player: Optional[str]
    cells: FrozenSet[Coord]
    letters: Dict[Coord, str]  # decoded label per cell; "BLANK" for an unresolved blank
    blank_cells: FrozenSet[Coord]
    score: Optional[int]
    needs_operator: bool


@dataclass(frozen=True)
class StallInfo:
    start_frame: int
    length: int
    reason: str
    attempted_cells: FrozenSet[Coord]


@dataclass(frozen=True)
class Provenance:
    lexicon_name: Optional[str] = None
    lexicon_word_count: Optional[int] = None
    classifier_checkpoint: Optional[str] = None
    venue: Optional[str] = None
    git_sha: Optional[str] = None
    wall_clock_s: Optional[float] = None


@dataclass(frozen=True)
class GameEvalReport:
    real_plays: int
    detected_turns: int
    matched_1to1: int
    merged: int
    split: int
    missed: int
    spurious: int
    first_divergence_index: Optional[int]  # 1-indexed truth turn_number; None if nothing diverged

    cell_precision_macro: float
    cell_recall_macro: float
    cell_f1_macro: float
    cell_precision_micro: float
    cell_recall_micro: float
    cell_f1_micro: float

    letter_accuracy: Optional[float]  # over MATCH-op intersection cells; None if no such cells
    letter_correct: int
    letter_total: int
    blank_recovered: int
    blank_total: int

    exact_score_matches: int
    score_abs_errors: List[int] = field(default_factory=list)
    final_cumulative_drift: Dict[str, int] = field(default_factory=dict)  # player -> computed - truth

    final_board_correct: int = 0
    final_board_truth_total: int = 0
    final_board_wrong: int = 0
    final_board_missing: int = 0
    final_board_extra: int = 0

    stalls: List[StallInfo] = field(default_factory=list)
    longest_stall: int = 0
    operator_routed_fraction: float = 0.0

    provenance: Provenance = field(default_factory=Provenance)

    def to_json_dict(self) -> dict:
        return {
            "real_plays": self.real_plays,
            "detected_turns": self.detected_turns,
            "matched_1to1": self.matched_1to1,
            "merged": self.merged,
            "split": self.split,
            "missed": self.missed,
            "spurious": self.spurious,
            "first_divergence_index": self.first_divergence_index,
            "cell_precision_macro": self.cell_precision_macro,
            "cell_recall_macro": self.cell_recall_macro,
            "cell_f1_macro": self.cell_f1_macro,
            "cell_precision_micro": self.cell_precision_micro,
            "cell_recall_micro": self.cell_recall_micro,
            "cell_f1_micro": self.cell_f1_micro,
            "letter_accuracy": self.letter_accuracy,
            "letter_correct": self.letter_correct,
            "letter_total": self.letter_total,
            "blank_recovered": self.blank_recovered,
            "blank_total": self.blank_total,
            "exact_score_matches": self.exact_score_matches,
            "score_abs_errors": self.score_abs_errors,
            "final_cumulative_drift": self.final_cumulative_drift,
            "final_board_correct": self.final_board_correct,
            "final_board_truth_total": self.final_board_truth_total,
            "final_board_wrong": self.final_board_wrong,
            "final_board_missing": self.final_board_missing,
            "final_board_extra": self.final_board_extra,
            "stalls": [
                {
                    "start_frame": s.start_frame,
                    "length": s.length,
                    "reason": s.reason,
                    "attempted_squares": sorted(format_square(c) for c in s.attempted_cells),
                }
                for s in self.stalls
            ],
            "longest_stall": self.longest_stall,
            "operator_routed_fraction": self.operator_routed_fraction,
            "provenance": {
                "lexicon_name": self.provenance.lexicon_name,
                "lexicon_word_count": self.provenance.lexicon_word_count,
                "classifier_checkpoint": self.provenance.classifier_checkpoint,
                "venue": self.provenance.venue,
                "git_sha": self.provenance.git_sha,
                "wall_clock_s": self.provenance.wall_clock_s,
            },
        }

    def summary(self) -> str:
        lines = [
            f"{self.detected_turns} detected / {self.real_plays} real plays "
            f"({self.matched_1to1} matched, {self.merged} merged, {self.split} split, "
            f"{self.missed} missed, {self.spurious} spurious)",
            f"first_divergence_index={self.first_divergence_index}",
            f"cell F1 macro={self.cell_f1_macro:.3f} micro={self.cell_f1_micro:.3f}",
            f"letter_accuracy={self.letter_accuracy}"
            f" ({self.letter_correct}/{self.letter_total})"
            f", blanks {self.blank_recovered}/{self.blank_total}",
            f"exact_score_matches={self.exact_score_matches}, "
            f"final_cumulative_drift={self.final_cumulative_drift}",
            f"final_board: correct={self.final_board_correct} wrong={self.final_board_wrong} "
            f"missing={self.final_board_missing} extra={self.final_board_extra} "
            f"/ {self.final_board_truth_total} truth cells",
            f"longest_stall={self.longest_stall} observations, "
            f"operator_routed_fraction={self.operator_routed_fraction:.2f}",
        ]
        for s in self.stalls:
            squares = ", ".join(sorted(format_square(c) for c in s.attempted_cells))
            lines.append(f"  stall @ frame {s.start_frame}, {s.length} observations, "
                          f"reason={s.reason!r}, squares=[{squares}]")
        return "\n".join(lines)


def _avg(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 1.0


def build_report(
    detected: Sequence[DetectedTurn],
    truth: Sequence[TruthTurn],
    stalls: Sequence[StallInfo] = (),
    final_board: Optional[BoardState] = None,
    truth_final_board: Optional[BoardState] = None,
    provenance: Provenance = Provenance(),
) -> GameEvalReport:
    detected_cells = [d.cells for d in detected]
    truth_cells = [t.cells for t in truth]
    ops = align_turns(detected_cells, truth_cells)

    matched_1to1 = sum(1 for o in ops if o.kind == "MATCH")
    merged = sum(1 for o in ops if o.kind == "MERGE")
    split = sum(1 for o in ops if o.kind == "SPLIT")
    missed = sum(1 for o in ops if o.kind == "MISSED")
    spurious = sum(1 for o in ops if o.kind == "SPURIOUS")

    clean_truth_indices = set()
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    total_tp = total_fp = total_fn = 0
    letter_correct = letter_total = 0
    blank_recovered = blank_total = 0
    exact_score_matches = 0
    score_abs_errors: List[int] = []

    for op in ops:
        if op.kind != "MATCH":
            continue
        di, ti = op.detected_indices[0], op.truth_indices[0]
        d, t = detected[di], truth[ti]

        tp = len(d.cells & t.cells)
        fp = len(d.cells - t.cells)
        fn = len(t.cells - d.cells)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        for coord in d.cells & t.cells:
            letter_total += 1
            truth_letter = t.letters.get(coord)
            detected_letter = d.letters.get(coord)
            is_correct = (
                detected_letter is not None
                and truth_letter is not None
                and detected_letter.upper() == truth_letter.upper()
            )
            if is_correct:
                letter_correct += 1
            if coord in t.blank_cells:
                blank_total += 1
                if is_correct:
                    blank_recovered += 1

        if d.cells == t.cells and d.score is not None and d.score == t.turn_score:
            exact_score_matches += 1
            clean_truth_indices.add(ti)
        if d.score is not None:
            score_abs_errors.append(abs(d.score - t.turn_score))

    first_divergence_index = None
    for idx, t in enumerate(truth):
        if idx not in clean_truth_indices:
            first_divergence_index = t.turn_number
            break

    cell_precision_micro = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    cell_recall_micro = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    cell_f1_micro = (
        2 * cell_precision_micro * cell_recall_micro / (cell_precision_micro + cell_recall_micro)
        if (cell_precision_micro + cell_recall_micro) else 0.0
    )
    letter_accuracy = letter_correct / letter_total if letter_total else None

    final_cumulative_drift: Dict[str, int] = {}
    computed_totals: Dict[str, int] = {}
    for d in detected:
        if d.player is not None and d.score is not None and not d.needs_operator:
            computed_totals[d.player] = computed_totals.get(d.player, 0) + d.score
    truth_final_cum: Dict[str, int] = {}
    for t in truth:
        truth_final_cum[t.player] = t.cumulative_score
    for player, truth_cum in truth_final_cum.items():
        final_cumulative_drift[player] = computed_totals.get(player, 0) - truth_cum

    final_board_correct = final_board_wrong = final_board_missing = final_board_extra = 0
    final_board_truth_total = 0
    if final_board is not None and truth_final_board is not None:
        truth_occ = {c: truth_final_board.get(c) for c in truth_final_board.occupied_cells()}
        final_board_truth_total = len(truth_occ)
        det_occ = {c: final_board.get(c) for c in final_board.occupied_cells()}
        for c, truth_tile in truth_occ.items():
            det_tile = det_occ.get(c)
            if det_tile is None:
                final_board_missing += 1
            elif det_tile.letter == truth_tile.letter:
                final_board_correct += 1
            else:
                final_board_wrong += 1
        final_board_extra = len(set(det_occ) - set(truth_occ))

    return GameEvalReport(
        real_plays=len(truth),
        detected_turns=len(detected),
        matched_1to1=matched_1to1,
        merged=merged,
        split=split,
        missed=missed,
        spurious=spurious,
        first_divergence_index=first_divergence_index,
        cell_precision_macro=_avg(precisions),
        cell_recall_macro=_avg(recalls),
        cell_f1_macro=_avg(f1s),
        cell_precision_micro=cell_precision_micro,
        cell_recall_micro=cell_recall_micro,
        cell_f1_micro=cell_f1_micro,
        letter_accuracy=letter_accuracy,
        letter_correct=letter_correct,
        letter_total=letter_total,
        blank_recovered=blank_recovered,
        blank_total=blank_total,
        exact_score_matches=exact_score_matches,
        score_abs_errors=score_abs_errors,
        final_cumulative_drift=final_cumulative_drift,
        final_board_correct=final_board_correct,
        final_board_truth_total=final_board_truth_total,
        final_board_wrong=final_board_wrong,
        final_board_missing=final_board_missing,
        final_board_extra=final_board_extra,
        stalls=list(stalls),
        longest_stall=max((s.length for s in stalls), default=0),
        operator_routed_fraction=(
            sum(1 for d in detected if d.needs_operator) / len(detected) if detected else 0.0
        ),
        provenance=provenance,
    )
