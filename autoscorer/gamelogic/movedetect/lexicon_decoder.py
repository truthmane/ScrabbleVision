"""Lexicon-constrained decoding: a beam search over each cell's full
temporal-voted class distribution, scored by pool feasibility (reusing
`constraint_decoder`'s machinery) plus how many of the words the reading
forms are real dictionary words.

**Phonies are legal, scoring Scrabble plays.** A word that isn't in any
dictionary, played and left unchallenged, scores exactly like a real
one -- so the lexicon here is a SOFT re-ranker, never a hard filter. It
can never reject a reading or invent a different one; the caller always
gets back the best reading found, whether or not every word it forms is
real, along with which words (if any) aren't. Measured: 98 of 282 real
words across 7 real WESPA games (35%) are absent from the vendored
generic list, so a hard filter would actively misread real boards.

`constraint_decoder.decode_feasible_reading` is left untouched -- its
existing tests and its `eval_harvested_moves.py` caller keep working
unchanged. This is a genuinely different algorithm (a position-ordered
beam, not a greedy confidence-sorted assignment), not a drop-in
replacement -- `GameWatcher` is the only caller switched over to it.
"""
from __future__ import annotations

import math
import string
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from autoscorer.gamelogic.board import BLANK, BoardState, Coord, Tile
from autoscorer.gamelogic.dictionary.lexicon import Lexicon
from autoscorer.gamelogic.movedetect.constraint_decoder import (
    CLASSIFIER_BLANK_LABEL,
    CellCandidates,
    _pool_key,
    remaining_supply,
)
from autoscorer.gamelogic.movedetect.word_resolver import Axis, run_through, word_text, words_formed

_LOG_EPS = 1e-9


@dataclass(frozen=True)
class Reading:
    labels: Dict[Coord, str]
    blank_cells: FrozenSet[Coord]
    log_prob: float
    invalid_words: Tuple[str, ...]
    all_words_valid: bool
    total_score: float  # log_prob + lexicon term; internal ranking key only


@dataclass
class _BeamState:
    assigned: Dict[Coord, str] = field(default_factory=dict)
    blank_cells: FrozenSet[Coord] = frozenset()
    remaining: Dict[str, int] = field(default_factory=dict)
    log_prob: float = 0.0
    lexicon_penalty: float = 0.0
    invalid_words: Tuple[str, ...] = ()
    blanks_used: int = 0

    def score(self, lexicon_weight: float) -> float:
        return self.log_prob + lexicon_weight * self.lexicon_penalty


def _order_cells_along_axis(cell_candidates: Sequence[CellCandidates]) -> Tuple[List[CellCandidates], Optional[Axis]]:
    """Position order (not confidence order, unlike `decode_feasible_reading`)
    -- required for prefix/incremental scoring. Assumes `cell_candidates`
    is already collinear (guaranteed by `placement_search`'s candidates,
    the only real caller) -- a single cell has no axis of its own."""
    coords = [cc.coord for cc in cell_candidates]
    if len(coords) <= 1:
        return list(cell_candidates), None
    rows = {r for r, _ in coords}
    axis: Axis = "row" if len(rows) == 1 else "col"
    key = (lambda cc: cc.coord[1]) if axis == "row" else (lambda cc: cc.coord[0])
    return sorted(cell_candidates, key=key), axis


def _cross_word_check(
    board_before: BoardState, coord: Coord, letter: str, is_blank: bool, cross_axis: Optional[Axis], lexicon: Optional[Lexicon],
) -> Tuple[float, Optional[str]]:
    """Checks the cross-word (if any) formed by placing `letter` at
    `coord` alone. Only board_before's existing tiles plus this one new
    cell matter here -- no other cell in this placement shares this
    perpendicular line (they're all on the same main axis, at other
    positions along it), so this needs no cumulative beam state."""
    if cross_axis is None:
        return 0.0, None
    hypothetical = board_before.with_placements({coord: Tile(letter=letter, is_blank=is_blank)})
    run = run_through(hypothetical, coord[0], coord[1], cross_axis)
    if len(run) <= 1:
        return 0.0, None
    text = word_text(hypothetical, run)
    if lexicon is None or lexicon.is_valid_word(text):
        return 0.0, None
    return -1.0, text


def _main_word_penalty_and_invalid(
    board_before: BoardState, assigned: Dict[Coord, str], blank_cells: FrozenSet[Coord],
    axis: Optional[Axis], lexicon: Optional[Lexicon],
) -> Tuple[float, List[str]]:
    """The main-word check (and, for a single new cell, both its
    row/column words) -- only meaningful once every cell in this
    placement is assigned, i.e. only called on a complete `assigned`."""
    if lexicon is None:
        return 0.0, []
    board_after = board_before.with_placements({
        coord: Tile(letter=letter, is_blank=(coord in blank_cells)) for coord, letter in assigned.items()
    })
    new_cells = list(assigned.keys())
    penalty = 0.0
    invalid: List[str] = []
    if axis is not None:
        main_run = run_through(board_after, new_cells[0][0], new_cells[0][1], axis)
        text = word_text(board_after, main_run)
        if not lexicon.is_valid_word(text):
            penalty -= 1.0
            invalid.append(text)
    else:
        # A single new cell: words_formed resolves whichever word(s) it
        # forms (row and/or column) -- both need checking here.
        for word_cells in words_formed(board_after, new_cells):
            text = word_text(board_after, word_cells)
            if not lexicon.is_valid_word(text):
                penalty -= 1.0
                invalid.append(text)
    return penalty, invalid


def decode_with_lexicon(
    cell_candidates: Sequence[CellCandidates],
    board_before: BoardState,
    racks: Sequence[Sequence[Tile]],
    lexicon: Optional[Lexicon] = None,
    beam_width: int = 12,
    top_labels_per_cell: int = 6,
    lexicon_weight: float = 2.0,
    max_blanks: int = 2,
) -> List[Reading]:
    """Best-first list of candidate readings (never empty), most likely
    first. `lexicon=None` or `lexicon_weight=0.0` reduces every lexicon
    term to zero, so ranking is by pool-feasible log-probability alone --
    used by the contract test asserting this reproduces
    `decode_feasible_reading` on its existing scenarios.

    Never invents a reading or drops a cell: if every candidate for some
    cell is pool-infeasible, that cell falls back to its raw top-1 label
    unconstrained (mirroring `decode_feasible_reading`'s own fallback),
    logged as a very low (not infinite) log-probability so the overall
    reading still ranks below any fully-feasible alternative.

    The main word's lexicon check is folded into the LAST cell's
    per-candidate scoring, before that step's beam truncation -- not done
    as a separate pass after the whole beam search finishes. Found by
    testing: a separate final pass computes the right penalty too late
    to matter, since `beam_width` may have already pruned away the
    candidate that needed it (all letters look identical on cross-word
    penalty and log-prob alone until the main word is known).
    """
    if not cell_candidates:
        return [Reading(labels={}, blank_cells=frozenset(), log_prob=0.0, invalid_words=(), all_words_valid=True, total_score=0.0)]

    ordered, axis = _order_cells_along_axis(cell_candidates)
    cross_axis: Optional[Axis] = None
    if axis is not None:
        cross_axis = "col" if axis == "row" else "row"

    initial_remaining = remaining_supply(board_before, racks)
    beam: List[_BeamState] = [_BeamState(remaining=dict(initial_remaining))]

    for cc in ordered:
        coord = cc.coord
        is_last_cell = cc is ordered[-1]
        candidates_here = cc.candidates[:top_labels_per_cell]
        next_beam: List[_BeamState] = []

        def _finalize(assigned, blank_cells, remaining, log_prob, lexicon_penalty, invalid_words, blanks_used):
            """Wraps a fully-formed successor state, folding in the
            main-word check here (not in a later pass) whenever this is
            the last cell -- see the function docstring."""
            if is_last_cell:
                main_penalty, main_invalid = _main_word_penalty_and_invalid(board_before, assigned, blank_cells, axis, lexicon)
                lexicon_penalty = lexicon_penalty + main_penalty
                invalid_words = invalid_words + tuple(main_invalid)
            next_beam.append(_BeamState(
                assigned=assigned, blank_cells=blank_cells, remaining=remaining, log_prob=log_prob,
                lexicon_penalty=lexicon_penalty, invalid_words=invalid_words, blanks_used=blanks_used,
            ))

        for state in beam:
            feasible_found = False
            for label, confidence in candidates_here:
                if label == CLASSIFIER_BLANK_LABEL:
                    if state.blanks_used >= max_blanks or state.remaining.get(BLANK, 0) <= 0:
                        continue
                    # The classifier's OWN confidence that this cell is a
                    # blank at all -- NOT a neutral 0. Treating "maybe
                    # blank" as free (0 log-prob) would make it beat any
                    # honestly-confident direct letter guess every time,
                    # since log(confidence) is always negative for a
                    # confidence below 1.0 -- a real bug caught by
                    # testing a plain, correctly-read "A" tile scoring
                    # worse than a low-confidence "it might be a blank"
                    # interpretation. Which SPECIFIC letter a blank is
                    # carries no additional image evidence, so every
                    # letter sub-branch shares this same base contribution.
                    blank_log_prob = math.log(confidence + _LOG_EPS)
                    for letter in string.ascii_uppercase:
                        if state.remaining.get(letter, 0) <= 0:
                            continue
                        feasible_found = True
                        penalty, invalid_word = _cross_word_check(board_before, coord, letter, True, cross_axis, lexicon)
                        new_remaining = dict(state.remaining)
                        new_remaining[BLANK] -= 1
                        _finalize(
                            {**state.assigned, coord: letter}, state.blank_cells | {coord}, new_remaining,
                            state.log_prob + blank_log_prob,
                            state.lexicon_penalty + penalty,
                            state.invalid_words + ((invalid_word,) if invalid_word else ()),
                            state.blanks_used + 1,
                        )
                else:
                    key = _pool_key(label)
                    if state.remaining.get(key, 0) <= 0:
                        continue
                    feasible_found = True
                    penalty, invalid_word = _cross_word_check(board_before, coord, label, False, cross_axis, lexicon)
                    new_remaining = dict(state.remaining)
                    new_remaining[key] -= 1
                    _finalize(
                        {**state.assigned, coord: label}, state.blank_cells, new_remaining,
                        state.log_prob + math.log(confidence + _LOG_EPS),
                        state.lexicon_penalty + penalty,
                        state.invalid_words + ((invalid_word,) if invalid_word else ()),
                        state.blanks_used,
                    )

            if not feasible_found:
                # Nothing pool-feasible for this cell from this state --
                # fall back to the raw top-1 label, unconstrained, same
                # spirit as decode_feasible_reading's own fallback. Very
                # low log-prob so this path only wins if literally
                # nothing else survives.
                label, _confidence = cc.candidates[0]
                is_blank = label == CLASSIFIER_BLANK_LABEL
                letter = "E" if is_blank else label  # a placeholder letter; never auto-published (blank/truncation gates catch this upstream)
                _finalize(
                    {**state.assigned, coord: letter}, state.blank_cells | ({coord} if is_blank else frozenset()),
                    dict(state.remaining), state.log_prob + math.log(_LOG_EPS),
                    state.lexicon_penalty, state.invalid_words, state.blanks_used,
                )

        next_beam.sort(key=lambda s: -s.score(lexicon_weight))
        beam = next_beam[:beam_width]

    results = [
        Reading(
            labels=dict(state.assigned),
            blank_cells=state.blank_cells,
            log_prob=state.log_prob,
            invalid_words=state.invalid_words,
            all_words_valid=not state.invalid_words,
            total_score=state.score(lexicon_weight),
        )
        for state in beam
    ]
    results.sort(key=lambda r: -r.total_score)
    return results
