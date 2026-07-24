"""Official Scrabble scoring: letter values, premium-square multipliers, and
the bingo bonus. Pure arithmetic over a resolved board state and word list --
no CV, no camera-derived confidence.

Key correctness point: a premium square only ever multiplies the tile that is
*newly placed* on it. Because tiles are never removed from the board in
normal play, "this square hasn't been used yet" is exactly equivalent to
"this cell was empty before this move" -- so no separate persistent
first-use flag is needed. Passing only `new_cells` (the cells that changed
this turn) as the set eligible for a multiplier is sufficient and correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set

from autoscorer.gamelogic.board import LETTER_VALUES, PREMIUM_SQUARES, BoardState, Coord

BINGO_BONUS = 50
BINGO_TILE_COUNT = 7


@dataclass(frozen=True)
class WordScore:
    cells: List[Coord]
    text: str
    score: int


@dataclass(frozen=True)
class MoveScore:
    words: List[WordScore]
    is_bingo: bool
    total: int


def score_word(board_after: BoardState, word_cells: Sequence[Coord], new_cells: Set[Coord]) -> WordScore:
    letter_total = 0
    word_multiplier = 1
    letters = []

    for coord in word_cells:
        tile = board_after.get(coord)
        assert tile is not None, f"score_word called with unoccupied cell {coord}"
        letters.append(tile.letter)

        value = 0 if tile.is_blank else LETTER_VALUES[tile.letter]

        if coord in new_cells:
            premium = PREMIUM_SQUARES.get(coord)
            if premium == "2L":
                value *= 2
            elif premium == "3L":
                value *= 3
            elif premium == "2W":
                word_multiplier *= 2
            elif premium == "3W":
                word_multiplier *= 3

        letter_total += value

    return WordScore(cells=list(word_cells), text="".join(letters), score=letter_total * word_multiplier)


def score_move(
    board_after: BoardState,
    words: Sequence[Sequence[Coord]],
    new_cells: Sequence[Coord],
) -> MoveScore:
    """Score every word formed by a single move. `new_cells` are the cells
    that changed this turn (used to gate premium-square eligibility);
    `words` is the full set of words formed, as produced by
    `word_resolver.words_formed`.
    """
    new_set = set(new_cells)
    word_scores = [score_word(board_after, word, new_set) for word in words]
    is_bingo = len(new_cells) == BINGO_TILE_COUNT
    total = sum(w.score for w in word_scores) + (BINGO_BONUS if is_bingo else 0)
    return MoveScore(words=word_scores, is_bingo=is_bingo, total=total)
