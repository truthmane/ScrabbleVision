"""Parses tournament-broadcast move-ticker notation and resolves it into
the same `new_tiles` shape `GameSession.submit_move` already accepts.

This is the core of WS3 (ticker-replay ground truth scaling, see
`docs/classifier-accuracy-plan.md`): transcribing a game's ticker lines and
replaying them through the already-tested rules engine reconstructs the
exact board state at every turn -- including which letter each blank was
played as -- without reading a single pixel. That reconstructed state can
then auto-label harvested frames instead of a human reading an axis table
cell by cell, and doubles as a free regression test of the scoring engine
against real official tournament scores, which nothing has exercised
before now.

Ticker line format, e.g.:
    "Krafchick, Joey J7 TaSKING 86 107"
    "Reinke, Thomas E2 VODU(N) 18 125 | VODU, to practice voodoo on"

    <player last, first> <position><WORD> <turn_score> <cumulative_score> [| definition]

Position: a coordinate starting with a digit (e.g. "14B") reads ACROSS;
one starting with a letter (e.g. "J7") reads DOWN. This is standard
tournament notation (cross-tables.com, NASPA broadcasts).

Word: letters in parentheses were already on the board before this move
(the play hooks onto/extends an existing word) -- only letters outside
parentheses are newly placed tiles this turn. A lowercase letter denotes
a blank tile played as that letter. Any other punctuation (a trailing
bingo marker, etc.) is ignored; the bingo bonus is derived by the rules
engine from the new-tile count, same as everywhere else.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from autoscorer.gamelogic.board import BOARD_SIZE, Coord

_POSITION_ACROSS = re.compile(r"^(\d{1,2})([A-Oa-o])$")
_POSITION_DOWN = re.compile(r"^([A-Oa-o])(\d{1,2})$")

_LINE_RE = re.compile(
    r"^(?P<player>[^,]+,\s*\S+)\s+"
    r"(?P<pos>[A-Oa-o]\d{1,2}|\d{1,2}[A-Oa-o])\s+"
    r"(?P<word>[A-Za-z()#.]+)\s+"
    r"(?P<turn_score>\d+)\s+"
    r"(?P<cum_score>\d+)"
    r"(?:\s*\|.*)?$"
)


@dataclass(frozen=True)
class TranscribedMove:
    player: str
    position: str
    word: str
    turn_score: int
    cumulative_score: int


@dataclass(frozen=True)
class NewTilePlacement:
    coord: Coord
    letter: str
    is_blank: bool


def parse_ticker_line(line: str) -> TranscribedMove:
    m = _LINE_RE.match(line.strip())
    if not m:
        raise ValueError(f"could not parse ticker line: {line!r}")
    return TranscribedMove(
        player=m.group("player").strip(),
        position=m.group("pos"),
        word=m.group("word"),
        turn_score=int(m.group("turn_score")),
        cumulative_score=int(m.group("cum_score")),
    )


def parse_position(pos: str) -> Tuple[Coord, str]:
    """Returns (start_coord, direction) with direction 'across' or 'down',
    as a 0-indexed (row, col) matching board.py's Coord convention.
    """
    m = _POSITION_ACROSS.match(pos)
    if m:
        row = int(m.group(1)) - 1
        col = ord(m.group(2).upper()) - ord("A")
        direction = "across"
    else:
        m = _POSITION_DOWN.match(pos)
        if not m:
            raise ValueError(f"unrecognized position notation: {pos!r}")
        col = ord(m.group(1).upper()) - ord("A")
        row = int(m.group(2)) - 1
        direction = "down"

    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"position {pos!r} resolves outside the board: row={row} col={col}")
    return (row, col), direction


def parse_word(word: str) -> List[Tuple[str, bool, bool]]:
    """Returns one (letter_upper, is_new_this_turn, is_blank) tuple per
    board cell the word occupies, in order along the play direction.
    """
    result = []
    in_parens = False
    for ch in word:
        if ch == "(":
            in_parens = True
            continue
        if ch == ")":
            in_parens = False
            continue
        if not ch.isalpha():
            continue  # bingo marker or other punctuation, not a tile
        result.append((ch.upper(), not in_parens, ch.islower()))
    return result


def resolve_new_tiles(move: TranscribedMove) -> List[NewTilePlacement]:
    """The subset of `move`'s word that is newly placed this turn, as board
    coordinates -- exactly what `GameSession.submit_move`'s `new_tiles`
    expects. Cells shown in parentheses (pre-existing tiles the play hooks
    onto) are skipped; the rules engine only needs to see what changed.
    """
    (row, col), direction = parse_position(move.position)
    placements = []
    for letter, is_new, is_blank in parse_word(move.word):
        if is_new:
            placements.append(NewTilePlacement((row, col), letter, is_blank))
        if direction == "across":
            col += 1
        else:
            row += 1
    return placements


# --- GCG format -------------------------------------------------------
# GCG ("Game Coordinator Games") is the plain-text format NASPA and other
# tournament bodies publish official game records in -- e.g.
# https://event.scrabbleplayers.org publishes one per tournament game. It's
# strictly better ground truth than a hand-transcribed broadcast ticker:
# machine-generated, unambiguous, and requires no video-watching labor.
#
# A play line looks like:
#     >Orry: ATSLKCI 8G LICK +20 20
#     >Mack: ENDHOEA G4 ENHA.OED +64 64
#
# <player>: <rack before the move> <position> <word> +<turn score> <cumulative score>
#
# The word uses '.' (not parentheses) for each pre-existing cell the play
# passes through -- one '.' per skipped cell, identity irrelevant since the
# replay already knows the board from prior moves. Lowercase = blank played
# as that letter, same as the ticker format. Lines that don't fit this
# shape (passes, exchanges, challenges, the end-of-game rack bonus/penalty
# line) are deliberately left unparsed here -- our rules engine doesn't
# model those anyway, and the replay is still valid for every PLAY line.
_GCG_LINE_RE = re.compile(
    r"^>(?P<player>[^:]+):\s+"
    r"(?P<rack>[A-Z?]*)\s+"
    r"(?P<pos>[A-Oa-o]\d{1,2}|\d{1,2}[A-Oa-o])\s+"
    r"(?P<word>[A-Za-z.]+)\s+"
    r"\+(?P<turn_score>-?\d+)\s+"
    r"(?P<cum_score>-?\d+)\s*$"
)


@dataclass(frozen=True)
class GcgMove:
    player: str
    rack: str
    position: str
    word: str
    turn_score: int
    cumulative_score: int


def parse_gcg_line(line: str) -> Optional[GcgMove]:
    """Returns `None` (rather than raising) for lines that aren't a
    standard play -- passes, exchanges, and the end-of-game bonus/penalty
    line all use different shapes; callers should skip those lines."""
    m = _GCG_LINE_RE.match(line.strip())
    if not m:
        return None
    return GcgMove(
        player=m.group("player").strip(),
        rack=m.group("rack"),
        position=m.group("pos"),
        word=m.group("word"),
        turn_score=int(m.group("turn_score")),
        cumulative_score=int(m.group("cum_score")),
    )


def parse_gcg_word(word: str) -> List[Tuple[Optional[str], bool, bool]]:
    """One entry per board cell in the play's run: '.' (a pre-existing
    cell) yields (None, False, False); a letter yields
    (letter_upper, True, is_blank)."""
    result: List[Tuple[Optional[str], bool, bool]] = []
    for ch in word:
        if ch == ".":
            result.append((None, False, False))
        else:
            result.append((ch.upper(), True, ch.islower()))
    return result


def resolve_new_tiles_gcg(move: GcgMove) -> List[NewTilePlacement]:
    (row, col), direction = parse_position(move.position)
    placements = []
    for letter, is_new, is_blank in parse_gcg_word(move.word):
        if is_new:
            placements.append(NewTilePlacement((row, col), letter, is_blank))
        if direction == "across":
            col += 1
        else:
            row += 1
    return placements
