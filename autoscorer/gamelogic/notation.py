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
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from autoscorer.gamelogic.board import BOARD_SIZE, Coord, Tile, format_square  # noqa: F401 -- format_square re-exported here for existing callers
from autoscorer.gamelogic.models import MoveType

if TYPE_CHECKING:
    from autoscorer.gamelogic.eventlog.store import GameState

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


# --- GCG export ---------------------------------------------------------
# The reverse direction: turn a live GameState's history into GCG text, the
# same format streaming graphics tooling already knows how to consume (per
# the user's own stated motivation -- this isn't a new format invented for
# this project). Only the lines a real-time feed actually needs are
# produced: per-move PLAY/EXCHANGE/PASS lines and the `#player` header
# pragmas. The final end-of-game rack bonus/penalty line (e.g.
# `>AdamLogan: (EENUV) +16 588` in the WESPA fixtures) is deliberately NOT
# produced -- this engine has no end-of-game detection to trigger it from,
# and real examples of that line don't even reduce to one obvious formula
# (see the fixture above: a plain "value of the opponent's unplayed tiles"
# computation doesn't reproduce it), so guessing at it risks a silently
# wrong number rather than an honestly absent line.


def rack_to_gcg(rack: Sequence[Tile]) -> str:
    """An unplayed blank shows as '?' (matching real GCG racks, e.g.
    "PIOTE??" for a rack holding two blanks); every other tile shows its
    (always uppercase) letter."""
    return "".join("?" if tile.letter is None else tile.letter for tile in rack)


def format_position(word_cells: Sequence[Coord]) -> str:
    """Inverse of `parse_position`: digit-first ("8G") reads across, single
    row; letter-first ("E8") reads down, single column. A one-cell word
    (only possible as an opening single-letter play) defaults to across,
    matching standard tournament notation for that case."""
    row, col = word_cells[0]
    col_letter = chr(ord("A") + col)
    is_down = len(word_cells) > 1 and len({c for _, c in word_cells}) == 1
    return f"{col_letter}{row + 1}" if is_down else f"{row + 1}{col_letter}"


def format_word_for_gcg(word_cells: Sequence[Coord], word_text: str, new_cells: Sequence[Coord], blank_cells: Sequence[Coord]) -> str:
    """One char per cell in `word_cells`: '.' for a pre-existing cell the
    play hooks through, else the letter newly placed there (lowercase if
    that cell was played as a blank) -- matches the GCG play-line word
    grammar `parse_gcg_word` already reads."""
    new_set = set(new_cells)
    blank_set = set(blank_cells)
    chars = []
    for coord, letter in zip(word_cells, word_text):
        if coord not in new_set:
            chars.append(".")
        else:
            chars.append(letter.lower() if coord in blank_set else letter)
    return "".join(chars)


def format_gcg_move_line(player_token: str, rack_before: Sequence[Tile], position: str, word: str, turn_score: int, cumulative_score: int) -> str:
    return f">{player_token}: {rack_to_gcg(rack_before)} {position} {word} +{turn_score} {cumulative_score}"


def format_gcg_exchange_line(player_token: str, rack_before: Sequence[Tile], exchanged_tiles: Sequence[Tile], cumulative_score: int) -> str:
    return f">{player_token}: {rack_to_gcg(rack_before)} -{rack_to_gcg(exchanged_tiles)} +0 {cumulative_score}"


def format_gcg_pass_line(player_token: str, rack_before: Sequence[Tile], cumulative_score: int) -> str:
    return f">{player_token}: {rack_to_gcg(rack_before)} - +0 {cumulative_score}"


def export_gcg(game_state: "GameState", player_names: Optional[Dict[str, str]] = None, description: str = "") -> str:
    """Renders a live `GameState`'s move history as GCG text. `player_names`
    maps each `player_id` (the token used in every `>player_id: ...` line,
    same identity `GameSession.submit_move` already takes) to a display
    name for the `#player` header pragma -- falls back to the id itself
    when not given, since nothing upstream currently tracks display names
    separately from ids.
    """
    player_names = player_names or {}
    lines: List[str] = []
    if description:
        lines.append(f"#description {description}")

    seen_players: List[str] = []
    for scored_move in game_state.history:
        pid = scored_move.candidate.player_id
        if pid not in seen_players:
            seen_players.append(pid)
    for i, pid in enumerate(seen_players, start=1):
        lines.append(f"#player{i} {pid} {player_names.get(pid, pid)}")

    cumulative: Dict[str, int] = {}
    for scored_move in game_state.history:
        candidate = scored_move.candidate
        pid = candidate.player_id
        move_type = candidate.move_type

        if move_type == MoveType.PLAY:
            word_score = scored_move.move_score.words[0]
            position = format_position(word_score.cells)
            word = format_word_for_gcg(word_score.cells, word_score.text, candidate.new_cells, candidate.blank_cells)
            cumulative[pid] = cumulative.get(pid, 0) + scored_move.move_score.total
            lines.append(format_gcg_move_line(
                pid, candidate.rack_before, position, word, scored_move.move_score.total, cumulative[pid],
            ))
        elif move_type == MoveType.EXCHANGE:
            lines.append(format_gcg_exchange_line(
                pid, candidate.rack_before, candidate.exchanged_tiles, cumulative.get(pid, 0),
            ))
        else:  # PASS
            lines.append(format_gcg_pass_line(pid, candidate.rack_before, cumulative.get(pid, 0)))

    return "\n".join(lines) + "\n"
