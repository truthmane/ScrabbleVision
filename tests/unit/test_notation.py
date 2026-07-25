import pytest

from autoscorer.gamelogic.notation import (
    parse_position,
    parse_ticker_line,
    parse_word,
    resolve_new_tiles,
)


def test_parse_ticker_line_extracts_all_fields():
    move = parse_ticker_line("Krafchick, Joey J7 TaSKING 86 107")
    assert move.player == "Krafchick, Joey"
    assert move.position == "J7"
    assert move.word == "TaSKING"
    assert move.turn_score == 86
    assert move.cumulative_score == 107


def test_parse_ticker_line_strips_trailing_definition():
    move = parse_ticker_line("Reinke, Thomas E2 VODU(N) 18 125 | VODU, to practice voodoo on")
    assert move.position == "E2"
    assert move.word == "VODU(N)"
    assert move.turn_score == 18
    assert move.cumulative_score == 125


def test_parse_ticker_line_rejects_garbage():
    with pytest.raises(ValueError):
        parse_ticker_line("not a valid ticker line at all")


def test_position_number_first_is_across():
    coord, direction = parse_position("14B")
    assert coord == (13, 1)
    assert direction == "across"


def test_position_letter_first_is_down():
    coord, direction = parse_position("J7")
    assert coord == (6, 9)
    assert direction == "down"


def test_position_out_of_bounds_rejected():
    with pytest.raises(ValueError):
        parse_position("20A")


def test_parse_word_marks_parenthesized_letters_as_not_new():
    letters = parse_word("VODU(N)")
    assert letters == [
        ("V", True, False),
        ("O", True, False),
        ("D", True, False),
        ("U", True, False),
        ("N", False, False),
    ]


def test_parse_word_lowercase_is_blank():
    letters = parse_word("TaSKING")
    assert letters[1] == ("A", True, True)
    assert all(not is_blank for _, _, is_blank in [letters[0]] + letters[2:])


def test_parse_word_ignores_bingo_marker():
    letters = parse_word("EXE(A)T#")
    assert [l for l, _, _ in letters] == ["E", "X", "E", "A", "T"]
    assert [is_new for _, is_new, _ in letters] == [True, True, True, False, True]


def test_resolve_new_tiles_down_word_skips_existing_letter():
    move = parse_ticker_line("Reinke, Thomas E2 VODU(N) 18 125")
    placements = resolve_new_tiles(move)
    # E2 -> col E (4), row 1 (0-indexed); down means row increases.
    assert [p.coord for p in placements] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert [p.letter for p in placements] == ["V", "O", "D", "U"]
    assert not any(p.is_blank for p in placements)


def test_resolve_new_tiles_across_word_with_blank():
    move = parse_ticker_line("Krafchick, Joey J7 TaSKING 86 107")
    placements = resolve_new_tiles(move)
    # J7 -> letter-first means DOWN, col J (9), row 6 (0-indexed).
    assert [p.coord for p in placements] == [(6 + i, 9) for i in range(7)]
    assert [p.letter for p in placements] == list("TASKING")
    assert [p.is_blank for p in placements] == [False, True, False, False, False, False, False]
