from autoscorer.api.session import GameSession
from autoscorer.gamelogic.board import Tile
from autoscorer.gamelogic.publish import PublishMode
from autoscorer.gamelogic.notation import (
    export_gcg,
    format_gcg_exchange_line,
    format_gcg_pass_line,
    format_position,
    format_word_for_gcg,
    parse_gcg_line,
    rack_to_gcg,
    resolve_new_tiles_gcg,
)


def test_rack_to_gcg_preserves_order_and_blank_marker():
    rack = [Tile("P"), Tile(None, is_blank=True), Tile("E")]
    assert rack_to_gcg(rack) == "P?E"


def test_format_position_across_for_a_single_row_multi_cell_word():
    assert format_position([(7, 7), (7, 8), (7, 9)]) == "8H"


def test_format_position_down_for_a_single_column_multi_cell_word():
    assert format_position([(6, 9), (7, 9)]) == "J7"


def test_format_position_single_cell_word_defaults_to_across():
    assert format_position([(7, 7)]) == "8H"


def test_format_word_for_gcg_dots_out_pre_existing_cells_and_lowercases_blanks():
    word_cells = [(6, 9), (7, 9)]
    word_text = "ZT"
    new_cells = [(6, 9)]
    blank_cells = [(6, 9)]
    assert format_word_for_gcg(word_cells, word_text, new_cells, blank_cells) == "z."


def test_format_gcg_exchange_and_pass_lines():
    rack = [Tile("D"), Tile("O"), Tile("G")]
    exchanged = [Tile("D")]
    assert format_gcg_exchange_line("Alice", rack, exchanged, 42) == ">Alice: DOG -D +0 42"
    assert format_gcg_pass_line("Alice", rack, 42) == ">Alice: DOG - +0 42"


def test_export_gcg_full_sequence_round_trips_through_the_parser():
    session = GameSession(mode=PublishMode.AUTONOMOUS)
    session.game_state.racks["Alice"] = [Tile("C"), Tile("A"), Tile("T"), Tile("D"), Tile("O"), Tile("G"), Tile("S")]
    session.game_state.racks["Bob"] = [Tile("P"), Tile("E"), Tile("X"), Tile("Y"), Tile("Z"), Tile("Q"), Tile("W")]

    # Move 1: Alice opens with CAT across the center square.
    r1 = session.submit_move(
        "Alice",
        new_tiles=[((7, 7), "C", False), ((7, 8), "A", False), ((7, 9), "T", False)],
        rack_after=[("D", False), ("O", False), ("G", False), ("S", False), ("E", False), ("N", False), ("I", False)],
    )
    assert r1.published

    # Move 2: Bob hooks "APE" down through the existing A.
    r2 = session.submit_move(
        "Bob",
        new_tiles=[((8, 8), "P", False), ((9, 8), "E", False)],
        rack_after=[("X", False), ("Y", False), ("Z", False), ("Q", False), ("W", False), ("R", False), ("U", False)],
    )
    assert r2.published

    # Move 3: Alice extends CAT into CATS -- her rack_before here should be
    # exactly move 1's rack_after (continuity across her own turns).
    assert tuple(session.game_state.racks["Alice"]) == (
        Tile("D"), Tile("O"), Tile("G"), Tile("S"), Tile("E"), Tile("N"), Tile("I"),
    )
    r3 = session.submit_move(
        "Alice",
        new_tiles=[((7, 10), "S", False)],
        rack_after=[("D", False), ("O", False), ("G", False), ("E", False), ("N", False), ("I", False), ("R", False)],
    )
    assert r3.published

    # Move 4: Bob plays a blank as Z, forming "ZT" down through the T in CATS.
    r4 = session.submit_move(
        "Bob",
        new_tiles=[((6, 9), "Z", True)],
        rack_after=[("Y", False), ("Q", False), ("W", False), ("R", False), ("U", False), ("L", False), ("M", False)],
    )
    assert r4.published

    # Move 5: Alice exchanges a single tile (D for U).
    r5 = session.submit_move(
        "Alice",
        rack_after=[("U", False), ("O", False), ("G", False), ("E", False), ("N", False), ("I", False), ("R", False)],
    )
    assert r5.published

    # Move 6: Bob passes -- no board or rack change.
    r6 = session.submit_move("Bob")
    assert r6.published

    text = export_gcg(
        session.game_state,
        player_names={"Alice": "Alice Smith", "Bob": "Bob Jones"},
        description="Test Game",
    )
    lines = text.splitlines()

    assert lines[0] == "#description Test Game"
    assert lines[1] == "#player1 Alice Alice Smith"
    assert lines[2] == "#player2 Bob Bob Jones"

    move_lines = lines[3:]
    assert len(move_lines) == 6

    # PLAY lines round-trip through the existing GCG parser, and reproduce
    # exactly the tiles that were actually submitted.
    expected_new_tiles = {
        0: [((7, 7), "C", False), ((7, 8), "A", False), ((7, 9), "T", False)],
        1: [((8, 8), "P", False), ((9, 8), "E", False)],
        2: [((7, 10), "S", False)],
        3: [((6, 9), "Z", True)],
    }
    for i, expected in expected_new_tiles.items():
        parsed = parse_gcg_line(move_lines[i])
        assert parsed is not None, f"line {i} failed to parse as a GCG play: {move_lines[i]!r}"
        placements = resolve_new_tiles_gcg(parsed)
        actual = [(p.coord, p.letter, p.is_blank) for p in placements]
        assert actual == expected

    # Exchange and pass lines are not parsed as plays by parse_gcg_line.
    assert parse_gcg_line(move_lines[4]) is None
    assert move_lines[4] == ">Alice: DOGENIR -D +0 " + str(session.game_state.scores["Alice"])
    assert parse_gcg_line(move_lines[5]) is None
    assert move_lines[5] == ">Bob: YQWRULM - +0 " + str(session.game_state.scores["Bob"])

    # Cumulative scores on the PLAY lines match the final tracked scores
    # (Alice's last score-changing move was move 3; Bob's was move 4).
    assert move_lines[2].endswith(f"+{r3.outcome.move_score.total} {session.game_state.scores['Alice']}")
    assert move_lines[3].endswith(f"+{r4.outcome.move_score.total} {session.game_state.scores['Bob']}")
