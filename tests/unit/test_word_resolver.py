from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.movedetect.word_resolver import word_text, words_formed


def test_first_move_single_word_no_cross_words():
    board_before = BoardState()
    new_cells = [(7, 6), (7, 7), (7, 8)]
    board_after = board_before.with_placements({
        (7, 6): Tile("C"), (7, 7): Tile("A"), (7, 8): Tile("T"),
    })

    words = words_formed(board_after, new_cells)

    assert len(words) == 1
    assert word_text(board_after, words[0]) == "CAT"


def test_single_new_tile_extends_existing_word():
    board_before = BoardState({
        (7, 6): Tile("C"), (7, 7): Tile("A"), (7, 8): Tile("T"),
    })
    new_cells = [(7, 9)]
    board_after = board_before.with_placements({(7, 9): Tile("S")})

    words = words_formed(board_after, new_cells)

    assert len(words) == 1
    assert word_text(board_after, words[0]) == "CATS"


def test_single_new_tile_forms_only_perpendicular_cross_word():
    board_before = BoardState({
        (7, 6): Tile("C"), (7, 7): Tile("A"), (7, 8): Tile("T"),
    })
    new_cells = [(8, 7)]
    board_after = board_before.with_placements({(8, 7): Tile("S")})

    words = words_formed(board_after, new_cells)

    assert len(words) == 1
    assert word_text(board_after, words[0]) == "AS"


def test_single_new_tile_forms_both_horizontal_and_vertical_words():
    # Existing "A" at (7,7) and "S" at (8,7); placing "T" at (7,8) is a
    # separate scenario -- here we place a new tile that simultaneously
    # completes a horizontal and a vertical word.
    board_before = BoardState({
        (7, 6): Tile("C"), (7, 8): Tile("T"),  # gap at (7,7)
        (6, 7): Tile("B"),
    })
    new_cells = [(7, 7)]
    board_after = board_before.with_placements({(7, 7): Tile("A")})

    words = words_formed(board_after, new_cells)
    texts = {word_text(board_after, w) for w in words}

    assert texts == {"CAT", "BA"}


def test_multi_tile_placement_with_main_and_cross_word():
    board_before = BoardState({(8, 7): Tile("S")})
    new_cells = [(7, 6), (7, 7), (7, 8)]
    board_after = board_before.with_placements({
        (7, 6): Tile("C"), (7, 7): Tile("A"), (7, 8): Tile("T"),
    })

    words = words_formed(board_after, new_cells)
    texts = {word_text(board_after, w) for w in words}

    assert texts == {"CAT", "AS"}


def test_isolated_single_tile_first_move_forms_length_one_word():
    board_before = BoardState()
    new_cells = [(7, 7)]
    board_after = board_before.with_placements({(7, 7): Tile("I")})

    words = words_formed(board_after, new_cells)

    assert len(words) == 1
    assert word_text(board_after, words[0]) == "I"
