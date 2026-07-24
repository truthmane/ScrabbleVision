from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.scoring.rules_engine import BINGO_BONUS, score_move, score_word


def test_plain_word_no_premiums():
    # Row 7, columns 4-6 are all premium-free per the standard board layout.
    # A, N, T are each worth 1 point, keeping the arithmetic simple.
    board = BoardState().with_placements({
        (7, 4): Tile("A"), (7, 5): Tile("N"), (7, 6): Tile("T"),
    })
    word = [(7, 4), (7, 5), (7, 6)]
    new_cells = {(7, 4), (7, 5), (7, 6)}

    result = score_word(board, word, new_cells)

    assert result.text == "ANT"
    assert result.score == 1 + 1 + 1


def test_double_letter_premium_applies_only_to_new_tile():
    # (7, 3) is a double-letter square.
    board = BoardState().with_placements({(7, 3): Tile("Z")})
    result = score_word(board, [(7, 3)], new_cells={(7, 3)})
    assert result.score == 10 * 2  # Z=10, doubled


def test_triple_letter_premium():
    # (5, 1) is a triple-letter square.
    board = BoardState().with_placements({(5, 1): Tile("K")})
    result = score_word(board, [(5, 1)], new_cells={(5, 1)})
    assert result.score == 5 * 3  # K=5, tripled


def test_double_word_premium_multiplies_whole_word():
    # (1, 1) is a double-word square. A, N, T are each worth 1 point.
    board = BoardState().with_placements({
        (1, 1): Tile("A"), (1, 2): Tile("N"), (1, 3): Tile("T"),
    })
    word = [(1, 1), (1, 2), (1, 3)]
    result = score_word(board, word, new_cells=set(word))
    assert result.score == (1 + 1 + 1) * 2


def test_triple_word_premium():
    # (0, 0) is a triple-word square. A, N, T are each worth 1 point.
    board = BoardState().with_placements({
        (0, 0): Tile("A"), (0, 1): Tile("N"), (0, 2): Tile("T"),
    })
    word = [(0, 0), (0, 1), (0, 2)]
    result = score_word(board, word, new_cells=set(word))
    assert result.score == (1 + 1 + 1) * 3


def test_premium_square_already_consumed_by_earlier_move_does_not_reapply():
    """The single most important correctness case: a premium square only
    ever pays out the first time a tile lands on it. Here (7,7) is the
    center double-word square, already occupied from an earlier move --
    a later word merely passing through it must NOT get the multiplier
    again.
    """
    # Board *after* this move: "ANT" bridges over the pre-existing "N" at
    # the center square. Only (7,6) and (7,8) are new this turn.
    board_after = BoardState({(7, 7): Tile("N")}).with_placements({
        (7, 6): Tile("A"), (7, 8): Tile("T"),
    })
    word = [(7, 6), (7, 7), (7, 8)]
    new_cells = {(7, 6), (7, 8)}  # (7,7) was already occupied -- not new

    result = score_word(board_after, word, new_cells)

    # No double-word multiplier applied: 1 (A) + 1 (N, no premium) + 1 (T) = 3
    assert result.score == 3


def test_bingo_bonus_added_when_seven_tiles_placed():
    # Seven premium-free cells in row 4 (col 4 and col 10 are the only
    # premiums in that row) -- values chosen for a simple, checkable sum.
    cells = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7)]
    letters = ["P", "L", "A", "Y", "E", "R", "S"]
    board = BoardState().with_placements({c: Tile(l) for c, l in zip(cells, letters)})

    move_score = score_move(board, words=[cells], new_cells=cells)

    letter_total = 3 + 1 + 1 + 4 + 1 + 1 + 1  # P L A Y E R S
    assert move_score.is_bingo
    assert move_score.total == letter_total + BINGO_BONUS


def test_six_tile_placement_does_not_trigger_bingo():
    cells = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6)]
    board = BoardState().with_placements({c: Tile("A") for c in cells})
    move_score = score_move(board, words=[cells], new_cells=cells)
    assert not move_score.is_bingo
    assert move_score.total == 6  # six A's at 1 point each, no premiums


def test_blank_tile_scores_zero_regardless_of_letter_or_premium():
    # (5, 1) is a triple-letter square; a blank played there must still
    # score 0, not 3x the letter it represents.
    board = BoardState().with_placements({(5, 1): Tile("Z", is_blank=True)})
    result = score_word(board, [(5, 1)], new_cells={(5, 1)})
    assert result.score == 0
