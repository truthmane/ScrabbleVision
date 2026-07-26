from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.dictionary.lexicon import Lexicon, LexiconInfo
from autoscorer.gamelogic.movedetect.constraint_decoder import CellCandidates, decode_feasible_reading
from autoscorer.gamelogic.movedetect.lexicon_decoder import decode_with_lexicon


def _fake_lexicon(words) -> Lexicon:
    words = frozenset(w.upper() for w in words)
    return Lexicon(words, LexiconInfo(name="fake", path="<memory>", word_count=len(words), is_default=False))


# --- Contract: with no lexicon, ranking is by pool-feasible log-prob
# alone, matching decode_feasible_reading -- except for blanks, which
# decode_with_lexicon deliberately handles differently (it always tries
# to resolve a specific letter; decode_feasible_reading passes "BLANK"
# straight through). That's by design, not a bug -- tested separately
# below. ---

def test_uncontested_candidates_all_get_their_top_choice():
    board = BoardState()
    candidates = [
        CellCandidates(coord=(7, 6), candidates=[("A", 0.9), ("R", 0.05)]),
        CellCandidates(coord=(7, 7), candidates=[("N", 0.8), ("M", 0.1)]),
    ]
    readings = decode_with_lexicon(candidates, board, racks=[[]], lexicon=None, lexicon_weight=0.0)
    assert readings[0].labels == {(7, 6): "A", (7, 7): "N"}
    assert readings[0].labels == decode_feasible_reading(candidates, board, racks=[[]])


def test_scarce_letter_goes_to_the_more_confident_cell():
    board = BoardState()
    candidates = [
        CellCandidates(coord=(7, 6), candidates=[("Q", 0.55), ("O", 0.2)]),
        CellCandidates(coord=(7, 7), candidates=[("Q", 0.9), ("O", 0.05)]),
    ]
    readings = decode_with_lexicon(candidates, board, racks=[[]], lexicon=None, lexicon_weight=0.0)
    assert readings[0].labels == decode_feasible_reading(candidates, board, racks=[[]])
    assert readings[0].labels[(7, 7)] == "Q"
    assert readings[0].labels[(7, 6)] == "O"


def test_falls_back_to_top_choice_when_nothing_in_topk_is_feasible():
    board = BoardState({(0, 0): Tile("Q")})
    candidates = [CellCandidates(coord=(7, 7), candidates=[("Q", 0.6)])]
    readings = decode_with_lexicon(candidates, board, racks=[[]], lexicon=None, lexicon_weight=0.0)
    assert readings[0].labels == decode_feasible_reading(candidates, board, racks=[[]])
    assert readings[0].labels[(7, 7)] == "Q"


def test_blank_is_resolved_to_a_real_letter_not_passed_through():
    """Deliberately different from decode_feasible_reading (which passes
    the raw "BLANK" label straight through): decode_with_lexicon always
    tries to resolve a blank to a specific, pool-feasible letter, since
    that's the whole point of lexicon-constrained decoding -- the image
    itself carries zero information about which letter a blank
    represents, so this can only ever come from pool feasibility (here)
    or the lexicon (see the dedicated blank-recovery test below)."""
    board = BoardState()
    candidates = [CellCandidates(coord=(7, 7), candidates=[("BLANK", 0.7), ("O", 0.1)])]
    readings = decode_with_lexicon(candidates, board, racks=[[]], lexicon=None, lexicon_weight=0.0)
    letter = readings[0].labels[(7, 7)]
    assert letter != "BLANK"
    assert letter.isalpha() and len(letter) == 1
    assert (7, 7) in readings[0].blank_cells


# --- Lexicon-specific behavior ---

def test_never_invents_a_reading_when_no_word_is_valid():
    board = BoardState()
    lexicon = _fake_lexicon(["CAT", "DOG"])
    candidates = [
        CellCandidates(coord=(7, 6), candidates=[("Z", 0.9)]),
        CellCandidates(coord=(7, 7), candidates=[("Q", 0.9)]),
        CellCandidates(coord=(7, 8), candidates=[("X", 0.9)]),
    ]
    readings = decode_with_lexicon(candidates, board, racks=[[]], lexicon=lexicon)
    assert readings[0].labels == {(7, 6): "Z", (7, 7): "Q", (7, 8): "X"}
    assert readings[0].all_words_valid is False
    assert "ZQX" in readings[0].invalid_words


def test_a_confident_direct_letter_guess_beats_a_low_confidence_blank_guess():
    """Regression test for a real scoring bug: the blank branch must be
    scored using the classifier's OWN confidence that the cell is a
    blank at all, not a neutral 0 -- a neutral 0 log-prob would beat
    ANY honestly-confident direct letter guess, since log(confidence) is
    always negative for a confidence below 1.0. Caught by GameWatcher's
    own test suite: a single, cleanly-read "A" tile was losing to a
    low-confidence "maybe it's a blank" interpretation on every call.
    """
    board = BoardState()
    candidates = [CellCandidates(coord=(7, 7), candidates=[("A", 0.9), ("BLANK", 0.05)])]
    readings = decode_with_lexicon(candidates, board, racks=[[]], lexicon=None)
    assert readings[0].labels[(7, 7)] == "A"
    assert (7, 7) not in readings[0].blank_cells


def test_lexicon_can_overturn_a_moderately_confident_misread():
    # (7,6)/(7,7)/(7,8) top guesses spell "CAR", but "CAT" is only
    # slightly less confident at the last cell and IS a real word.
    board = BoardState()
    lexicon = _fake_lexicon(["CAT"])
    candidates = [
        CellCandidates(coord=(7, 6), candidates=[("C", 0.9)]),
        CellCandidates(coord=(7, 7), candidates=[("A", 0.9)]),
        CellCandidates(coord=(7, 8), candidates=[("R", 0.55), ("T", 0.45)]),
    ]
    readings = decode_with_lexicon(candidates, board, racks=[[]], lexicon=lexicon, lexicon_weight=2.0)
    assert readings[0].labels == {(7, 6): "C", (7, 7): "A", (7, 8): "T"}
    assert readings[0].all_words_valid is True


def test_a_confident_reading_beats_the_lexicon_protecting_phonies():
    """The lexicon must never overturn a HIGH-confidence reading just
    because it isn't a real word -- phonies are legal, scoring plays,
    and this is the mechanism that keeps them readable."""
    board = BoardState()
    lexicon = _fake_lexicon(["CAR"])  # "CAT" not in this lexicon at all
    candidates = [
        CellCandidates(coord=(7, 6), candidates=[("C", 0.97)]),
        CellCandidates(coord=(7, 7), candidates=[("A", 0.97)]),
        CellCandidates(coord=(7, 8), candidates=[("T", 0.95), ("R", 0.02)]),
    ]
    readings = decode_with_lexicon(candidates, board, racks=[[]], lexicon=lexicon, lexicon_weight=2.0)
    assert readings[0].labels == {(7, 6): "C", (7, 7): "A", (7, 8): "T"}
    assert readings[0].all_words_valid is False
    assert "CAT" in readings[0].invalid_words


def test_cross_word_validity_is_checked_incrementally():
    # An existing "DO" runs down column 6 from row 6; placing "C" at
    # (7,6) as part of a horizontal word also forms the cross-word "DOC".
    board = BoardState({(6, 6): Tile("D"), (7, 6): Tile("O")})
    # New horizontal word at row 8: (8,6)="C", forming vertical DOC, and
    # (8,7)="T".
    lexicon = _fake_lexicon(["DOC", "CT"])
    candidates = [
        CellCandidates(coord=(8, 6), candidates=[("C", 0.9)]),
        CellCandidates(coord=(8, 7), candidates=[("T", 0.9)]),
    ]
    readings = decode_with_lexicon(candidates, board, racks=[[]], lexicon=lexicon)
    assert readings[0].labels == {(8, 6): "C", (8, 7): "T"}
    assert readings[0].all_words_valid is True


def test_blank_recovery_uses_the_lexicon_when_confidence_gives_no_signal():
    board = BoardState()
    lexicon = _fake_lexicon(["CAT"])
    candidates = [
        CellCandidates(coord=(7, 6), candidates=[("C", 0.9)]),
        CellCandidates(coord=(7, 7), candidates=[("A", 0.9)]),
        CellCandidates(coord=(7, 8), candidates=[("BLANK", 0.6)]),
    ]
    readings = decode_with_lexicon(candidates, board, racks=[[]], lexicon=lexicon, max_blanks=1)
    assert readings[0].labels[(7, 8)] == "T"
    assert (7, 8) in readings[0].blank_cells
    assert readings[0].all_words_valid is True


def test_respects_pool_feasibility_for_blank_letter_choices():
    # Only 1 Q exists and it's already on the board -- a blank must
    # never be resolved to a letter the pool can't actually support.
    board = BoardState({(0, 0): Tile("Q")})
    candidates = [CellCandidates(coord=(7, 7), candidates=[("BLANK", 0.6)])]
    readings = decode_with_lexicon(candidates, board, racks=[[]], lexicon=None)
    assert readings[0].labels[(7, 7)] != "Q"


def test_empty_cell_candidates_returns_a_single_empty_reading():
    readings = decode_with_lexicon([], BoardState(), racks=[[]])
    assert len(readings) == 1
    assert readings[0].labels == {}
    assert readings[0].all_words_valid is True
