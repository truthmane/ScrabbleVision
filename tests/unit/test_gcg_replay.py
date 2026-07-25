from pathlib import Path

import pytest

from training.collect.replay_game import read_gcg_moves, replay_gcg_game

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
FIXTURE = FIXTURES_DIR / "spc2026_final_game1.gcg"

# Five real games from the 2026 NASPA Scrabble Players Championship
# (event.scrabbleplayers.org), spanning different player pairs and rounds
# -- including the eventual champion's own finals games, exchanges, and
# multiple bingos -- so this isn't just one lucky game replaying cleanly.
ALL_FIXTURES = [
    FIXTURES_DIR / "spc2026_final_game1.gcg",
    FIXTURES_DIR / "spc2026_game8.gcg",
    FIXTURES_DIR / "spc2026_game9.gcg",
    FIXTURES_DIR / "spc2026_final_game3.gcg",
    FIXTURES_DIR / "spc2026_game22.gcg",
]


def test_read_gcg_moves_skips_header_and_endgame_bonus_line():
    moves = read_gcg_moves(FIXTURE)
    # 24 real plays; the 4 '#' header lines and the trailing "(EFT) +12 383"
    # endgame bonus line (no position/word -- our rules engine doesn't
    # model that) are all correctly skipped, not misparsed as moves.
    assert len(moves) == 24
    assert moves[0].player == "Orry"
    assert moves[0].word == "LICK"
    assert moves[-1].player == "Mack"
    assert moves[-1].word == ".E"


def test_replay_reproduces_every_real_official_score():
    """The core validation this fixture exists for: our from-scratch
    rules engine, fed nothing but position+word+rack from an official
    NASPA championship final, reproduces every single turn score and
    running cumulative total the tournament itself recorded -- across
    24 real turns, multiple bingos, several blanks, and hooks through
    existing words in both directions.
    """
    moves = read_gcg_moves(FIXTURE)
    results = replay_gcg_game(moves)

    assert len(results) == 24
    for i, turn in enumerate(results, start=1):
        assert turn.error is None, f"turn {i} ({turn.move.player} {turn.move.word}): {turn.error}"
        assert turn.score_matches, (
            f"turn {i} ({turn.move.player} {turn.move.word}): "
            f"computed {turn.computed_score} != official {turn.move.turn_score}"
        )
        assert turn.cumulative_matches, (
            f"turn {i} ({turn.move.player} {turn.move.word}): "
            f"cumulative computed vs official mismatch"
        )

    # Final recorded cumulative scores before the endgame bonus line:
    # Orry 326, Mack 371.
    assert results[-2].move.player == "Orry"
    assert results[-2].computed_score == 22
    assert results[-1].move.player == "Mack"
    assert results[-1].computed_score == 10


@pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda p: p.stem)
def test_replay_matches_official_scores_across_multiple_real_games(fixture: Path):
    """Broader than the single-game test above: every parseable play in
    each of four real, independently-played championship games (different
    player pairs, different rounds, including exchanges the parser
    correctly skips) reproduces the tournament's own recorded score."""
    moves = read_gcg_moves(fixture)
    assert len(moves) > 15, f"{fixture.name}: suspiciously few plays parsed ({len(moves)})"

    results = replay_gcg_game(moves)
    for i, turn in enumerate(results, start=1):
        assert turn.error is None, f"{fixture.name} turn {i}: {turn.error}"
        assert turn.score_matches, f"{fixture.name} turn {i}: computed {turn.computed_score} != official {turn.move.turn_score}"
        assert turn.cumulative_matches, f"{fixture.name} turn {i}: cumulative mismatch"
