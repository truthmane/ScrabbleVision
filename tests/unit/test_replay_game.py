from pathlib import Path
from typing import Dict

from autoscorer.gamelogic.board import BoardState, Coord, Tile
from autoscorer.gamelogic.movedetect.word_resolver import words_formed
from autoscorer.gamelogic.scoring.rules_engine import score_move
from training.collect.replay_game import read_moves, replay_game


def _expected_score(board_before: BoardState, placements: Dict[Coord, Tile]) -> int:
    """Computes a move's score the same way `process_turn` does internally,
    independent of the ticker-notation parser -- used to build a synthetic
    ticker fixture whose reported scores are known to be correct, so the
    test actually exercises the notation-parsing + replay bridge rather
    than just checking replay_game agrees with itself."""
    board_after = board_before.with_placements(placements)
    words = words_formed(board_after, list(placements.keys()))
    return score_move(board_after, words, list(placements.keys())).total


def test_replay_reconstructs_scores_and_board_across_a_small_game(tmp_path: Path):
    board = BoardState()

    # Move 1: CAT down through the center (7,7), first move.
    move1_placements = {(7, 7): Tile("C"), (8, 7): Tile("A"), (9, 7): Tile("T")}
    score1 = _expected_score(board, move1_placements)
    board = board.with_placements(move1_placements)

    # Move 2: hooks the existing A at (8,7), plays "AND" across (A pre-existing).
    move2_placements = {(8, 8): Tile("N"), (8, 9): Tile("D")}
    score2 = _expected_score(board, move2_placements)
    board = board.with_placements(move2_placements)

    # Move 3: hooks the existing D at (8,9), plays "DOG" down -- O played by a blank.
    move3_placements = {(9, 9): Tile("O", is_blank=True), (10, 9): Tile("G")}
    score3 = _expected_score(board, move3_placements)
    board = board.with_placements(move3_placements)

    lines = [
        f"Player One, P1 H8 CAT {score1} {score1}",
        f"Player Two, P2 9H (A)ND {score2} {score2}",
        f"Player One, P1 J9 (D)oG {score3} {score1 + score3}",
    ]
    moves_file = tmp_path / "game.txt"
    moves_file.write_text("\n".join(lines) + "\n")

    moves = read_moves(moves_file)
    results = replay_game(moves)

    assert len(results) == 3
    for turn in results:
        assert turn.error is None
        assert turn.score_matches, f"{turn.move}: computed {turn.computed_score} != ticker {turn.move.turn_score}"
        assert turn.cumulative_matches

    final_board = results[-1].board_after
    assert final_board.get((7, 7)) == Tile("C")
    assert final_board.get((8, 7)) == Tile("A")
    assert final_board.get((9, 7)) == Tile("T")
    assert final_board.get((8, 8)) == Tile("N")
    assert final_board.get((8, 9)) == Tile("D")
    assert final_board.get((9, 9)) == Tile("O", is_blank=True)
    assert final_board.get((10, 9)) == Tile("G")


def test_read_moves_skips_blank_lines_and_comments(tmp_path: Path):
    moves_file = tmp_path / "game.txt"
    moves_file.write_text(
        "# a full game transcript\n"
        "\n"
        "Player One, P1 H8 CAT 10 10\n"
        "\n"
        "# second move\n"
        "Player Two, P2 9H (A)ND 5 5\n"
    )
    moves = read_moves(moves_file)
    assert len(moves) == 2
    assert moves[0].word == "CAT"
    assert moves[1].word == "(A)ND"


def test_replay_reports_error_for_illegal_placement(tmp_path: Path):
    # Second move doesn't connect to anything and doesn't cover the center.
    moves_file = tmp_path / "game.txt"
    moves_file.write_text(
        "Player One, P1 H8 CAT 10 10\n"
        "Player Two, P2 A1 ZZZ 30 30\n"
    )
    moves = read_moves(moves_file)
    results = replay_game(moves)
    assert results[0].error is None
    assert results[1].error is not None
