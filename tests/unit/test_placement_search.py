from autoscorer.gamelogic.board import BoardState, Tile
from autoscorer.gamelogic.movedetect.placement_search import (
    CandidatePlacement,
    enumerate_candidate_placements,
)


def _cell_sets(candidates):
    return {tuple(sorted(c.cells)) for c in candidates}


def test_empty_confirmed_cells_yields_no_candidates():
    assert enumerate_candidate_placements(frozenset(), BoardState()) == []


def test_a_simple_collinear_row_is_one_candidate():
    board = BoardState()
    confirmed = frozenset({(7, 6), (7, 7), (7, 8)})
    candidates = enumerate_candidate_placements(confirmed, board)
    assert ((7, 6), (7, 7), (7, 8)) in _cell_sets(candidates)


def test_non_collinear_blob_splits_into_legal_collinear_candidates():
    """The exact real blob from the log that jammed the pipeline for 163
    consecutive observations: (3,2) is an existing tile the play bridges
    through. No candidate here may span multiple rows AND columns."""
    board = BoardState()
    board.set((3, 2), Tile(letter="X", is_blank=False))
    confirmed = frozenset({(1, 1), (2, 1), (3, 0), (3, 1), (3, 3)})
    candidates = enumerate_candidate_placements(confirmed, board)

    for c in candidates:
        rows = {r for r, _ in c.cells}
        cols = {col for _, col in c.cells}
        assert len(rows) == 1 or len(cols) == 1, f"non-collinear candidate: {c.cells}"

    cell_sets = _cell_sets(candidates)
    assert ((1, 1), (2, 1), (3, 1)) in cell_sets  # the column-1 run
    assert ((3, 0), (3, 1), (3, 3)) in cell_sets  # the row-3 run, bridged through (3,2)


def test_cluster_max_size_reflects_the_largest_candidate_from_that_cluster():
    board = BoardState()
    board.set((3, 2), Tile(letter="X", is_blank=False))
    confirmed = frozenset({(1, 1), (2, 1), (3, 0), (3, 1), (3, 3)})
    candidates = enumerate_candidate_placements(confirmed, board)
    assert all(c.cluster_max_size == 3 for c in candidates)


def test_two_disconnected_single_cell_clusters_each_become_their_own_candidate():
    board = BoardState()
    confirmed = frozenset({(0, 0), (10, 10)})
    candidates = enumerate_candidate_placements(confirmed, board)
    cell_sets = _cell_sets(candidates)
    assert ((0, 0),) in cell_sets
    assert ((10, 10),) in cell_sets


def test_a_gap_not_bridged_by_an_occupied_cell_splits_into_two_runs():
    board = BoardState()  # (0,1) stays genuinely empty -- no bridge
    confirmed = frozenset({(0, 0), (0, 2)})
    candidates = enumerate_candidate_placements(confirmed, board)
    cell_sets = _cell_sets(candidates)
    assert ((0, 0),) in cell_sets
    assert ((0, 2),) in cell_sets
    assert ((0, 0), (0, 2)) not in cell_sets


def test_hook_through_an_existing_tile_still_clusters_and_yields_one_full_run():
    # the real "ARB.RIZE" case: new cells on both sides of an existing tile.
    board = BoardState()
    board.set((7, 7), Tile(letter="B", is_blank=False))
    confirmed = frozenset({(7, 5), (7, 6), (7, 8), (7, 9)})
    candidates = enumerate_candidate_placements(confirmed, board)
    cell_sets = _cell_sets(candidates)
    assert ((7, 5), (7, 6), (7, 8), (7, 9)) in cell_sets


def test_soft_cells_extend_a_hard_run_at_either_end_and_are_flagged():
    board = BoardState()
    confirmed = frozenset({(7, 6), (7, 7)})
    soft = frozenset({(7, 8)})
    candidates = enumerate_candidate_placements(confirmed, board, soft_cells=soft)
    extended = [c for c in candidates if c.used_soft_cells]
    assert len(extended) == 1
    assert extended[0].cells == frozenset({(7, 6), (7, 7), (7, 8)})
    hard = [c for c in candidates if not c.used_soft_cells]
    assert any(c.cells == frozenset({(7, 6), (7, 7)}) for c in hard)


def test_soft_cells_not_adjacent_to_any_run_are_ignored():
    board = BoardState()
    confirmed = frozenset({(7, 6), (7, 7)})
    soft = frozenset({(10, 10)})
    candidates = enumerate_candidate_placements(confirmed, board, soft_cells=soft)
    assert not any(c.used_soft_cells for c in candidates)


def test_soft_cell_extension_never_replaces_hard_candidates():
    board = BoardState()
    confirmed = frozenset({(7, 6), (7, 7)})
    soft = frozenset({(7, 8)})
    candidates = enumerate_candidate_placements(confirmed, board, soft_cells=soft)
    assert any(not c.used_soft_cells and c.cells == frozenset({(7, 6), (7, 7)}) for c in candidates)


def test_candidate_placement_is_frozen_and_hashable():
    c = CandidatePlacement(cells=frozenset({(0, 0)}), cluster_max_size=1)
    assert {c} == {c}
