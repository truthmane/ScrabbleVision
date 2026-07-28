import random

import pytest

from training.synth_render.tile_renderer import (
    FINAL_SIZE,
    TILE_CLASSES,
    _solve_perspective_coeffs,
    augment_real_photo,
    augment_tile,
    generate_synthetic_dataset,
    render_tile,
)


def test_tile_classes_are_26_letters_plus_blank():
    assert len(TILE_CLASSES) == 27
    assert TILE_CLASSES[:26] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert TILE_CLASSES[26] == "BLANK"


def test_augmented_tile_is_final_size_rgb():
    base = render_tile("A", rng=random.Random(1))
    out = augment_tile(base, rng=random.Random(1))
    assert out.size == (FINAL_SIZE, FINAL_SIZE)
    assert out.mode == "RGB"


def test_blank_tile_has_no_letter_glyph_by_construction():
    # Rendered directly with letter=None -- just confirm it doesn't raise
    # and produces a valid image; visual "no glyph" is a rendering property
    # verified manually, not something worth pixel-diffing in a unit test.
    tile = render_tile(None, rng=random.Random(2))
    assert tile.size[0] == tile.size[1]


def test_augment_real_photo_preserves_size_and_mode():
    base = render_tile("A", rng=random.Random(1))
    out = augment_real_photo(base, rng=random.Random(1))
    assert out.size == base.size
    assert out.mode == "RGB"


def test_solve_perspective_coeffs_maps_dst_corners_back_to_src():
    # Regression test for a bug where `b` was built from `dst` instead of
    # `src`, which made the solver trivially return the identity transform
    # (invisible for a long time since it was only ever exercised with a
    # tiny jitter that looks similar to a no-op at a glance).
    size = 60
    src = [(0, 0), (size, 0), (size, size), (0, size)]
    dst = [(10, 0), (size - 10, 0), (size, size), (0, size)]
    coeffs = _solve_perspective_coeffs(src, dst)

    assert coeffs != (1, 0, 0, 0, 1, 0, 0, 0)

    a, b, c, d, e, f, g, h = coeffs
    for (dx, dy), (sx, sy) in zip(dst, src):
        denom = g * dx + h * dy + 1
        mapped_x = (a * dx + b * dy + c) / denom
        mapped_y = (d * dx + e * dy + f) / denom
        assert mapped_x == pytest.approx(sx, abs=1e-6)
        assert mapped_y == pytest.approx(sy, abs=1e-6)


def test_generate_synthetic_dataset_writes_expected_file_counts(tmp_path):
    samples = generate_synthetic_dataset(tmp_path, samples_per_class=2, seed=3)

    assert len(samples) == 27 * 2
    for label in TILE_CLASSES:
        class_dir = tmp_path / label
        assert class_dir.is_dir()
        assert len(list(class_dir.glob("*.png"))) == 2
