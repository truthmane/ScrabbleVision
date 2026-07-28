import numpy as np

from autoscorer.perception.classify.blank_heuristic import (
    DEFAULT_CV_THRESHOLD,
    looks_smooth_like_a_blank,
    patch_coefficient_of_variation,
)


def _uniform_crop(value: int = 180, size: int = 60) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _textured_crop(size: int = 60) -> np.ndarray:
    crop = np.full((size, size, 3), 180, dtype=np.uint8)
    # A strong-contrast "glyph" stroke through the center, well within the
    # patch the heuristic actually samples.
    crop[size // 3: 2 * size // 3, size // 4: 3 * size // 4] = 10
    return crop


def test_uniform_patch_has_low_coefficient_of_variation():
    cv_val = patch_coefficient_of_variation(_uniform_crop())
    assert cv_val < 1.0


def test_textured_patch_has_high_coefficient_of_variation():
    cv_val = patch_coefficient_of_variation(_textured_crop())
    assert cv_val > DEFAULT_CV_THRESHOLD


def test_looks_smooth_like_a_blank_matches_the_threshold():
    assert looks_smooth_like_a_blank(_uniform_crop())
    assert not looks_smooth_like_a_blank(_textured_crop())


def test_a_slightly_noisy_but_still_smooth_patch_stays_below_threshold():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 3, (60, 60, 3))
    crop = np.clip(180 + noise, 0, 255).astype(np.uint8)
    assert looks_smooth_like_a_blank(crop)
