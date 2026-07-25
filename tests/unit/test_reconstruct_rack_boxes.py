import numpy as np
import pytest

from training.detect.reconstruct_rack_boxes import (
    DEFAULT_ABOVE_LEDGE,
    DEFAULT_BELOW_LEDGE,
    detect_ledge_top,
    reconstruct_boxes_from_groups,
)


def _image_with_wood_ledge(ledge_top: int, height: int = 150, width: int = 900) -> np.ndarray:
    """BGR image: dark background above `ledge_top`, warm wood color at
    and below it -- OpenCV convention (B, G, R channel order)."""
    img = np.full((height, width, 3), (40, 40, 40), dtype=np.uint8)  # dark, neutral background
    img[ledge_top:, :] = (60, 140, 190)  # BGR: strong red, weak blue -- "wood"
    return img


def test_detect_ledge_top_finds_the_wood_boundary():
    img = _image_with_wood_ledge(ledge_top=70)
    assert detect_ledge_top(img) == 70


def test_detect_ledge_top_returns_none_with_no_wood_present():
    img = np.full((150, 900, 3), (40, 40, 40), dtype=np.uint8)
    assert detect_ledge_top(img) is None


def test_reconstruct_boxes_divides_each_group_evenly():
    img = _image_with_wood_ledge(ledge_top=70)
    groups = [(100, 400, ["A", "B", "C"])]
    boxes = reconstruct_boxes_from_groups(img, groups)

    assert len(boxes) == 3
    assert [b.label for b in boxes] == ["A", "B", "C"]
    step = (400 - 100) / 3
    for i, box in enumerate(boxes):
        assert box.x1 == 100 + int(i * step)
        assert box.x2 == 100 + int((i + 1) * step)


def test_reconstruct_boxes_uses_ledge_relative_y_band_by_default():
    ledge = 70
    img = _image_with_wood_ledge(ledge_top=ledge)
    boxes = reconstruct_boxes_from_groups(img, [(0, 90, ["Z"])])
    assert boxes[0].y1 == ledge - DEFAULT_ABOVE_LEDGE
    assert boxes[0].y2 == ledge + DEFAULT_BELOW_LEDGE


def test_reconstruct_boxes_respects_explicit_y_band_override():
    img = _image_with_wood_ledge(ledge_top=70)
    boxes = reconstruct_boxes_from_groups(img, [(0, 90, ["Z"])], y_band=(10, 60))
    assert boxes[0].y1 == 10
    assert boxes[0].y2 == 60


def test_reconstruct_boxes_handles_multiple_groups():
    img = _image_with_wood_ledge(ledge_top=70)
    groups = [(0, 90, ["A", "B"]), (200, 350, ["C", "D", "E"])]
    boxes = reconstruct_boxes_from_groups(img, groups)
    assert [b.label for b in boxes] == ["A", "B", "C", "D", "E"]


def test_reconstruct_boxes_raises_without_ledge_or_override():
    img = np.full((150, 900, 3), (40, 40, 40), dtype=np.uint8)  # no wood anywhere
    with pytest.raises(ValueError):
        reconstruct_boxes_from_groups(img, [(0, 90, ["A"])])
