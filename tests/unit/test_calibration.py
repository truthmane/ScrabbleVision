"""Tests the calibration math against synthetically generated images -- no
camera or real photos needed, since this is pure geometry/CV code that can
be exercised with numpy-generated pixel data.
"""
import cv2
import numpy as np

from autoscorer.perception.calibration.homography import (
    CANONICAL_CELL_PX,
    CANONICAL_SIZE,
    CORNER_MARKER_IDS,
    DEFAULT_OCCUPANCY_INSET_FRAC,
    ARUCO_DICTIONARY,
    calibrate_from_aruco,
    calibrate_from_corners,
    cell_bounds,
    crop_cell,
    crop_cell_inset,
)


def _canonical_test_image() -> np.ndarray:
    """A distinctive canonical image: white background, a solid red square
    painted over what should be board cell (3, 4), for verifying that
    rectification recovers the correct cell content."""
    img = np.full((CANONICAL_SIZE, CANONICAL_SIZE, 3), 255, dtype=np.uint8)
    x1, y1, x2, y2 = cell_bounds(3, 4)
    img[y1:y2, x1:x2] = (0, 0, 255)  # BGR red
    return img


def _warp(image: np.ndarray, dst_corners: np.ndarray) -> np.ndarray:
    src_corners = np.array(
        [[0, 0], [CANONICAL_SIZE, 0], [CANONICAL_SIZE, CANONICAL_SIZE], [0, CANONICAL_SIZE]],
        dtype=np.float32,
    )
    h, _ = cv2.findHomography(src_corners, dst_corners)
    return cv2.warpPerspective(image, h, (CANONICAL_SIZE, CANONICAL_SIZE))


def test_calibrate_from_corners_recovers_cell_content_through_a_perspective_warp():
    canonical = _canonical_test_image()

    # Simulate a camera viewing the board at a mild skewed angle: the
    # "photographed" corners are not a perfect axis-aligned square.
    photographed_corners = np.array(
        [[50, 30], [860, 10], [890, 880], [20, 900]], dtype=np.float32,
    )
    photo = _warp(canonical, photographed_corners)

    calibration = calibrate_from_corners(
        [tuple(p) for p in photographed_corners]
    )
    rectified = calibration.rectify(photo)

    red_crop = crop_cell(rectified, 3, 4)
    # Allow some tolerance at the crop edges for interpolation artifacts.
    center = red_crop[10:-10, 10:-10]
    assert np.mean(center[:, :, 2]) > 200  # strongly red channel
    assert np.mean(center[:, :, 1]) < 60   # weak green/blue -> not white


def test_crop_cell_inset_is_smaller_than_crop_cell_and_centered_within_it():
    # WS3: occupancy detection uses crop_cell_inset instead of crop_cell to
    # avoid the grid-line/neighbor-bleed border that sits at the edge of
    # every full-cell crop -- pin the geometry so a future change can't
    # silently widen the inset back out to zero or drift off-center.
    image = np.arange(CANONICAL_SIZE * CANONICAL_SIZE * 3, dtype=np.uint8).reshape(
        CANONICAL_SIZE, CANONICAL_SIZE, 3
    )
    full = crop_cell(image, 3, 4)
    inset = crop_cell_inset(image, 3, 4)

    trim = round(CANONICAL_CELL_PX * DEFAULT_OCCUPANCY_INSET_FRAC)
    assert full.shape == (CANONICAL_CELL_PX, CANONICAL_CELL_PX, 3)
    assert inset.shape == (CANONICAL_CELL_PX - 2 * trim, CANONICAL_CELL_PX - 2 * trim, 3)
    np.testing.assert_array_equal(inset, full[trim:-trim, trim:-trim])


def test_calibrate_from_corners_rejects_wrong_number_of_points():
    import pytest
    with pytest.raises(ValueError):
        calibrate_from_corners([(0, 0), (1, 1)])


def _marker_image(marker_id: int, size: int = 80) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, size)


def _compose_markers_on_canvas(canvas_size: int, marker_positions: dict) -> np.ndarray:
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    for marker_id, (x, y) in marker_positions.items():
        marker = _marker_image(marker_id)
        m = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        h, w = m.shape[:2]
        canvas[y:y + h, x:x + w] = m
    return canvas


def test_calibrate_from_aruco_detects_all_four_corner_markers():
    canvas_size = 1000
    margin = 40
    positions = {
        CORNER_MARKER_IDS["top_left"]: (margin, margin),
        CORNER_MARKER_IDS["top_right"]: (canvas_size - margin - 80, margin),
        CORNER_MARKER_IDS["bottom_right"]: (canvas_size - margin - 80, canvas_size - margin - 80),
        CORNER_MARKER_IDS["bottom_left"]: (margin, canvas_size - margin - 80),
    }
    image = _compose_markers_on_canvas(canvas_size, positions)

    calibration = calibrate_from_aruco(image)

    assert calibration is not None
    assert calibration.homography.shape == (3, 3)


def test_calibrate_from_aruco_returns_none_when_a_marker_is_missing():
    canvas_size = 1000
    margin = 40
    # Only three of the four corner markers present.
    positions = {
        CORNER_MARKER_IDS["top_left"]: (margin, margin),
        CORNER_MARKER_IDS["top_right"]: (canvas_size - margin - 80, margin),
        CORNER_MARKER_IDS["bottom_right"]: (canvas_size - margin - 80, canvas_size - margin - 80),
    }
    image = _compose_markers_on_canvas(canvas_size, positions)

    assert calibrate_from_aruco(image) is None
