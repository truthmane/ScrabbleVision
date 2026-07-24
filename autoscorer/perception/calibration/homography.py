"""Maps a camera image of the physical board to a canonical, square-on,
per-cell-addressable image via a homography. Two ways to obtain the four
corner points: ArUco markers (primary -- self-recalibrating every frame if
markers stay in view) or a one-time manual corner click (fallback for a
venue without markers affixed).

The canonical output is always a `CANONICAL_SIZE`-pixel square covering
exactly the 15x15 playing grid, so downstream code can address any cell by
simple pixel arithmetic -- no further geometry needed once rectified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from autoscorer.gamelogic.board import BOARD_SIZE

CANONICAL_CELL_PX = 60
CANONICAL_SIZE = CANONICAL_CELL_PX * BOARD_SIZE  # 900px square

# Convention: four ArUco markers (from the standard 4x4_50 dictionary) taped
# just outside the four corners of the playing grid, one per corner. This is
# a placeholder convention -- adjust IDs/placement once real hardware is
# chosen, but the calibration math is independent of that choice.
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
CORNER_MARKER_IDS = {
    "top_left": 0,
    "top_right": 1,
    "bottom_right": 2,
    "bottom_left": 3,
}

_CANONICAL_CORNERS = np.array(
    [[0, 0], [CANONICAL_SIZE, 0], [CANONICAL_SIZE, CANONICAL_SIZE], [0, CANONICAL_SIZE]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class BoardCalibration:
    homography: np.ndarray  # 3x3, maps source image points -> canonical points

    def rectify(self, image: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(image, self.homography, (CANONICAL_SIZE, CANONICAL_SIZE))


def _homography_from_corners(
    top_left: Tuple[float, float],
    top_right: Tuple[float, float],
    bottom_right: Tuple[float, float],
    bottom_left: Tuple[float, float],
) -> BoardCalibration:
    source = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    homography, _ = cv2.findHomography(source, _CANONICAL_CORNERS)
    if homography is None:
        raise ValueError("could not compute a homography from the given corners")
    return BoardCalibration(homography=homography)


def calibrate_from_corners(corners: Sequence[Tuple[float, float]]) -> BoardCalibration:
    """Manual fallback: `corners` must be (top_left, top_right, bottom_right,
    bottom_left) pixel coordinates in the source image, as clicked by an
    operator during session setup.
    """
    if len(corners) != 4:
        raise ValueError(f"expected exactly 4 corner points, got {len(corners)}")
    return _homography_from_corners(*corners)


def calibrate_from_aruco(image: np.ndarray) -> Optional[BoardCalibration]:
    """Detect the four corner ArUco markers and compute a homography from
    their centers. Returns None if not all four corner markers are visible
    in this frame -- callers should keep the previous calibration in that
    case rather than treating it as fatal (a marker can be briefly occluded
    by a hand without the camera having actually moved).
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None:
        return None

    centers_by_id = {}
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        centers_by_id[int(marker_id)] = marker_corners.reshape(4, 2).mean(axis=0)

    required = CORNER_MARKER_IDS.values()
    if not all(marker_id in centers_by_id for marker_id in required):
        return None

    return _homography_from_corners(
        top_left=tuple(centers_by_id[CORNER_MARKER_IDS["top_left"]]),
        top_right=tuple(centers_by_id[CORNER_MARKER_IDS["top_right"]]),
        bottom_right=tuple(centers_by_id[CORNER_MARKER_IDS["bottom_right"]]),
        bottom_left=tuple(centers_by_id[CORNER_MARKER_IDS["bottom_left"]]),
    )


def cell_bounds(row: int, col: int) -> Tuple[int, int, int, int]:
    """Pixel bounds (x1, y1, x2, y2) of board cell (row, col) in a
    canonical-rectified image."""
    x1 = col * CANONICAL_CELL_PX
    y1 = row * CANONICAL_CELL_PX
    return x1, y1, x1 + CANONICAL_CELL_PX, y1 + CANONICAL_CELL_PX


def crop_cell(rectified_image: np.ndarray, row: int, col: int) -> np.ndarray:
    x1, y1, x2, y2 = cell_bounds(row, col)
    return rectified_image[y1:y2, x1:x2]
