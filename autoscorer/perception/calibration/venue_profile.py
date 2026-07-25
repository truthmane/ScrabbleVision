"""Persists per-venue calibration so it survives between sessions --
the master plan calls for `configs/venues/<name>.json` explicitly
("returning venues skip full recalibration"), but until now every corner
set this project ever calibrated (NASPA broadcast, WESPA Word Wars) lived
only in a chat transcript or scratchpad variable, gone the moment the
session ended. A venue with markers taped down would use
`calibrate_from_aruco` fresh every session instead; this file is for the
manual-corner-click fallback, which is what every real venue this project
has actually touched has needed so far.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from autoscorer.perception.calibration.homography import BoardCalibration, calibrate_from_corners
from autoscorer.perception.occupancy.detector import DEFAULT_DIFF_THRESHOLD, DEFAULT_GRADIENT_THRESHOLD
from autoscorer.perception.stillness.detector import DEFAULT_MOTION_THRESHOLD, DEFAULT_STILL_FRAME_COUNT

Corner = Tuple[float, float]

DEFAULT_VENUES_DIR = Path(__file__).resolve().parents[3] / "configs" / "venues"


@dataclass(frozen=True)
class VenueProfile:
    """One venue's saved calibration. `corners` are (top_left, top_right,
    bottom_right, bottom_left) pixel coordinates in that venue's camera
    frame, exactly the order `calibrate_from_corners` expects.
    `motion_threshold` is included because the stillness gate's noise
    floor is venue-specific too (lighting flicker, camera sensor noise) --
    same reasoning as the corners themselves.

    `occupancy_diff_threshold`/`occupancy_gradient_threshold` matter more
    than they look: a venue whose premium squares carry busy printed
    graphics (e.g. WESPA's "DOUBLE LETTER SCORE" text, versus a plainer
    board) can score genuinely empty cells well above the NASPA-tuned
    defaults on both signals -- getting this wrong doesn't just hurt
    accuracy, it multiplies classifier calls per settled frame (see
    `board_reader.read_new_cells_voted`'s docstring), the difference
    between a fast per-frame check and one that silently takes tens of
    seconds. Always re-tune against that venue's own empty-board photo
    (`reference_board_path`) rather than trusting either venue's defaults.

    `reference_board_path`, if set, points at a real empty-board reference
    photo (already rectified) **stored locally, never committed to the
    repo** -- same copyright policy as every other real broadcast photo
    this project has touched. `load_reference_board()` reads it back.
    """
    name: str
    corners: Tuple[Corner, Corner, Corner, Corner]
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    still_frame_count: int = DEFAULT_STILL_FRAME_COUNT
    occupancy_diff_threshold: float = DEFAULT_DIFF_THRESHOLD
    occupancy_gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD
    reference_board_path: Optional[str] = None
    notes: str = field(default="")

    def calibration(self) -> BoardCalibration:
        return calibrate_from_corners(self.corners)

    def load_reference_board(self) -> np.ndarray:
        """Reads the real, rectified empty-board reference photo this
        profile points at. Raises if `reference_board_path` isn't set or
        the file isn't present locally (it's never committed -- see the
        class docstring)."""
        import cv2

        if self.reference_board_path is None:
            raise ValueError(f"venue {self.name!r} has no reference_board_path set")
        path = Path(self.reference_board_path)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"reference board photo not found at {path}")
        return image

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "corners": [list(c) for c in self.corners],
            "motion_threshold": self.motion_threshold,
            "still_frame_count": self.still_frame_count,
            "occupancy_diff_threshold": self.occupancy_diff_threshold,
            "occupancy_gradient_threshold": self.occupancy_gradient_threshold,
            "reference_board_path": self.reference_board_path,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict) -> "VenueProfile":
        corners = tuple(tuple(c) for c in data["corners"])
        if len(corners) != 4:
            raise ValueError(f"expected exactly 4 corners, got {len(corners)}")
        return VenueProfile(
            name=data["name"],
            corners=corners,
            motion_threshold=data.get("motion_threshold", DEFAULT_MOTION_THRESHOLD),
            still_frame_count=data.get("still_frame_count", DEFAULT_STILL_FRAME_COUNT),
            occupancy_diff_threshold=data.get("occupancy_diff_threshold", DEFAULT_DIFF_THRESHOLD),
            occupancy_gradient_threshold=data.get("occupancy_gradient_threshold", DEFAULT_GRADIENT_THRESHOLD),
            reference_board_path=data.get("reference_board_path"),
            notes=data.get("notes", ""),
        )


def save_venue_profile(profile: VenueProfile, directory: Path = DEFAULT_VENUES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{profile.name}.json"
    path.write_text(json.dumps(profile.to_dict(), indent=2) + "\n")
    return path


def load_venue_profile(name: str, directory: Path = DEFAULT_VENUES_DIR) -> VenueProfile:
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no saved venue profile at {path}")
    return VenueProfile.from_dict(json.loads(path.read_text()))


def list_venue_profiles(directory: Path = DEFAULT_VENUES_DIR) -> list:
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))
