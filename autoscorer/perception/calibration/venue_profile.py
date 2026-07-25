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
from typing import Tuple

from autoscorer.perception.calibration.homography import BoardCalibration, calibrate_from_corners
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
    """
    name: str
    corners: Tuple[Corner, Corner, Corner, Corner]
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    still_frame_count: int = DEFAULT_STILL_FRAME_COUNT
    notes: str = field(default="")

    def calibration(self) -> BoardCalibration:
        return calibrate_from_corners(self.corners)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "corners": [list(c) for c in self.corners],
            "motion_threshold": self.motion_threshold,
            "still_frame_count": self.still_frame_count,
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
