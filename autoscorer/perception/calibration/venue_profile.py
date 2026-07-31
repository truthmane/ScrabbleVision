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
from typing import List, Optional, Tuple, Union

import numpy as np

from autoscorer.perception.calibration.homography import BoardCalibration, calibrate_from_corners
from autoscorer.perception.clock.clock_reader import ClockRegions
from autoscorer.perception.occupancy.detector import DEFAULT_DIFF_THRESHOLD, DEFAULT_GRADIENT_THRESHOLD
from autoscorer.perception.stillness.detector import DEFAULT_MOTION_THRESHOLD, DEFAULT_STILL_FRAME_COUNT

Corner = Tuple[float, float]
ClockBox = Tuple[int, int, int, int]

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
    Store it under `~/.autoscorer/venue_references/`, not a session-scoped
    scratchpad dir or anywhere inside a git worktree -- both get wiped
    between sessions (a worktree can be pruned/recreated same as a
    scratchpad can be cleaned), silently breaking `load_reference_board()`
    for every venue profile that pointed into one. This actually happened
    to the WESPA profile's reference photo once; see the "moved there
    2026-07-31" note in `configs/venues/wespa_word_wars.json`.

    `additional_reference_board_paths` covers a real wrinkle a single
    reference photo can't: some venues have more than one genuinely valid
    *empty* appearance -- WESPA's center square sometimes shows a sponsor
    logo overlay and sometimes the plain premium-square color underneath,
    neither of which means a tile is there, but a single fixed reference
    only ever captures one state and reads the other as spuriously
    occupied (see `occupancy/detector.py`'s module docstring for exactly
    how this was found -- 61 false detections on that one cell across one
    real clip). Add every other genuinely-empty state you've observed
    here; `load_reference_board()` returns all of them together and
    occupancy detection only counts a cell as occupied if it differs from
    *every* one, not just the first.
    """
    name: str
    corners: Tuple[Corner, Corner, Corner, Corner]
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    still_frame_count: int = DEFAULT_STILL_FRAME_COUNT
    still_seconds: Optional[float] = None
    """The REAL calibrated meaning of the stillness gate, in wall-clock
    seconds -- e.g. 25.0 for WESPA (5 frames at the 0.2 fps every
    calibration run for that venue actually used). `still_frame_count`
    alone is a frame COUNT with no fixed real-world meaning: the same 5
    frames means 25s of stillness at 0.2 fps but only 2.5s at 2.0 fps,
    and at 2.5s a player's ordinary pause between placing individual
    tiles reads as "the turn is over," fragmenting real multi-tile words
    -- a real, measured bug (a CLI defaulting to the wrong fps produced
    hours of what looked exactly like a code regression before this was
    found). When set, this is authoritative: `effective_still_frame_count`
    derives the actual frame count from whatever sample_fps a run is
    using, so the gate means the same thing regardless of rate.  `None`
    (the default, for a profile that hasn't recorded this yet) falls back
    to the raw `still_frame_count` field, unconverted -- re-derive and set
    this for any profile whose `still_frame_count` was ever tuned against
    a specific, known sample_fps."""
    occupancy_diff_threshold: float = DEFAULT_DIFF_THRESHOLD
    occupancy_gradient_threshold: float = DEFAULT_GRADIENT_THRESHOLD
    reference_board_path: Optional[str] = None
    additional_reference_board_paths: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = field(default="")
    lexicon: Optional[str] = None
    """Name or path resolved via `dictionary.lexicon.load_lexicon` --
    e.g. a licensed CSW/NWL drop-in under `configs/lexicons/` for a real
    tournament's actual dictionary. `None` (the default) means "use
    whatever `load_lexicon` falls back to" (the vendored generic list),
    not "no lexicon" -- the decoder always has one to rank against."""
    clock_regions: Optional[Tuple[ClockBox, ClockBox]] = None
    """(left_box, right_box) pixel bounds of this venue's two on-screen
    chess-clock displays in the RAW (unrectified) camera frame -- see
    `perception.clock.clock_reader`'s module docstring for why this
    exists: board-pixel stillness alone cannot distinguish "player
    finished their move" from "player is mid-word and just paused,"
    which real broadcast footage showed fragmenting real multi-tile
    words into several wrong partial-word candidates an operator had to
    manually reject. `None` (the default) means this venue has no clock
    configured yet -- `GameWatcher` falls back to stillness-only timing
    exactly as before this existed, so this is a strictly opt-in,
    backward-compatible addition. Screen-position labels only ("left"/
    "right"), never tied to a specific named player -- this venue always
    renders one player on each side, and `GameWatcher` only needs to know
    whether the active side changed, not which named player is which."""

    def calibration(self) -> BoardCalibration:
        return calibrate_from_corners(self.corners)

    def clock_regions_obj(self) -> Optional[ClockRegions]:
        if self.clock_regions is None:
            return None
        left, right = self.clock_regions
        return ClockRegions(left=tuple(left), right=tuple(right))

    def effective_still_frame_count(self, sample_fps: float) -> int:
        """The frame count `GameWatcher`'s stillness gate should actually
        use, given the real sampling rate a run is using -- see
        `still_seconds`'s docstring for why this can't just be the raw
        `still_frame_count` field. Rounds to the nearest frame, floored
        at 1 (a gate requiring zero frames is meaningless)."""
        if self.still_seconds is not None:
            return max(1, round(self.still_seconds * sample_fps))
        return self.still_frame_count

    def load_reference_board(self) -> Union[np.ndarray, List[np.ndarray]]:
        """Reads the real, rectified empty-board reference photo(s) this
        profile points at -- a single image if `additional_reference_board_paths`
        is empty (the common case), or a list of every reference together
        if there are more (see the class docstring for why a venue would
        have more than one). Raises if `reference_board_path` isn't set or
        any referenced file isn't present locally (none are committed --
        see the class docstring)."""
        import cv2

        if self.reference_board_path is None:
            raise ValueError(f"venue {self.name!r} has no reference_board_path set")

        paths = [self.reference_board_path, *self.additional_reference_board_paths]
        images = []
        for path_str in paths:
            path = Path(path_str)
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"reference board photo not found at {path}")
            images.append(image)

        return images[0] if len(images) == 1 else images

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "corners": [list(c) for c in self.corners],
            "motion_threshold": self.motion_threshold,
            "still_frame_count": self.still_frame_count,
            "still_seconds": self.still_seconds,
            "occupancy_diff_threshold": self.occupancy_diff_threshold,
            "occupancy_gradient_threshold": self.occupancy_gradient_threshold,
            "reference_board_path": self.reference_board_path,
            "additional_reference_board_paths": list(self.additional_reference_board_paths),
            "notes": self.notes,
            "lexicon": self.lexicon,
            "clock_regions": (
                [list(self.clock_regions[0]), list(self.clock_regions[1])]
                if self.clock_regions is not None else None
            ),
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
            still_seconds=data.get("still_seconds"),
            occupancy_diff_threshold=data.get("occupancy_diff_threshold", DEFAULT_DIFF_THRESHOLD),
            occupancy_gradient_threshold=data.get("occupancy_gradient_threshold", DEFAULT_GRADIENT_THRESHOLD),
            reference_board_path=data.get("reference_board_path"),
            additional_reference_board_paths=tuple(data.get("additional_reference_board_paths", [])),
            notes=data.get("notes", ""),
            lexicon=data.get("lexicon"),
            clock_regions=(
                (tuple(data["clock_regions"][0]), tuple(data["clock_regions"][1]))
                if data.get("clock_regions") is not None else None
            ),
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
