import numpy as np

from autoscorer.perception.calibration.homography import CANONICAL_SIZE
from autoscorer.perception.calibration.venue_profile import (
    VenueProfile,
    list_venue_profiles,
    load_venue_profile,
    save_venue_profile,
)


def _sample_profile(name="test_venue") -> VenueProfile:
    return VenueProfile(
        name=name,
        corners=((10.0, 20.0), (500.0, 15.0), (510.0, 480.0), (5.0, 490.0)),
        motion_threshold=12.5,
        still_frame_count=4,
        notes="a test venue",
    )


def test_save_and_load_round_trips_all_fields(tmp_path):
    profile = _sample_profile()
    save_venue_profile(profile, directory=tmp_path)

    loaded = load_venue_profile("test_venue", directory=tmp_path)

    assert loaded == profile


def test_load_raises_for_unknown_venue(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_venue_profile("nonexistent", directory=tmp_path)


def test_list_venue_profiles_returns_saved_names_sorted(tmp_path):
    save_venue_profile(_sample_profile("zeta"), directory=tmp_path)
    save_venue_profile(_sample_profile("alpha"), directory=tmp_path)

    assert list_venue_profiles(tmp_path) == ["alpha", "zeta"]


def test_list_venue_profiles_empty_directory_returns_empty_list(tmp_path):
    assert list_venue_profiles(tmp_path / "does_not_exist") == []


def test_calibration_produces_a_working_rectifier():
    profile = _sample_profile()
    calibration = profile.calibration()

    image = np.zeros((500, 520, 3), dtype=np.uint8)
    rectified = calibration.rectify(image)

    assert rectified.shape == (CANONICAL_SIZE, CANONICAL_SIZE, 3)


def test_real_wespa_profile_is_saved_and_loadable():
    # The actual venue this project has real, verified corners for --
    # pins that configs/venues/wespa_word_wars.json exists and is valid,
    # not just that the save/load machinery works on throwaway data.
    profile = load_venue_profile("wespa_word_wars")
    assert profile.name == "wespa_word_wars"
    assert len(profile.corners) == 4
    calibration = profile.calibration()
    assert calibration.homography.shape == (3, 3)
