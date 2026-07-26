import cv2
import numpy as np
import pytest

from autoscorer.perception.calibration.homography import CANONICAL_SIZE
from autoscorer.perception.calibration.venue_profile import (
    VenueProfile,
    list_venue_profiles,
    load_venue_profile,
    save_venue_profile,
)
from autoscorer.perception.occupancy.detector import DEFAULT_DIFF_THRESHOLD, DEFAULT_GRADIENT_THRESHOLD


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


def test_real_wespa_profile_has_occupancy_thresholds_tuned_away_from_naspa_defaults():
    # Confirms the real, measured fix stuck: NASPA-tuned defaults caused
    # 47/225 real empty WESPA cells to score as spuriously occupied (busy
    # premium-square graphics the defaults never had to reject) -- this
    # profile should carry venue-specific values, not silently fall back
    # to the defaults tuned for a different board's graphics.
    profile = load_venue_profile("wespa_word_wars")
    assert profile.occupancy_diff_threshold != DEFAULT_DIFF_THRESHOLD
    assert profile.occupancy_gradient_threshold != DEFAULT_GRADIENT_THRESHOLD


def test_occupancy_thresholds_default_to_module_defaults_when_unset():
    profile = _sample_profile()
    assert profile.occupancy_diff_threshold == DEFAULT_DIFF_THRESHOLD
    assert profile.occupancy_gradient_threshold == DEFAULT_GRADIENT_THRESHOLD
    assert profile.reference_board_path is None


def test_load_reference_board_raises_when_path_not_set():
    profile = _sample_profile()
    with pytest.raises(ValueError):
        profile.load_reference_board()


def test_lexicon_defaults_to_none():
    profile = _sample_profile()
    assert profile.lexicon is None


def test_lexicon_round_trips_through_save_and_load(tmp_path):
    profile = VenueProfile(
        name="csw_venue",
        corners=((0.0, 0.0), (500.0, 0.0), (500.0, 500.0), (0.0, 500.0)),
        lexicon="csw24",
    )
    save_venue_profile(profile, directory=tmp_path)
    loaded = load_venue_profile("csw_venue", directory=tmp_path)
    assert loaded.lexicon == "csw24"


def test_a_saved_profile_without_a_lexicon_key_still_loads(tmp_path):
    """Older saved profiles (including the committed
    configs/venues/wespa_word_wars.json) predate this field -- loading
    one must not require it."""
    import json
    profile = _sample_profile("legacy_venue")
    data = profile.to_dict()
    del data["lexicon"]
    (tmp_path / "legacy_venue.json").write_text(json.dumps(data))

    loaded = load_venue_profile("legacy_venue", directory=tmp_path)
    assert loaded.lexicon is None


def test_load_reference_board_raises_when_file_missing(tmp_path):
    profile = VenueProfile(
        name="test_venue",
        corners=((10.0, 20.0), (500.0, 15.0), (510.0, 480.0), (5.0, 490.0)),
        reference_board_path=str(tmp_path / "does_not_exist.jpg"),
    )
    with pytest.raises(FileNotFoundError):
        profile.load_reference_board()


def test_load_reference_board_reads_the_saved_image(tmp_path):
    image = np.full((60, 80, 3), 123, dtype=np.uint8)
    image_path = tmp_path / "reference.jpg"
    cv2.imwrite(str(image_path), image)

    profile = VenueProfile(
        name="test_venue",
        corners=((10.0, 20.0), (500.0, 15.0), (510.0, 480.0), (5.0, 490.0)),
        reference_board_path=str(image_path),
    )
    loaded = profile.load_reference_board()
    assert loaded.shape == (60, 80, 3)


def test_load_reference_board_with_additional_paths_returns_a_list(tmp_path):
    primary = np.full((60, 80, 3), 100, dtype=np.uint8)
    secondary = np.full((60, 80, 3), 200, dtype=np.uint8)
    primary_path = tmp_path / "primary.jpg"
    secondary_path = tmp_path / "secondary.jpg"
    cv2.imwrite(str(primary_path), primary)
    cv2.imwrite(str(secondary_path), secondary)

    profile = VenueProfile(
        name="test_venue",
        corners=((10.0, 20.0), (500.0, 15.0), (510.0, 480.0), (5.0, 490.0)),
        reference_board_path=str(primary_path),
        additional_reference_board_paths=(str(secondary_path),),
    )
    loaded = profile.load_reference_board()
    assert isinstance(loaded, list)
    assert len(loaded) == 2
    assert loaded[0].mean() == 100
    assert loaded[1].mean() == 200


def test_additional_reference_board_paths_round_trips_through_save_and_load(tmp_path):
    profile = VenueProfile(
        name="test_venue",
        corners=((10.0, 20.0), (500.0, 15.0), (510.0, 480.0), (5.0, 490.0)),
        reference_board_path="primary.jpg",
        additional_reference_board_paths=("alt1.jpg", "alt2.jpg"),
    )
    save_venue_profile(profile, directory=tmp_path)
    loaded = load_venue_profile("test_venue", directory=tmp_path)
    assert loaded == profile
    assert loaded.additional_reference_board_paths == ("alt1.jpg", "alt2.jpg")
