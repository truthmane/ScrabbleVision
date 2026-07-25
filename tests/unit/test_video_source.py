import numpy as np
import pytest

from autoscorer.perception.capture.video_source import VideoFrameSource

cv2 = pytest.importorskip("cv2")


_STEP = 16  # coarse enough to survive mp4v's lossy quantization between frames


def _write_test_video(path, num_frames: int, fps: float = 10.0, size=(64, 48)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for i in range(num_frames):
        # A distinct, verifiable-within-tolerance pixel value per frame --
        # lets a test check frames come back in order and none are
        # dropped/duplicated, not just that *some* frames came back. Uses a
        # tolerance band (not exact equality) since mp4v is lossy and won't
        # round-trip an exact byte value.
        frame = np.full((size[1], size[0], 3), (i * _STEP) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _approx_frame_index(frame) -> int:
    return round(int(frame[0, 0, 0]) / _STEP)


def test_iterates_every_frame_at_native_rate(tmp_path):
    video_path = tmp_path / "test.mp4"
    _write_test_video(video_path, num_frames=15, fps=10.0)

    source = VideoFrameSource(video_path)
    frames = list(source)

    assert len(frames) == 15
    # Each frame's fill value should match its position (within lossy
    # compression tolerance), confirming order and no dropped frames.
    for i, frame in enumerate(frames):
        assert _approx_frame_index(frame) == i


def test_sample_fps_skips_frames_proportionally(tmp_path):
    video_path = tmp_path / "test.mp4"
    _write_test_video(video_path, num_frames=30, fps=30.0)

    # Sampling at 10fps from a 30fps video should yield roughly a third
    # of the frames (every 3rd frame), not all 30.
    source = VideoFrameSource(video_path, sample_fps=10.0)
    frames = list(source)

    assert 8 <= len(frames) <= 12
    # Confirms it's actually skipping (stride 3), not just truncating --
    # the first few sampled frames should be 0, 3, 6, ...
    assert _approx_frame_index(frames[0]) == 0
    assert _approx_frame_index(frames[1]) == 3


def test_raises_on_missing_file(tmp_path):
    with pytest.raises(ValueError):
        VideoFrameSource(tmp_path / "does_not_exist.mp4")


def test_context_manager_releases_capture(tmp_path):
    video_path = tmp_path / "test.mp4"
    _write_test_video(video_path, num_frames=5, fps=10.0)

    with VideoFrameSource(video_path) as source:
        frames = list(source)
    assert len(frames) == 5
