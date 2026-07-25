"""The capture layer the master architecture plan calls for
(`perception/{calibration,capture,occupancy,classify,rack}/`) but that,
until now, didn't exist: every CV component in this repo has only ever
been run against a single still frame decoded by hand from a browser
screenshot. This is the first piece that turns "a validated function"
into "something that can watch a video."

Deliberately reads from a **video file**, not a live camera. A camera
device is a thin, untestable wrapper around the same `cv2.VideoCapture`
API a file already uses (same read-a-frame-in-a-loop shape) -- the real,
non-trivial work is everything downstream that consumes a frame stream
(stillness gating, diffing, scoring), and that's exactly as testable
against a file (including a synthetic one built in a test) as against a
camera. Point this at a recorded broadcast or a live camera's RTSP/device
index equally well; nothing downstream cares which.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np


class VideoFrameSource:
    """Iterates BGR frames from a video file (or any path/index
    `cv2.VideoCapture` accepts, including a camera device index or RTSP
    URL). Supports frame-rate sampling since a broadcast video decoded at
    its native 30/60fps has far more frames than the stillness gate needs
    (`DEFAULT_STILL_FRAME_COUNT=5` consecutive *sampled* frames, not 5
    consecutive encoded frames) -- sampling also keeps CPU/GPU classifier
    calls bounded to a sane rate instead of running on every frame.
    """

    def __init__(self, source: Union[str, Path, int], sample_fps: Optional[float] = None) -> None:
        self._source = str(source) if isinstance(source, Path) else source
        self._capture = cv2.VideoCapture(self._source)
        if not self._capture.isOpened():
            raise ValueError(f"could not open video source: {source!r}")

        self._sample_fps = sample_fps
        self._native_fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_stride = max(1, round(self._native_fps / sample_fps)) if sample_fps else 1

    @property
    def native_fps(self) -> float:
        return self._native_fps

    @property
    def frame_count(self) -> Optional[int]:
        """Total frames in the source, or None if the backend can't report
        it (e.g. a live camera device/stream has no fixed length)."""
        count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else None

    def __iter__(self) -> Iterator[np.ndarray]:
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                return
            if index % self._frame_stride == 0:
                yield frame
            index += 1

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "VideoFrameSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
