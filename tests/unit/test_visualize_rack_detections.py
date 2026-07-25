import numpy as np
import pytest

sv = pytest.importorskip("supervision", reason="supervision (an rfdetr[train] dependency) not installed")

from training.detect.visualize_rack_detections import draw_detections


def test_draw_detections_draws_one_box_per_detection():
    image = np.zeros((150, 900, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[10.0, 20.0, 100.0, 120.0], [200.0, 15.0, 290.0, 110.0]]),
        confidence=np.array([0.91, 0.42]),
        class_id=np.array([0, 25]),  # "A", "Z" if class_names is A..Z+BLANK
    )
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["BLANK"]

    annotated = draw_detections(image, detections, class_names)

    assert annotated.shape == image.shape
    # The original must be untouched (draw_detections returns a copy).
    assert np.array_equal(image, np.zeros((150, 900, 3), dtype=np.uint8))
    # Something was actually drawn -- the annotated image differs from blank.
    assert not np.array_equal(annotated, image)


def test_draw_detections_handles_no_detections():
    image = np.zeros((150, 900, 3), dtype=np.uint8)
    detections = sv.Detections.empty()
    class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["BLANK"]

    annotated = draw_detections(image, detections, class_names)
    assert np.array_equal(annotated, image)
