import pytest

rfdetr = pytest.importorskip("rfdetr", reason="rfdetr[train] not installed")

from training.detect.train_rack_detector import MODEL_CLASSES, build_model


def test_model_classes_covers_the_documented_sizes():
    assert set(MODEL_CLASSES) == {"nano", "small", "medium", "base", "large"}
    for size, class_name in MODEL_CLASSES.items():
        assert hasattr(rfdetr, class_name), f"{size!r} -> rfdetr.{class_name} doesn't exist"


def test_build_model_returns_the_matching_rfdetr_class():
    # Only "nano" -- instantiating each size downloads/builds a real
    # pretrained backbone, which is slow and needs network access; the
    # class-name wiring itself is already checked without that cost above.
    model = build_model("nano")
    assert type(model).__name__ == MODEL_CLASSES["nano"]


def test_build_model_rejects_unknown_size():
    with pytest.raises(KeyError):
        build_model("gigantic")
