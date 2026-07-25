import json
import random

import numpy as np

from training.detect.build_rack_dataset import (
    CATEGORIES,
    build_dataset,
    make_synthetic_entries,
    write_split,
)
from training.synth_render.rack_scene_renderer import TileBox


def test_make_synthetic_entries_returns_the_requested_count():
    rng = random.Random(0)
    entries = make_synthetic_entries(5, rng)
    assert len(entries) == 5
    for image, boxes, file_name in entries:
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert len(boxes) >= 1
        assert file_name.endswith(".jpg")


def test_write_split_produces_valid_coco_json(tmp_path):
    rng = random.Random(1)
    entries = make_synthetic_entries(3, rng)
    split_dir = tmp_path / "train"
    write_split(split_dir, entries)

    coco_path = split_dir / "_annotations.coco.json"
    assert coco_path.exists()
    coco = json.loads(coco_path.read_text())

    assert len(coco["images"]) == 3
    assert coco["categories"] == CATEGORIES
    assert len(coco["annotations"]) == sum(len(boxes) for _, boxes, _ in entries)

    # Every image file COCO references actually exists on disk.
    for img_record in coco["images"]:
        assert (split_dir / img_record["file_name"]).exists()

    # Every annotation's bbox is well-formed and its image_id resolves.
    image_ids = {img["id"] for img in coco["images"]}
    for ann in coco["annotations"]:
        assert ann["image_id"] in image_ids
        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0
        assert ann["area"] == w * h


def test_build_dataset_writes_all_three_splits(tmp_path):
    build_dataset(tmp_path, num_train_synthetic=4, num_valid_synthetic=2, num_test_synthetic=2, seed=7)

    for split in ("train", "valid", "test"):
        coco_path = tmp_path / split / "_annotations.coco.json"
        assert coco_path.exists()
        coco = json.loads(coco_path.read_text())
        assert len(coco["images"]) > 0


def test_build_dataset_folds_real_entries_into_train_only(tmp_path):
    real_image = np.zeros((150, 900, 3), dtype=np.uint8)
    real_boxes = [TileBox(label="Q", x1=10, y1=20, x2=100, y2=110)]
    real_entries = [(real_image, real_boxes, "real_0000.jpg")]

    build_dataset(
        tmp_path, num_train_synthetic=3, num_valid_synthetic=2, num_test_synthetic=2,
        real_train_entries=real_entries, seed=3,
    )

    train_coco = json.loads((tmp_path / "train" / "_annotations.coco.json").read_text())
    valid_coco = json.loads((tmp_path / "valid" / "_annotations.coco.json").read_text())

    assert any(img["file_name"] == "real_0000.jpg" for img in train_coco["images"])
    assert len(train_coco["images"]) == 4  # 3 synthetic + 1 real
    assert not any(img["file_name"].startswith("real_") for img in valid_coco["images"])
