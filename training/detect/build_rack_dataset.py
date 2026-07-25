"""Assembles a Roboflow-standard COCO-format object-detection dataset for
the rack-tile detector: unlimited synthetic rack scenes
(`rack_scene_renderer.generate_rack_scene`) as the bulk of train/valid/test,
plus optionally a handful of real rack photos (reconstructed via
`reconstruct_rack_boxes.py`) folded into train only -- real data is scarce
enough here that none of it should be spent on validation yet; validating
real-world performance should come from deploying against fresh real
frames later, the same pattern WS3 used for the tile classifier.

Directory layout matches what RF-DETR's `build_roboflow_from_coco` expects:
    dataset_dir/{train,valid,test}/{*.jpg, _annotations.coco.json}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from training.synth_render.rack_scene_renderer import TILE_CLASSES, TileBox, compress_like_a_real_photo, generate_rack_scene

CATEGORIES = [{"id": i + 1, "name": name, "supercategory": "tile"} for i, name in enumerate(TILE_CLASSES)]
_CATEGORY_ID_BY_NAME = {c["name"]: c["id"] for c in CATEGORIES}

# (BGR image array, boxes, file_name) -- file_name distinguishes synthetic
# from real entries in one split without either overwriting the other.
SceneEntry = Tuple[np.ndarray, Sequence[TileBox], str]


def _pil_scene_to_bgr(scene_image, rng: random.Random) -> np.ndarray:
    compressed = compress_like_a_real_photo(scene_image, rng)
    return np.array(compressed)[:, :, ::-1]  # RGB -> BGR for cv2.imwrite


def make_synthetic_entries(count: int, rng: random.Random, prefix: str = "synth") -> List[SceneEntry]:
    entries: List[SceneEntry] = []
    for i in range(count):
        scene = generate_rack_scene(rng=rng)
        image = _pil_scene_to_bgr(scene.image, rng)
        entries.append((image, scene.boxes, f"{prefix}_{i:05d}.jpg"))
    return entries


def write_split(split_dir: Path, entries: Sequence[SceneEntry]) -> None:
    """Writes one train/valid/test split: every image in `entries` plus a
    single `_annotations.coco.json` covering all of them. Overwrites
    anything already in `split_dir` -- callers combining synthetic and
    real entries should pass them together in one call, not two.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1

    for image_id, (image, boxes, file_name) in enumerate(entries, start=1):
        cv2.imwrite(str(split_dir / file_name), image)
        height, width = image.shape[:2]
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

        for box in boxes:
            w, h = box.x2 - box.x1, box.y2 - box.y1
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": _CATEGORY_ID_BY_NAME[box.label],
                "bbox": [box.x1, box.y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))


def build_dataset(
    out_dir: Path,
    num_train_synthetic: int = 800,
    num_valid_synthetic: int = 100,
    num_test_synthetic: int = 100,
    real_train_entries: Optional[List[SceneEntry]] = None,
    seed: int = 0,
) -> None:
    rng = random.Random(seed)

    train_entries = make_synthetic_entries(num_train_synthetic, rng)
    if real_train_entries:
        train_entries = train_entries + list(real_train_entries)

    valid_entries = make_synthetic_entries(num_valid_synthetic, rng, prefix="synth_valid")
    test_entries = make_synthetic_entries(num_test_synthetic, rng, prefix="synth_test")

    write_split(out_dir / "train", train_entries)
    write_split(out_dir / "valid", valid_entries)
    write_split(out_dir / "test", test_entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--train", type=int, default=800)
    parser.add_argument("--valid", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    build_dataset(args.out_dir, args.train, args.valid, args.test, seed=args.seed)
    print(f"wrote synthetic-only dataset to {args.out_dir} "
          f"({args.train} train / {args.valid} valid / {args.test} test)")


if __name__ == "__main__":
    main()
