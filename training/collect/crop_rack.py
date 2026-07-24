"""Crops individual rack tiles given manually-identified pixel groups.

Real physical rack tiles are NOT evenly spaced across a whole rack -- players
leave gaps, so a fixed division of the entire rack width produces garbage for
any rack that isn't completely full and contiguous. This instead takes one or
more (x1, x2, letters) groups, each a single *contiguous* cluster of tiles,
and evenly divides only within each group. A rack showing "BATE  ELS" (a gap
in the middle) needs two groups, not one covering the whole rack.

Automatic tile-boundary detection was tried first (contour detection on the
rack strip) and rejected: touching tiles merge into one blob under simple
thresholding, so there's no cheap way to find individual tile edges within a
contiguous group automatically. A real fix needs object detection (RF-DETR)
trained on bounding-box-labeled rack images, which doesn't exist yet --
manual grouping is a stand-in for data collection purposes only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2


def parse_groups(args: List[str]) -> List[Tuple[int, int, List[str]]]:
    if len(args) % 3 != 0:
        raise ValueError("groups must come in (x1, x2, letters) triples")
    groups = []
    for i in range(0, len(args), 3):
        x1, x2, letters = int(args[i]), int(args[i + 1]), args[i + 2].split(",")
        groups.append((x1, x2, letters))
    return groups


def crop_rack(image_path: Path, out_dir: Path, prefix: str, groups: List[Tuple[int, int, List[str]]]) -> int:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"could not read image at {image_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for x1, x2, letters in groups:
        step = (x2 - x1) / len(letters)
        for i, letter in enumerate(letters):
            slot_x1, slot_x2 = int(x1 + i * step), int(x1 + (i + 1) * step)
            crop = img[:, slot_x1:slot_x2]
            cv2.imwrite(str(out_dir / f"{letter}_{prefix}_{count}.png"), crop)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: crop_rack.py rack.jpg out/ f2L 95 450 B,A,T,E 515 785 E,L,S",
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("prefix", help="Short tag identifying this frame/rack, used in output filenames")
    parser.add_argument(
        "groups", nargs="+",
        help="Repeated (x1 x2 letters) triples, one per contiguous tile cluster, e.g. 95 450 B,A,T,E",
    )
    args = parser.parse_args()

    groups = parse_groups(args.groups)
    count = crop_rack(args.image_path, args.out_dir, args.prefix, groups)
    print(f"wrote {count} rack tile crops")


if __name__ == "__main__":
    main()
