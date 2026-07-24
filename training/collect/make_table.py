"""Builds an axis-labeled contact sheet of board cells, for reading ground
truth precisely instead of eyeballing a raw board photo or a hand-built
composite. The first real-data collection pass had an off-by-one row error
caught exactly this way -- read letters off the table this script produces,
not off anything assembled by hand, and cross-check against words visible
in the source image as a sanity check before trusting a batch of labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from autoscorer.perception.calibration.homography import crop_cell

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def make_table(
    rectified_image_path: Path, out_path: Path,
    row_start: int, row_end: int, col_start: int, col_end: int,
    cell_display_size: int = 100, margin: int = 30,
) -> None:
    img = cv2.imread(str(rectified_image_path))
    if img is None:
        raise ValueError(f"could not read image at {rectified_image_path}")

    rows = list(range(row_start, row_end))
    cols = list(range(col_start, col_end))
    width = margin + len(cols) * cell_display_size
    height = margin + len(rows) * cell_display_size

    sheet = Image.new("RGB", (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(sheet)
    font = _load_font(16)

    for ci, c in enumerate(cols):
        draw.text((margin + ci * cell_display_size + cell_display_size // 2 - 5, 5), str(c), font=font, fill=(255, 255, 0))
    for ri, r in enumerate(rows):
        draw.text((5, margin + ri * cell_display_size + cell_display_size // 2 - 8), str(r), font=font, fill=(255, 255, 0))

    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            crop = crop_cell(img, r, c)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tile_img = Image.fromarray(crop_rgb).resize((cell_display_size - 2, cell_display_size - 2))
            sheet.paste(tile_img, (margin + ci * cell_display_size, margin + ri * cell_display_size))

    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rectified_image", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("row_start", type=int)
    parser.add_argument("row_end", type=int)
    parser.add_argument("col_start", type=int)
    parser.add_argument("col_end", type=int)
    args = parser.parse_args()

    make_table(args.rectified_image, args.out_path, args.row_start, args.row_end, args.col_start, args.col_end)
    print(f"saved {args.out_path}")


if __name__ == "__main__":
    main()
