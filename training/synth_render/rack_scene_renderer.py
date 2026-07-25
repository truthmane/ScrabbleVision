"""Synthetic rack-scene generator: composes multiple rendered tiles onto a
rack-strip background with realistic uneven spacing, recording each
tile's exact bounding box. This is training data for an object detector
(RF-DETR) that localizes individual rack tiles -- the piece the master
architecture plan calls for in Phase 5 and `docs/classifier-accuracy-plan.md`
flags as a genuine open-layout detection problem: real rack tiles are NOT
evenly spaced (players group letters, leave gaps for tiles they're
considering exchanging), so the board's fixed-grid `crop_cell` approach
never applied to racks -- see `training/collect/crop_rack.py`'s docstring,
which used manual per-group pixel ranges as a stand-in for exactly this.

Bootstraps the same way `tile_renderer.py` bootstrapped the classifier:
unlimited free synthetic data now, real photos fine-tune later (see
`training/detect/build_rack_dataset.py`, which reconstructs real
bounding boxes from the rack photos already harvested for WS3).
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from training.synth_render.tile_renderer import TILE_CLASSES, render_tile

RACK_WIDTH = 900
RACK_HEIGHT = 160
TILE_RENDER_SIZE = 100
# Rack ledges are usually a warm wood tone; real ones vary (light maple to
# dark walnut, sometimes black plastic) -- augmentation only needs
# plausible variety, not exact color-matching, same philosophy as the
# board's SQUARE_COLORS.
WOOD_COLORS = [(181, 136, 99), (161, 116, 79), (198, 152, 111), (140, 100, 70)]


@dataclass(frozen=True)
class TileBox:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class RackScene:
    image: Image.Image
    boxes: List[TileBox]


def _rack_background(rng: random.Random, width: int = RACK_WIDTH, height: int = RACK_HEIGHT) -> Image.Image:
    wood = rng.choice(WOOD_COLORS)
    img = Image.new("RGB", (width, height), wood)
    draw = ImageDraw.Draw(img)
    # Horizontal wood-grain streaks: a handful of slightly darker/lighter
    # lines running the width of the rack, not perfectly straight.
    for _ in range(rng.randint(8, 16)):
        y = rng.uniform(0, height)
        shade = rng.randint(-25, 15)
        line_color = tuple(max(0, min(255, c + shade)) for c in wood)
        points = [(x, y + rng.uniform(-3, 3)) for x in range(0, width + 1, 40)]
        draw.line(points, fill=line_color, width=rng.choice([1, 1, 2]))
    return img


def _tile_with_mask(letter: Optional[str], rng: random.Random) -> Tuple[Image.Image, Image.Image]:
    """A rendered tile plus a matching alpha mask (255 inside the rounded
    tile shape, 0 outside) at `TILE_RENDER_SIZE`, lightly rotated -- same
    small-angle jitter `tile_renderer.augment_tile` uses for board tiles."""
    tile = render_tile(letter, size=TILE_RENDER_SIZE, rng=rng)
    mask = Image.new("L", tile.size, 0)
    margin = TILE_RENDER_SIZE * 0.06
    ImageDraw.Draw(mask).rounded_rectangle(
        (margin, margin, TILE_RENDER_SIZE - margin, TILE_RENDER_SIZE - margin),
        radius=int(TILE_RENDER_SIZE * 0.08), fill=255,
    )

    angle = rng.uniform(-10, 10)
    tile = tile.rotate(angle, resample=Image.BICUBIC, fillcolor=None)
    mask = mask.rotate(angle, resample=Image.BICUBIC, fillcolor=0)

    arr = np.asarray(tile).astype(np.float32)
    brightness = rng.uniform(0.8, 1.2)
    contrast = rng.uniform(0.88, 1.12)
    arr = (arr - 128) * contrast + 128
    arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
    tile = Image.fromarray(arr)

    return tile, mask


def _split_into_groups(tile_count: int, rng: random.Random) -> List[int]:
    """How many tiles fall in each contiguous group, e.g. 7 tiles might
    split as [4, 3] (one gap) or [2, 3, 2] (two gaps) -- mirrors how a
    real rack often shows one or two visible gaps, not perfectly even
    spacing across all 7 slots."""
    max_groups = min(3, tile_count)
    num_groups = rng.randint(1, max_groups)
    if num_groups == 1:
        return [tile_count]
    # Distribute tile_count into num_groups positive-sized groups.
    cuts = sorted(rng.sample(range(1, tile_count), num_groups - 1))
    bounds = [0] + cuts + [tile_count]
    return [bounds[i + 1] - bounds[i] for i in range(num_groups)]


def generate_rack_scene(
    rng: Optional[random.Random] = None,
    tile_count: Optional[int] = None,
    blank_probability: float = 0.05,
) -> RackScene:
    """One synthetic rack photo: 1-7 tiles (unless `tile_count` is given),
    split into 1-3 contiguous groups with gaps between them, each tile
    independently rotated/lit, composited onto a wood-grain background,
    with a tight bounding box recorded per tile.
    """
    rng = rng or random.Random()
    tile_count = tile_count if tile_count is not None else rng.randint(1, 7)

    background = _rack_background(rng)
    groups = _split_into_groups(tile_count, rng)

    letters: List[Optional[str]] = []
    for _ in range(tile_count):
        if rng.random() < blank_probability:
            letters.append(None)
        else:
            letters.append(rng.choice([c for c in TILE_CLASSES if c != "BLANK"]))

    boxes: List[TileBox] = []
    letter_iter = iter(letters)

    within_group_gap = lambda: rng.randint(-4, 6)  # tiles in a group nearly touch, sometimes overlap a hair
    between_group_gap = lambda: rng.randint(50, 160)

    x = rng.randint(10, 60)
    baseline_y = rng.randint(20, RACK_HEIGHT - TILE_RENDER_SIZE - 20)

    for group_idx, group_size in enumerate(groups):
        for i in range(group_size):
            letter = next(letter_iter)
            tile_img, mask = _tile_with_mask(letter, rng)

            y = baseline_y + rng.randint(-6, 6)
            if x + TILE_RENDER_SIZE > RACK_WIDTH - 10:
                break  # ran out of room (shouldn't happen at realistic counts, but stay safe)

            background.paste(tile_img, (x, y), mask)

            local_bbox = mask.getbbox()
            if local_bbox is not None:
                lx1, ly1, lx2, ly2 = local_bbox
                label = "BLANK" if letter is None else letter
                boxes.append(TileBox(label=label, x1=x + lx1, y1=y + ly1, x2=x + lx2, y2=y + ly2))

            x += TILE_RENDER_SIZE + within_group_gap()

        if group_idx < len(groups) - 1:
            x += between_group_gap()

    return RackScene(image=background, boxes=boxes)


def compress_like_a_real_photo(image: Image.Image, rng: random.Random) -> Image.Image:
    """Whole-scene JPEG re-encode + optional slight blur, applied once to
    the finished composite -- a real photo is compressed as a whole frame,
    not tile-by-tile, unlike the per-tile classifier's augmentation."""
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
    if rng.random() < 0.8:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=rng.randint(55, 92))
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
    return image
