"""Synthetic training data for the per-cell letter classifier.

Renders the 27-class problem (A-Z plus a truly blank tile -- see note
below) from the standardized tile font, then augments each render with
perspective/rotation/lighting/blur/compression variation so a classifier
can be pretrained before any real camera or physical board exists.

Important real-world detail baked into this design: a physical blank tile
has NO letter printed on it, even once played -- players/officials track
what it "counts as" separately (scorecard, memory, or a pencil mark
depending on house rules). So the classifier's job is only ever "which
letter glyph is printed here, or is this tile blank" -- determining what a
blank *represents* for scoring is a downstream concern (context/dictionary
inference or an operator flag), not something this classifier can see.
"""
from __future__ import annotations

import io
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from autoscorer.gamelogic.board import LETTER_VALUES

TILE_CLASSES = list(string.ascii_uppercase) + ["BLANK"]

RENDER_SIZE = 240  # supersampled render size, downsampled to FINAL_SIZE for anti-aliasing
FINAL_SIZE = 60  # matches CANONICAL_CELL_PX in perception/calibration/homography.py

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# Approximate premium-square background colors (BGR-agnostic, plain RGB),
# loosely matching the classic US layout. Real editions vary; augmentation
# only needs plausible variety, not exact color-matching.
SQUARE_COLORS = {
    "plain": (222, 217, 201),
    "2L": (173, 216, 230),
    "3L": (65, 132, 180),
    "2W": (245, 176, 176),
    "3W": (205, 50, 50),
}

TILE_FACE_COLORS = [(238, 230, 209), (241, 241, 235)]  # cream, off-white


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render_tile(letter: Optional[str], size: int = RENDER_SIZE, rng: Optional[random.Random] = None) -> Image.Image:
    """Render one crisp synthetic tile face. `letter` is None for a blank
    (unmarked) tile, else a single uppercase A-Z character.
    """
    rng = rng or random.Random()
    face_color = rng.choice(TILE_FACE_COLORS)
    margin = int(size * 0.06)
    tile_box = (margin, margin, size - margin, size - margin)

    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    corner_radius = int(size * 0.08)
    draw.rounded_rectangle(tile_box, radius=corner_radius, fill=face_color, outline=(120, 112, 96), width=max(1, size // 120))

    if letter is not None:
        letter_font = _load_font(int(size * 0.5))
        text_color = (25, 25, 25)
        bbox = draw.textbbox((0, 0), letter, font=letter_font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_x = size / 2 - text_w / 2 - bbox[0]
        text_y = size / 2 - text_h / 2 - bbox[1] - size * 0.03
        draw.text((text_x, text_y), letter, font=letter_font, fill=text_color)

        value = LETTER_VALUES[letter]
        value_font = _load_font(int(size * 0.16))
        value_str = str(value)
        vbbox = draw.textbbox((0, 0), value_str, font=value_font)
        vw, vh = vbbox[2] - vbbox[0], vbbox[3] - vbbox[1]
        draw.text(
            (tile_box[2] - vw - size * 0.1, tile_box[3] - vh - size * 0.08),
            value_str, font=value_font, fill=text_color,
        )

    return img


def _random_background(rng: random.Random, size: int) -> Image.Image:
    color = rng.choice(list(SQUARE_COLORS.values()))
    jitter = tuple(max(0, min(255, c + rng.randint(-10, 10))) for c in color)
    return Image.new("RGB", (size, size), jitter)


def _solve_perspective_coeffs(src: List[Tuple[float, float]], dst: List[Tuple[float, float]]) -> Tuple[float, ...]:
    """The 8 perspective-transform coefficients PIL's `Image.transform(...,
    Image.PERSPECTIVE, coeffs)` expects: for each OUTPUT pixel at a `dst`
    corner, sample the INPUT image at the corresponding `src` corner (PIL's
    own transform is defined output->input) -- shared by every caller that
    needs a perspective warp from one quadrilateral to another, so the
    linear-algebra itself only exists in one place.

    `b` (the right-hand side of the least-squares solve) must be built
    from `src`, not `dst` -- plugging a `dst` corner into the solved
    formula has to yield the matching `src` corner, which is the entire
    point of the mapping. Building `b` from `dst` instead (an earlier bug
    here, found via a visual grid-warp test that came back looking
    completely untransformed) makes the solve trivially return the
    identity transform, since it's then asking "what maps dst back to
    itself" rather than "what maps dst back to src". This stayed
    invisible for a long time because it was only ever exercised with a
    tiny (~5-6%) random jitter (`_perspective_coeffs`, below) -- a no-op
    identity transform and a genuinely-applied tiny jitter look almost
    identical at a glance, so nothing before this actually visually
    verified the warp was doing anything.
    """
    matrix = []
    for (sx, sy), (dx, dy) in zip(dst, src):
        matrix.append([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy])
        matrix.append([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy])
    A = np.array(matrix, dtype=np.float64)
    b = np.array([coord for point in src for coord in point], dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return tuple(coeffs)


def _perspective_coeffs(rng: random.Random, size: int, max_jitter_frac: float = 0.06) -> Tuple[float, ...]:
    j = size * max_jitter_frac
    src = [(0, 0), (size, 0), (size, size), (0, size)]
    dst = [(x + rng.uniform(-j, j), y + rng.uniform(-j, j)) for x, y in src]
    return _solve_perspective_coeffs(src, dst)


def augment_tile(tile_img: Image.Image, rng: Optional[random.Random] = None) -> Image.Image:
    """Apply lighting/rotation/perspective/blur/compression augmentation,
    composite onto a random premium-square-colored background, and
    downsample to FINAL_SIZE."""
    rng = rng or random.Random()
    size = tile_img.size[0]

    background = _random_background(rng, size)
    angle = rng.uniform(-12, 12)
    rotated = tile_img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=None)
    mask = Image.new("L", tile_img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (size * 0.04, size * 0.04, size * 0.96, size * 0.96), radius=int(size * 0.08), fill=255,
    )
    mask = mask.rotate(angle, resample=Image.BICUBIC, fillcolor=0)
    composited = Image.composite(rotated, background, mask)

    coeffs = _perspective_coeffs(rng, size, max_jitter_frac=0.05)
    warped = composited.transform((size, size), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC)

    arr = np.asarray(warped).astype(np.float32)
    brightness = rng.uniform(0.75, 1.25)
    contrast = rng.uniform(0.85, 1.15)
    arr = (arr - 128) * contrast + 128
    arr = arr * brightness
    gamma = rng.uniform(0.85, 1.2)
    arr = 255.0 * np.power(np.clip(arr, 0, 255) / 255.0, gamma)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    lit = Image.fromarray(arr)

    if rng.random() < 0.5:
        lit = lit.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))

    final = lit.resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS)

    if rng.random() < 0.7:
        buf = io.BytesIO()
        final.save(buf, format="JPEG", quality=rng.randint(55, 90))
        buf.seek(0)
        final = Image.open(buf).convert("RGB")

    return final


def augment_real_photo(image: Image.Image, rng: Optional[random.Random] = None) -> Image.Image:
    """Light augmentation for an already-real tile photo (Tier 2 fine-tuning
    data) -- deliberately NOT `augment_tile`'s pipeline, which composites a
    clean synthetic render onto a flat background via a rounded-rect mask;
    running that on a real photo would discard the real camera's actual
    background/edge texture around the tile instead of preserving it. This
    just multiplies a small real dataset with mild rotation/lighting/blur,
    keeping the photo's real content intact.
    """
    rng = rng or random.Random()
    size = image.size[0]

    # Rotate via cv2 with reflected-edge border fill, not PIL's solid-color
    # fillcolor -- a fixed white (or any solid color) wedge in the corners
    # is an artifact the model can partially key on instead of the glyph,
    # and it doesn't resemble anything a real camera would ever produce.
    angle = rng.uniform(-8, 8)
    arr_in = np.array(image)
    center = (size / 2, size / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_arr = cv2.warpAffine(
        arr_in, rotation_matrix, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101,
    )
    rotated = Image.fromarray(rotated_arr)

    arr = np.asarray(rotated).astype(np.float32)
    brightness = rng.uniform(0.85, 1.15)
    contrast = rng.uniform(0.9, 1.1)
    arr = (arr - 128) * contrast + 128
    arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
    lit = Image.fromarray(arr)

    if rng.random() < 0.4:
        lit = lit.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))

    if rng.random() < 0.5:
        buf = io.BytesIO()
        lit.save(buf, format="JPEG", quality=rng.randint(65, 92))
        buf.seek(0)
        lit = Image.open(buf).convert("RGB")

    return lit


@dataclass(frozen=True)
class GeneratedSample:
    label: str
    path: Path


def generate_synthetic_dataset(
    output_dir: Path,
    samples_per_class: int = 200,
    seed: int = 0,
) -> list:
    """Write `samples_per_class` augmented crops per class (A-Z + BLANK)
    to `output_dir/<label>/*.png`. Returns the list of generated samples.
    """
    rng = random.Random(seed)
    output_dir = Path(output_dir)
    samples = []

    for label in TILE_CLASSES:
        letter = None if label == "BLANK" else label
        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        base = render_tile(letter, rng=rng)
        for i in range(samples_per_class):
            augmented = augment_tile(base, rng=rng)
            path = class_dir / f"{label}_{i:04d}.png"
            augmented.save(path)
            samples.append(GeneratedSample(label=label, path=path))

    return samples
