import numpy as np
from PIL import Image, ImageDraw, ImageFont

from autoscorer.perception.classify.canonicalize import FINAL_SIZE, canonicalize, canonicalize_pil


def _render_letter(letter: str, glyph_color, bg_color, size: int = 120) -> Image.Image:
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=int(size * 0.6))
    bbox = draw.textbbox((0, 0), letter, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((size / 2 - w / 2 - bbox[0], size / 2 - h / 2 - bbox[1]), letter, font=font, fill=glyph_color)
    return img


def test_output_shape_and_dtype():
    img = _render_letter("A", (0, 0, 0), (230, 230, 230))
    out = canonicalize(img)
    assert out.shape == (FINAL_SIZE, FINAL_SIZE)
    assert out.dtype == np.uint8


def test_polarity_normalized_dark_on_light_stays_dark_on_light():
    dark_on_light = _render_letter("A", (10, 10, 10), (240, 240, 240))
    out = canonicalize(dark_on_light)
    # Center (glyph) should end up darker than the border (background).
    h, w = out.shape
    center = out[h // 3:2 * h // 3, w // 3:2 * w // 3].mean()
    border = out.mean()
    assert center < border


def test_polarity_normalized_light_on_dark_gets_inverted_to_match():
    light_on_dark = _render_letter("A", (245, 245, 245), (15, 15, 15))
    out = canonicalize(light_on_dark)
    h, w = out.shape
    center = out[h // 3:2 * h // 3, w // 3:2 * w // 3].mean()
    border = out.mean()
    # After canonicalization this should look like the dark-on-light case:
    # glyph (center) darker than surrounding background, regardless of the
    # original tile's real-world color scheme.
    assert center < border


def test_canonicalize_pil_returns_grayscale_pil_image():
    img = _render_letter("Z", (0, 0, 0), (255, 255, 255))
    out = canonicalize_pil(img)
    assert out.mode == "L"
    assert out.size == (FINAL_SIZE, FINAL_SIZE)


def test_loosely_bounded_crop_gets_tightened_around_the_tile():
    # Simulate a rack-style crop: the actual tile occupies a small portion
    # of a much larger frame with padding on all sides (this is exactly the
    # framing mismatch that measurably hurt accuracy on manually-bounded
    # rack crops vs. the calibrated per-cell board crops).
    tile = _render_letter("A", (10, 10, 10), (235, 235, 235), size=60)
    padded = Image.new("RGB", (240, 240), (200, 170, 120))  # wood-tray-ish color
    padded.paste(tile, (90, 90))

    tight = _render_letter("A", (10, 10, 10), (235, 235, 235), size=60)

    out_loose = canonicalize(padded)
    out_tight = canonicalize(tight)

    # Both should end up with the glyph occupying a similar, non-trivial
    # fraction of the frame -- i.e. the loose crop's glyph shouldn't be left
    # tiny relative to the tightly-bounded one.
    def glyph_fraction(arr):
        return float((arr < arr.mean()).mean())

    loose_frac = glyph_fraction(out_loose)
    tight_frac = glyph_fraction(out_tight)
    assert abs(loose_frac - tight_frac) < 0.25
