import random

from training.synth_render.rack_scene_renderer import (
    RACK_HEIGHT,
    RACK_WIDTH,
    generate_rack_scene,
)


def test_generates_the_requested_number_of_tiles_and_boxes():
    rng = random.Random(0)
    scene = generate_rack_scene(rng=rng, tile_count=5)
    assert len(scene.boxes) == 5


def test_every_box_is_within_image_bounds_and_non_degenerate():
    rng = random.Random(1)
    for _ in range(20):
        scene = generate_rack_scene(rng=rng, tile_count=rng.randint(1, 7))
        for box in scene.boxes:
            assert 0 <= box.x1 < box.x2 <= RACK_WIDTH
            assert 0 <= box.y1 < box.y2 <= RACK_HEIGHT
            assert box.label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or box.label == "BLANK"


def test_boxes_are_left_to_right_in_placement_order():
    # Tiles are placed in a strictly increasing x position (groups plus
    # gaps), so box centers should come out in non-decreasing x order --
    # a sanity check that box coordinates track actual placement, not some
    # constant/degenerate value.
    rng = random.Random(2)
    scene = generate_rack_scene(rng=rng, tile_count=6)
    centers = [(b.x1 + b.x2) / 2 for b in scene.boxes]
    assert centers == sorted(centers)


def test_blank_tiles_are_labeled_blank():
    rng = random.Random(3)
    scene = generate_rack_scene(rng=rng, tile_count=7, blank_probability=1.0)
    assert all(box.label == "BLANK" for box in scene.boxes)


def test_zero_blank_probability_never_produces_blanks():
    rng = random.Random(4)
    for _ in range(10):
        scene = generate_rack_scene(rng=rng, tile_count=5, blank_probability=0.0)
        assert all(box.label != "BLANK" for box in scene.boxes)


def test_scene_image_has_the_expected_size():
    rng = random.Random(5)
    scene = generate_rack_scene(rng=rng, tile_count=3)
    assert scene.image.size == (RACK_WIDTH, RACK_HEIGHT)


def test_generation_is_deterministic_given_the_same_seeded_rng():
    scene_a = generate_rack_scene(rng=random.Random(99), tile_count=4)
    scene_b = generate_rack_scene(rng=random.Random(99), tile_count=4)
    assert scene_a.boxes == scene_b.boxes
