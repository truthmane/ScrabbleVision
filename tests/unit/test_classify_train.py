"""Smoke tests for the classifier training pipeline -- deliberately tiny
(few classes, few samples, few epochs) so this stays fast; this validates
plumbing correctness, not real-world model accuracy (that needs the full
synthetic set plus real fine-tuning data, run separately as an actual
training job, not a unit test).
"""
import random

import pytest
import torch

from training.classify.infer import TileClassifierModel
from training.classify.model import TileClassifier
from training.classify.train import evaluate, run_training, save_checkpoint
from training.synth_render.tile_renderer import augment_tile, render_tile


def _tiny_dataset(tmp_path, labels=("A", "B", "BLANK"), samples_per_class=30, seed=0):
    rng = random.Random(seed)
    for label in labels:
        letter = None if label == "BLANK" else label
        class_dir = tmp_path / label
        class_dir.mkdir(parents=True, exist_ok=True)
        base = render_tile(letter, rng=rng)
        for i in range(samples_per_class):
            augment_tile(base, rng=rng).save(class_dir / f"{i:03d}.png")
    return tmp_path


def test_training_pipeline_runs_and_learns_on_a_tiny_dataset(tmp_path):
    data_dir = _tiny_dataset(tmp_path)

    model, result = run_training(data_dir, epochs=25, batch_size=8, device=torch.device("cpu"), seed=0)

    assert set(result.classes) == {"A", "B", "BLANK"}
    # Three well-separated synthetic classes with plenty of samples should
    # be easily learnable -- a low accuracy here would indicate a real
    # plumbing bug (wrong labels, broken loss, etc.), not just "hard data".
    assert result.final_val_accuracy > 0.8


def test_checkpoint_round_trip_and_inference(tmp_path):
    data_dir = _tiny_dataset(tmp_path)
    model, result = run_training(data_dir, epochs=25, batch_size=8, device=torch.device("cpu"), seed=0)

    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(model, result.classes, checkpoint_path)

    loaded = TileClassifierModel(checkpoint_path, device="cpu")
    assert set(loaded.classes) == {"A", "B", "BLANK"}

    sample = render_tile("A", rng=random.Random(99))
    augmented = augment_tile(sample, rng=random.Random(99))
    label, confidence = loaded.predict(augmented)
    assert label in loaded.classes
    assert 0.0 <= confidence <= 1.0


def test_predict_topk_batch_matches_individual_predict_topk_calls(tmp_path):
    """Regression test for the classifier-call batching added to speed up
    the live pipeline's documented ~5s/settled-frame bottleneck
    (board_reader.read_new_cells_voted calls the classifier once per
    occupied cell per window frame -- batching all of those into one
    forward pass is only safe if it can't change any individual result).
    The model runs in eval mode, so BatchNorm uses its stored running
    statistics rather than the current batch's -- one image's result
    should never depend on what else shares its batch. Uses several
    different letters, not just repeats of one, so a batching bug that
    only shows up when classes differ within a batch would still be
    caught.
    """
    data_dir = _tiny_dataset(tmp_path)
    model, result = run_training(data_dir, epochs=25, batch_size=8, device=torch.device("cpu"), seed=0)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(model, result.classes, checkpoint_path)
    loaded = TileClassifierModel(checkpoint_path, device="cpu")

    rng = random.Random(42)
    images = [
        augment_tile(render_tile(letter, rng=rng), rng=rng)
        for letter in ["A", "B", None, "A", "B"]
    ]

    individually = [loaded.predict_topk(img, k=3) for img in images]
    batched = loaded.predict_topk_batch(images, k=3)

    assert len(batched) == len(images)
    for one_by_one, from_batch in zip(individually, batched):
        assert len(one_by_one) == len(from_batch)
        for (label_a, conf_a), (label_b, conf_b) in zip(one_by_one, from_batch):
            assert label_a == label_b
            assert conf_a == pytest.approx(conf_b, abs=1e-5)


def test_predict_topk_batch_handles_an_empty_list(tmp_path):
    data_dir = _tiny_dataset(tmp_path)
    model, result = run_training(data_dir, epochs=25, batch_size=8, device=torch.device("cpu"), seed=0)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(model, result.classes, checkpoint_path)
    loaded = TileClassifierModel(checkpoint_path, device="cpu")

    assert loaded.predict_topk_batch([], k=3) == []


def test_fine_tuning_from_checkpoint_requires_matching_classes(tmp_path):
    data_dir = _tiny_dataset(tmp_path / "base", labels=("A", "B", "BLANK"))
    model, result = run_training(data_dir, epochs=5, batch_size=8, device=torch.device("cpu"), seed=1)
    checkpoint_path = tmp_path / "base.pt"
    save_checkpoint(model, result.classes, checkpoint_path)

    mismatched_dir = _tiny_dataset(tmp_path / "mismatched", labels=("A", "C", "BLANK"), seed=2)
    try:
        run_training(mismatched_dir, epochs=1, device=torch.device("cpu"), init_checkpoint=checkpoint_path)
        assert False, "expected a ValueError for mismatched class sets"
    except ValueError as e:
        assert "don't match" in str(e)


def test_fine_tuning_from_checkpoint_continues_training_successfully(tmp_path):
    data_dir = _tiny_dataset(tmp_path / "base", labels=("A", "B", "BLANK"), seed=1)
    model, result = run_training(data_dir, epochs=25, batch_size=8, device=torch.device("cpu"), seed=1)
    checkpoint_path = tmp_path / "base.pt"
    save_checkpoint(model, result.classes, checkpoint_path)

    fine_tuned_model, fine_tuned_result = run_training(
        data_dir, epochs=3, batch_size=8, lr=1e-4, device=torch.device("cpu"), seed=1,
        init_checkpoint=checkpoint_path,
    )
    assert fine_tuned_result.classes == result.classes
    # Fine-tuning from an already-good checkpoint on the same data shouldn't
    # collapse accuracy -- this is a plumbing check (weights actually loaded
    # and training continued), not a claim about real fine-tuning gains.
    assert fine_tuned_result.final_val_accuracy > 0.5
