"""Smoke tests for the classifier training pipeline -- deliberately tiny
(few classes, few samples, few epochs) so this stays fast; this validates
plumbing correctness, not real-world model accuracy (that needs the full
synthetic set plus real fine-tuning data, run separately as an actual
training job, not a unit test).
"""
import random

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
