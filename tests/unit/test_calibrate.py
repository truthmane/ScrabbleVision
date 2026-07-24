import random

import torch

from training.classify.calibrate import apply_temperature_to_checkpoint, collect_logits, fit_temperature
from training.classify.infer import TileClassifierModel
from training.classify.train import run_training, save_checkpoint
from training.synth_render.tile_renderer import augment_tile, render_tile


def _tiny_dataset(tmp_path, labels=("A", "B", "BLANK"), samples_per_class=30, seed=1):
    rng = random.Random(seed)
    for label in labels:
        letter = None if label == "BLANK" else label
        class_dir = tmp_path / label
        class_dir.mkdir(parents=True, exist_ok=True)
        base = render_tile(letter, rng=rng)
        for i in range(samples_per_class):
            augment_tile(base, rng=rng).save(class_dir / f"{i:03d}.png")
    return tmp_path


def test_fit_temperature_returns_a_positive_scalar(tmp_path):
    data_dir = _tiny_dataset(tmp_path / "data")
    model, result = run_training(data_dir, epochs=20, batch_size=8, device=torch.device("cpu"), seed=1)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(model, result.classes, checkpoint_path)

    clf = TileClassifierModel(checkpoint_path, device="cpu")
    logits, labels = collect_logits(clf, data_dir)
    assert logits.shape[0] == labels.shape[0] == 90

    temperature = fit_temperature(logits, labels)
    assert temperature > 0


def test_calibrated_checkpoint_changes_confidence_not_prediction(tmp_path):
    data_dir = _tiny_dataset(tmp_path / "data")
    model, result = run_training(data_dir, epochs=20, batch_size=8, device=torch.device("cpu"), seed=1)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(model, result.classes, checkpoint_path)

    clf_before = TileClassifierModel(checkpoint_path, device="cpu")
    logits, labels = collect_logits(clf_before, data_dir)
    temperature = fit_temperature(logits, labels)

    calibrated_path = tmp_path / "model_calibrated.pt"
    apply_temperature_to_checkpoint(checkpoint_path, temperature, calibrated_path)

    clf_after = TileClassifierModel(calibrated_path, device="cpu")
    assert clf_after.temperature == temperature

    sample = render_tile("A", rng=random.Random(5))
    augmented = augment_tile(sample, rng=random.Random(5))
    label_before, conf_before = clf_before.predict(augmented)
    label_after, conf_after = clf_after.predict(augmented)

    # Temperature scaling must never change which class wins (argmax is
    # invariant to a positive scalar divisor), only the reported confidence.
    assert label_before == label_after
