# Saved model checkpoints

## `tile_classifier_v1.pt`

The per-cell letter classifier described in `docs/classifier-accuracy-plan.md`.

- **Architecture**: `PretrainedTileClassifier` (MobileNetV3-Small, ImageNet-pretrained,
  first conv adapted to 1-channel input) — see `training/classify/model.py`.
- **Classes**: 27 (`A`-`Z` + `BLANK`), in the order stored in the checkpoint's `classes` list.
  Always read this list back from the checkpoint rather than assuming an order.
- **Calibration**: temperature-scaled, `T ≈ 1.402` (stored in the checkpoint; applied
  automatically by `TileClassifierModel`, see below). Fitted on the same held-out set used to
  report accuracy — see the calibration caveat in `training/classify/calibrate.py`'s docstring.
- **Training data**: synthetic renders (`training/synth_render`) pretrained, then fine-tuned on
  ~350 real tile crops pulled from broadcast footage (2 tournament productions, several games).
- **Honest accuracy**: **60.6%** (60/99) on a *game-disjoint* held-out set — 3 entire games/tables
  never touched during training, not just random held-out tiles. Board-tile accuracy is
  meaningfully higher (~68%) than rack-tile accuracy (~43%); see the accuracy-plan doc for why
  (rack crops currently come from looser manual bounding boxes, not the board's calibrated
  per-cell homography).
- **Not yet production-ready**: this is a checkpoint from an active accuracy-improvement pass,
  not a finished model. The confidence-fallback publish gateway (`PublishMode
  .AUTONOMOUS_WITH_CONFIDENCE_FALLBACK`) exists specifically so a model at this accuracy level
  can still be used safely — most predictions won't clear a high confidence threshold and will
  correctly route to operator review.

### Loading it

```python
from training.classify.infer import TileClassifierModel

model = TileClassifierModel("models/tile_classifier_v1.pt", device="cpu")
label, confidence = model.predict(image)  # image: PIL.Image or RGB numpy array
```

`TileClassifierModel` reads `model_type`, `classes`, and `temperature` out of the checkpoint
itself and applies the matching canonicalization transform automatically — no need to know any
of this to use it correctly, it's documented here for anyone inspecting the file directly.
