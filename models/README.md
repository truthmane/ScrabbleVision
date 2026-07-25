# Saved model checkpoints

## `tile_classifier_v1.pt`

The per-cell letter classifier described in `docs/classifier-accuracy-plan.md`.

- **Architecture**: `PretrainedTileClassifier` (MobileNetV3-Small, ImageNet-pretrained,
  first conv adapted to 1-channel input) — see `training/classify/model.py`.
- **Classes**: 27 (`A`-`Z` + `BLANK`), in the order stored in the checkpoint's `classes` list.
  Always read this list back from the checkpoint rather than assuming an order.
- **Calibration**: temperature-scaled, `T ≈ 1.352` (stored in the checkpoint; applied
  automatically by `TileClassifierModel`, see below). Fitted on the same held-out set used to
  report accuracy — see the calibration caveat in `training/classify/calibrate.py`'s docstring.
- **Training data**: synthetic renders (`training/synth_render`) pretrained, then fine-tuned on
  ~620 real tile crops pulled from broadcast footage across **6 tournament productions/venues**
  (2 earlier broadcast productions + 4 real 2026 NASPA Scrabble Players Championship games,
  auto-labeled via WS3's GCG-replay pipeline — see `docs/classifier-accuracy-plan.md`).
- **Honest accuracy**: **64.6%** (64/99) on a *game-disjoint* held-out set — 3 entire games/tables
  never touched during training, not just random held-out tiles (up from 60.6% before the WS3
  data — see the accuracy-plan doc's WS3 section for the honest story of what did and didn't move
  this number: venue diversity mattered, raw frame volume from one venue did not). Board-tile
  accuracy is meaningfully higher (~72%) than rack-tile accuracy (~46%); see the accuracy-plan doc
  for why (rack crops currently come from looser manual bounding boxes, not the board's calibrated
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
