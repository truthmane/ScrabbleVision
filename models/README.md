# Saved model checkpoints

## `tile_classifier_v1.pt`

The per-cell letter classifier described in `docs/classifier-accuracy-plan.md`.

- **Architecture**: `PretrainedTileClassifier` (MobileNetV3-Small, ImageNet-pretrained,
  first conv adapted to 1-channel input) — see `training/classify/model.py`.
- **Classes**: 27 (`A`-`Z` + `BLANK`), in the order stored in the checkpoint's `classes` list.
  Always read this list back from the checkpoint rather than assuming an order.
- **Calibration**: temperature-scaled, `T ≈ 1.171` (stored in the checkpoint; applied
  automatically by `TileClassifierModel`, see below). Fitted on the same held-out set used to
  report accuracy — see the calibration caveat in `training/classify/calibrate.py`'s docstring.
- **Training data**: synthetic renders (`training/synth_render`) pretrained, then fine-tuned on
  real tile crops now committed at `training/data/real_tiles/` (513 tiles as of this checkpoint,
  see that directory's own README for exact per-venue provenance) — **the previous ~620-tile
  dataset this checkpoint's predecessor was built from no longer exists**; it lived only in an
  ephemeral per-session scratchpad and was lost between sessions. `training/data/real_tiles/` is
  committed specifically so this can't happen again — the actual data a checkpoint depends on
  should always be inspectable/extendable, not just the checkpoint itself.
- **Honest accuracy**: **85.1%** (103/121) on a *venue-disjoint* held-out set — all Causeway
  Challenge 2026 tiles (both tables) held out entirely, fine-tuned only on the 4 Mack Meller
  2026 Scrabble Players Championship games (392 tiles), a real venue/camera the held-out set
  never touched. Baseline (the pre-fine-tune checkpoint, before this session's data) scored
  **81.0%** (98/121) on the exact same held-out set — so the honest lift from this round of real
  data is +4.1 points, not the full 85.1%.
- **Catastrophic forgetting is real here and worth remembering for the next retrain**: an
  identical fine-tune at the "normal" settings used historically (8 epochs, `lr=1e-3`) actively
  *regressed* held-out accuracy to 72.7% — it overfit hard to the one new venue's specific tile
  style (all-black tiles, one font) at the expense of everything the checkpoint previously knew,
  since there's no way to mix in the lost old dataset alongside the new one anymore. A much
  gentler continuation (2 epochs, `lr=1e-4`) was what actually produced the +4.1 point real win
  above; 1 epoch at the same `lr` scored 84.3%, slightly behind 2. Try gentle settings first on
  any future continuation fine-tune from this checkpoint, and always A/B against a real held-out
  venue before trusting a "final validation accuracy" number from `train.py` itself (that number
  is an in-distribution split of the *new* training data alone and was 100% in every run tried
  here, completely uninformative about generalization).
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

## `rack_detector_v1.pth`

The rack-tile object detector described in `training/detect/README.md`. Stored via **Git LFS**
(see `.gitattributes`) since it's ~115MB, over GitHub's 100MB per-file limit for regular commits.

- **Architecture**: RF-DETR-nano (`rfdetr.RFDETRNano`) — a real-time-oriented detection
  transformer (Roboflow), fine-tuned on a single RunPod RTX 4090.
- **Classes**: 27 (`A`-`Z` + `BLANK`), same vocabulary as the tile classifier above — but this
  model only *localizes* tiles; letter classification is deliberately left to
  `tile_classifier_v1.pt` (see `read_rack` in `autoscorer/perception/board_reader.py`), not this
  checkpoint's own class predictions.
- **Training data**: unlimited synthetic rack scenes (`training/synth_render/rack_scene_renderer.py`)
  plus 136 real tile boxes from **3 distinct venues/productions** (2026 NASPA broadcast, "Let's
  Play Scrabble"/CSW production, WESPA Word Wars), folded into `train` only.
- **Honest accuracy**: tested on two real held-out rack photos never used in training —
  **all 14 tiles localized correctly (zero misses, zero duplicate boxes)**, and **14/14
  correctly read** through the production path (`board_reader.read_rack`, which crops each
  detected box and re-reads it with `tile_classifier_v1.pt`). The number worth quoting is
  **7/7 on the WESPA rack**, a venue the classifier had never seen at all — it was last
  trained at `4544db3`, before that venue existed in this project. The other 7/7 (NASPA) is
  clean for the detector but optimistic for the classifier, which saw rack crops from that
  same game in fine-tuning. Note: RF-DETR's *own* classification head only gets 12/14 on
  these photos — the two-stage split (detector localizes, classifier reads) is measurably
  better than letting the detector label. See `training/detect/README.md` for the full
  comparison.
- **Not yet wired into a full end-to-end run**: `read_rack` exists in the perception layer, but
  the constraint decoder hasn't yet been re-validated with real (non-empty) racks in place of the
  `racks=[]` placeholder it always used before this checkpoint existed.

### Loading it

```python
from training.detect.visualize_rack_detections import load_model

detector = load_model("models/rack_detector_v1.pth")
```

Pass `detector` straight into `autoscorer.perception.board_reader.read_rack(rack_frame, detector, classifier)`
along with a loaded `TileClassifierModel` — `read_rack` handles the BGR→RGB conversion and
per-box classification internally.
