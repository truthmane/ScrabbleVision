# Rack-tile object detector (Phase 5)

Real rack tiles aren't evenly spaced the way board cells are (players
group letters, leave gaps for tiles they're considering exchanging), so
the board's fixed-grid `crop_cell` approach never applied to racks --
`training/collect/crop_rack.py`'s manual per-group pixel ranges have been
a deliberate stand-in for this since WS3. This directory builds the real
fix: an RF-DETR object detector that localizes individual rack tiles
directly, the way the master architecture plan's Phase 5 always intended.

## Pipeline

1. **`training/synth_render/rack_scene_renderer.py`** -- generates
   unlimited free synthetic rack photos (1-7 tiles, realistic gaps,
   wood-grain background) with exact bounding boxes. Bootstraps the
   detector the same way `tile_renderer.py` bootstrapped the classifier.
2. **`training/detect/reconstruct_rack_boxes.py`** -- turns the real rack
   photos already harvested for WS3 into bounding-box training data too,
   using the manual `(x1, x2, letters)` groups `crop_rack.py` already
   recorded -- no new manual box-drawing needed, at least for the tiles
   already collected. Detects the wood rack ledge by color to derive a
   y-band; pass an explicit `y_band` override on photos where that
   heuristic doesn't apply.
3. **`training/detect/build_rack_dataset.py`** -- assembles a
   Roboflow-standard COCO dataset (`{train,valid,test}/` +
   `_annotations.coco.json` per split) that RF-DETR trains on directly.
   Real photos are folded into `train` only -- real data is scarce enough
   here (8 photos, 45 tiles as of this writing) that none of it should be
   spent on validation; validating real-world performance should come
   from deploying against fresh real frames later, the same pattern WS3
   used for the classifier.
4. **`training/detect/train_rack_detector.py`** -- trains an RF-DETR
   model on the assembled dataset.

## Running it

### Local smoke test (CPU or MPS, before spending any GPU money)

```bash
pip install "rfdetr[train]"

python -c "
from pathlib import Path
from training.detect.build_rack_dataset import build_dataset
build_dataset(Path('/tmp/rack_smoke'), num_train_synthetic=8, num_valid_synthetic=4, num_test_synthetic=4)
"

python -m training.detect.train_rack_detector /tmp/rack_smoke \
    --model nano --epochs 1 --batch-size 1 --device cpu \
    --output-dir /tmp/rack_smoke_output
```

This exists purely to catch integration bugs (malformed boxes, bad
category IDs, missing files) for free before running on a paid GPU --
not to produce a usable model. 1 epoch on 8 tiny images will not detect
anything useful. Confirmed working this way: 8 images, 1 epoch, CPU,
ran clean end-to-end in well under a minute.

**`--device mps` on a full-size dataset works, but has a slow first
epoch** -- tried it (2008 train images, RF-DETR-nano): epoch 1 alone
took ~10 minutes (looked stalled at first -- under 100/502 steps after
6 minutes -- almost certainly MPS kernel compilation/caching warm-up),
but every epoch after that took only ~9 minutes, and the full 10-epoch
run finished in about 95 minutes total. Slower than a real GPU would be,
but usable overnight if you don't want to wait for a RunPod pod to
provision. See "First local result" below for what that run produced.

### Real training run (RunPod or similar)

Recommended GPU: **RTX 4090 (24GB)** on RunPod, or an A5000/L4 if 4090s
aren't available. RF-DETR is a real-time-oriented detection transformer
(Roboflow) explicitly designed to fine-tune on a single consumer GPU --
this dataset is small enough that an A100/H100 would be solving a
problem this project doesn't have. Expect well under an hour for a full
run.

```bash
# On the pod:
pip install "rfdetr[train]"
# Copy the dataset directory built by build_rack_dataset.py onto the pod
# (scp, rsync, or a RunPod network volume), then:
python -m training.detect.train_rack_detector /workspace/rack_detect_dataset \
    --model nano --epochs 50 --batch-size 8 --device cuda \
    --output-dir /workspace/rack_detector_output
```

`--model` accepts `nano`, `small`, `medium`, `base`, `large` (see
`train_rack_detector.MODEL_CLASSES`) -- start with `nano`, the smallest
and fastest to iterate with; step up only if held-out accuracy on the
synthetic valid/test splits demands it.

## First local result (RF-DETR-nano, 10 epochs, MPS, 2026-07-25)

Trained on the exact dataset `build_dataset` produces by default (2008
train / 300 valid / 300 test synthetic scenes + the 45 real boxes).
Synthetic valid mAP@50:95 reached 0.999 by epoch 10 -- expected, since
synthetic scenes are easier than real photos and this checkpoint has
seen real examples of literally every training photo, so that number
alone proves very little about real-world performance.

**Ran `visualize_rack_detections.py` against 3 of the 8 real training
photos as a sanity check** (not a held-out test -- these photos were in
`train`, so this checks "did the pipeline learn anything real at all,"
not generalization): 15 of 16 real tiles across those 3 photos were
detected with a correct label and a tight box at `--threshold 0.3`; one
tile (an "L" in a 7-tile rack) was missed entirely. No false-positive
boxes at that threshold. Promising, but not a real accuracy number.

**Then ran it against one genuinely fresh real photo** (a 7-tile rack,
"XETSREM", from a different move of the same game, never used in
training) -- the actual test. Result was noticeably weaker: E, T, and S
came back correct and tight; X was localized correctly but misclassified
as "K"; the R tile produced two overlapping detections (one plausible,
one spurious extra); the M tile was missed entirely. Roughly half the
tiles clean, real failure modes on the rest (misclassification,
duplicate boxes, a clean miss) -- not memorization, but not yet reliable
either.

**This is the same lesson WS3 already taught the tile classifier**: 45
real boxes from 8 photos of one broadcast/camera setup isn't enough
*diversity* for real generalization, even though it's enough to fit
those exact photos well. The fix is almost certainly the same one that
worked there -- more real photos from *different* games/venues/cameras,
not just more frames of the same one -- rather than more epochs on this
same 45-box set. Do this before trusting the checkpoint for anything.

## What's real vs. synthetic right now

- **Synthetic**: unlimited, generated fresh every run, used for
  train/valid/test.
- **Real**: 8 rack photos harvested from the 2026 NASPA broadcast (see
  `docs/classifier-accuracy-plan.md`'s WS3 section) -- 45 tiles total,
  folded into `train` only. The raw photos themselves aren't committed
  to the repo (copyrighted broadcast footage, same policy as the rest of
  this project's real-tile data); only the reconstruction code is.

## Next steps once a trained model exists

- Wire the detector's output boxes into `autoscorer/perception/board_reader.py`
  as a rack-reading counterpart to `read_new_cells`/`read_new_cells_voted` --
  crop each detected box, run it through the existing `TileClassifierModel`
  (no retraining needed there; the detector only needs to *localize* tiles,
  the classifier already knows how to *read* them).
- Feed real rack contents into `autoscorer/gamelogic/movedetect/constraint_decoder.py`'s
  `decode_feasible_reading` (currently called with `racks=[]` everywhere
  since no rack camera/detector existed yet) -- this tightens the
  pool-feasibility budget and should make constraint decoding meaningfully
  more accurate.
- Validate against fresh real rack photos (not the 8 already spent on
  training) the same way WS3 validated the classifier -- harvest new
  frames, run the detector, compare against the known rack contents from
  a `.gcg` replay.
