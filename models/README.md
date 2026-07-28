# Saved model checkpoints

## `tile_classifier_v1.pt`

The per-cell letter classifier described in `docs/classifier-accuracy-plan.md`.

- **Architecture**: `PretrainedTileClassifier` (MobileNetV3-Small, ImageNet-pretrained,
  first conv adapted to 1-channel input) — see `training/classify/model.py`.
- **Classes**: 27 (`A`-`Z` + `BLANK`), in the order stored in the checkpoint's `classes` list.
  Always read this list back from the checkpoint rather than assuming an order.
- **Calibration**: temperature-scaled, `T ≈ 1.176` (stored in the checkpoint; applied
  automatically by `TileClassifierModel`, see below). Fitted on the same held-out sets used to
  report accuracy — see the calibration caveat in `training/classify/calibrate.py`'s docstring.
- **Training data**: synthetic renders (`training/synth_render`) pretrained, then fine-tuned
  incrementally on real tile crops now committed at `training/data/real_tiles/` (4,633 tiles as
  of this checkpoint, see that directory's own README for exact per-venue provenance) — **the
  previous ~620-tile dataset this checkpoint's predecessor was built from no longer exists**; it
  lived only in an ephemeral per-session scratchpad and was lost between sessions.
  `training/data/real_tiles/` is committed specifically so this can't happen again — the actual
  data a checkpoint depends on should always be inspectable/extendable, not just the checkpoint
  itself.
- **Honest accuracy**: **97.5%** (118/121) board tiles on the same *venue-disjoint* Causeway
  held-out set used throughout this project's history (was 93.4%, 113/121 a few rounds back) — the
  model is a chain of incremental continuation fine-tunes, each measured on this same held-out set:
  the 4 Mack Meller 2026 SPC games (392 tiles, 81.0% → 85.1%), 7 more cross-tables.com games (680
  tiles, 85.1% → 91.7%), 10 more spanning 8 distinct tournaments 2019-2026 (981 tiles, 91.7% →
  93.4%), 1,189 real WESPA broadcast rack tiles (93.4% → 96.7%), then 1,270 more real rack tiles
  from two more venues (96.7% → 97.5%, see below) — none of these training venues ever touch the
  held-out set.
- **Rack accuracy across five real venues — the honest, multi-domain picture, not one number.**
  A round-3 correction is worth stating plainly first: an earlier version of this README described
  the WESPA rack tiles as "a clean rendered graphic, not a real photograph" — **that was wrong**.
  Direct pixel-level zoom on multiple broadcasts (WESPA, this project's own "4th of July" harvest,
  and two more NASPA-produced streams) showed all of them are genuine photographs of physical
  racks; the visual mistake came from judging small downscaled screenshots instead of zooming in.
  Corrected going forward: WESPA and "4th of July / Bob Linn Superstars" both use white
  tiles/red-maroon lettering; NASPA's "NWL Championship Division Day 3" stream uses **black
  tiles/white lettering** — the same physical tile-set style as the original, hardest held-out set
  below. Five held-out measurements, each on a different real venue never trained on:
  | Held-out set | Domain | Before this round | After |
  |---|---|---|---|
  | Causeway (board) | white tiles, venue-disjoint | 96.7% (117/121) | **97.5% (118/121)** |
  | WESPA rack, time-disjoint split | white/red, real photo | 94.7% (268/283) | **95.1% (269/283)** |
  | "4th of July" rack, time-disjoint split | white/red, real photo, different tournament | 89.8% zero-shot | **94.7% (177/187)** |
  | NASPA "Day 3" rack, time-disjoint split | **black/white**, real photo, different tournament | 69.8% zero-shot (74/106) | **85.9% (91/106)** |
  | Original NASPA Finals rack (`held_out_rack_naspa`, 116 tiles) | black/white, real photo, a *different specific game* than Day 3 | 64.7% (75/116) | **61.2% (71/116)** |
  Four of five improved, three of them clearly (Day 3's own held-out jumped 16 points — real
  evidence the model learned genuine black-tile-domain generalization, not just Day-3-specific
  overfitting). The one regression (NASPA Finals, -3.5 points) is on the smallest set (116 tiles,
  where a swing of 3-4 tiles is close to one standard error on the underlying binomial estimate) —
  promoted anyway after explicit user sign-off, since the weight of evidence (large, clearly-real
  wins elsewhere) outweighs a single-digit-tile difference on the noisiest measurement. **Still not
  a solved problem**: even the best number here (85.9% on Day 3) trails board accuracy by over 10
  points, and the Finals-specific number hasn't moved past ~60-65% across three different training
  rounds — there may be something about that one specific game/camera-moment that's unusually hard
  independent of general venue diversity. See `training/data/real_tiles/README.md`'s "Rack tile
  crops" section for how each venue was harvested.
- **Catastrophic forgetting is real here, confirmed on two separate rounds now.** An identical
  fine-tune at the "normal" settings used historically (8 epochs, `lr=1e-3`) actively *regressed*
  held-out accuracy the first time this was tried (round 1: 72.7%, vs. 85.1% with gentle
  settings). Round 2 went further and tested training **from scratch on the full real dataset**
  (ImageNet-init, no continuation checkpoint, all 1,072 non-held-out real tiles at once) instead
  of incremental continuation — expecting this might do better since it isn't limited by whatever
  the previous checkpoint already learned. It did much *worse*: 53.7% at 8 epochs/`lr=1e-3`, 66.1%
  at 15 epochs/`lr=3e-4` — both far below the 91.7% the incremental continuation reached, most
  likely because training from scratch this way skips the synthetic-render pretraining stage this
  checkpoint's lineage benefited from, and 1,072 real crops split ~27 ways just isn't enough
  signal to learn 27-way tile classification unaided. **The lesson holds across two independent
  rounds: a gentle (2 epoch, `lr=1e-4`) continuation from the current checkpoint beats both
  aggressive continuation settings AND training from scratch on the full dataset.** Always A/B
  against a real held-out venue before trusting a "final validation accuracy" number from
  `train.py` itself (that number is an in-distribution split of the *new* training data alone and
  is uninformative about generalization on its own).
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
- **Now wired into a full end-to-end run** (`tests/slow/test_rack_detector_end_to_end.py`):
  drives one real `GameWatcher` with both the real board pipeline and this real rack detector +
  the real tile classifier together, against real turns from the WESPA Game 1 broadcast (rack
  ground truth pulled straight from the `.gcg`'s own `rack` field). `GameWatcher.racks` is no
  longer the `racks=[]` placeholder `decode_with_lexicon`'s pool-feasibility check always fell
  back to before this. Honest result on synthetic rack renders (not the real photos this
  checkpoint was originally validated on): **90% per-tile accuracy (19/21) on accepted reads**,
  with one of four rack reads correctly rejected by a new safety check rather than silently
  corrupting state (see below) — lower than the original 14/14 on real photos, as expected for a
  different, harder rendering distribution, not a regression.
- **Found and fixed two real bugs while building that first end-to-end run**, both invisible
  until board and rack detection were actually driven together for the first time:
  1. `training/synth_render/rack_scene_renderer.py`'s `generate_rack_scene` silently DROPPED
     tiles that didn't fit within its fixed canvas width — random gap rolls for a full 7-tile
     rack could occasionally exceed the canvas, and the function quietly rendered (and recorded
     ground truth for) fewer tiles than requested instead of guaranteeing every tile appears.
     Fixed by precomputing gaps and shrinking them (never dropping a tile) when they don't fit.
  2. A rack misread (most often a real letter hallucinated as a `BLANK`) that claims more of some
     letter exists across board+racks than the real 100-tile set has doesn't just make that one
     rack wrong — silently committed, it corrupts `remaining_supply`'s pool math for every future
     decode using ANY player's rack, including completely unrelated board turns. `record_rack` now
     checks `compute_pool_state` before committing a candidate rack and rejects (routes to
     operator, keeps the last known-good rack) anything that would create an impossible tile
     supply, rather than silently corrupting tracked state.

### Loading it

```python
from training.detect.visualize_rack_detections import load_model

detector = load_model("models/rack_detector_v1.pth")
```

Pass `detector` straight into `autoscorer.perception.board_reader.read_rack(rack_frame, detector, classifier)`
along with a loaded `TileClassifierModel` — `read_rack` handles the BGR→RGB conversion and
per-box classification internally.
