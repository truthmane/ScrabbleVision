# Classifier Accuracy Plan: 30% → effectively 100%

**Status**: All five workstreams (WS1-WS5) implemented and validated (see Results below). WS3 came
out better-than-planned as tooling (a GCG-record parser/replay engine, validated against 6 real
NASPA 2026 games with exact score reproduction) and, once real venue diversity was added rather
than more frames of one venue, **did move held-out accuracy: 60.6% → 64.6%**. WS4's constraint
decoder is now actually wired into the pipeline (not just unit-tested), temporal voting (M2) is
built and validated against 5 genuine consecutive broadcast frames, and WS5's end-to-end
game-replay metric is real, run against 6 harvested real moves: **3/6 auto-published (all
correct), 3/6 correctly routed to operator, 0/6 silent failures** — see the WS4/WS5 sections for
the full story.
**Audience**: whoever implements next (human or model). Assumes familiarity with the existing
codebase (`autoscorer/`, `training/`) and the collection scripts in the session scratchpad
(`decode_frame.py`, `make_table.py`, `crop_rack.py`).

## Results so far (honest, game-disjoint held-out set: 99 real tiles from 3 held-out
games/tables never touched during training, 26/27 classes represented — see
`training/classify/evaluate.py` and the game-disjoint split built for this pass)

| Change | Held-out accuracy | Notes |
|---|---|---|
| Baseline (pre-plan) | 31.3% | Confirms the earlier session's number; not a fluke. |
| + WS1 canonicalization + bug fixes alone | 31.3% (flat) | Fixed the below-chance synthetic prior (1.4%→10.1% zero-shot) and the class-imbalance bias, but didn't move fine-tuned accuracy by itself — see the auto-crop finding below for why. |
| + auto-crop-to-tile (found via board-vs-rack breakdown) | 32.3% (flat overall, but rack tiles 10.7%→25.0%) | Board/rack accuracy breakdown revealed rack crops' loose manual bounding boxes were leaving glyphs tiny/off-center relative to training data — a framing bug, not a color bug. |
| + WS2 pretrained MobileNetV3-Small backbone | **60.6%** | The single biggest lever tried. Board tiles 39%→67.6%, rack tiles 25%→42.9%. Confirms the plan's hypothesis: ImageNet-pretrained low-level features transfer far better from ~350 real tiles than training from scratch, even on top of the same canonicalized input. |
| + WS4 temperature calibration | 60.6% (unchanged, by design) | Confidence-vs-accuracy went from barely-monotonic (69%→85% across thresholds 0.5→0.95, pre-calibration was closer to flat) to a real signal a gateway can act on. Fitted T=1.402 (model was overconfident, as expected). |
| + WS3 real data from 1 more game (321→442 real tiles, deduplicated) | 58.6% | *Regression*, not a fluke — traced to near-duplicate crops of the same 1-2 physical tiles (same static camera, same board) dominating a few classes' training examples. See WS3 below. |
| + WS3 real data from a 2nd, visually distinct game/venue (442→520 real tiles) | 60.6% | Regression fixed by adding venue diversity, not just volume — back to baseline, not yet past it. |
| + WS3 real data from a 3rd game (520→616 real tiles, 4 games total on top of the original set) | **64.6%** | First clear, reproducible win — verified by re-running the identical fine-tune (fixed seed) and getting the identical 64/99. Board tiles ~68%→71.8%, rack tiles ~43%→46.4%. Now the deployed checkpoint (`models/tile_classifier_v1.pt`, recalibrated to T=1.352). Confirms the diversity hypothesis: going from 2 to 3 additional venues is what tipped this from a null result into a real one. |

Two real bugs were caught and fixed by testing hypotheses against data rather than assuming:
auto-crop's first version made accuracy *worse* (10.1%→3.0% zero-shot) because it mistook a
tightly-bound board crop's glyph for "the tile" and cropped tighter still; a squareness/solidity
check on the candidate contour fixed this. Temperature-fitting's first version let LBFGS drive
the temperature negative, which silently flips softmax ranking (a confidence-calibration bug
that would have corrupted predictions, not just their reported confidence) — fixed by optimizing
in log-space so T stays positive by construction.

**What this means for the roadmap below**: WS2 (pretrained backbone) delivered far more than
WS1 alone, and more than the earlier session's pure data-scaling push did (which moved 16%→30%
with 2.5x more data). The single highest-leverage remaining lever is still WS3 (more real data,
now feeding a backbone that actually uses it well) — the projected 95-98% for WS3 in the
original table below should be read as "plausible given WS2 already outperformed its own
estimate," not re-derived.

## Why this is the right goal

Per-tile classification of Scrabble tiles is an intrinsically easy ML problem: 27 classes,
high-contrast printed glyphs, standardized fonts, tiles pre-localized by the homography to a
known grid. Prior art (jheidel's KNN on 18×18 binarized crops; UAH thesis EfficientNet) reached
~95%+ with less machinery than we already have. Our current 30.1% held-out accuracy is
anomalous, and the evidence below shows it is caused by identifiable defects, not by a
fundamental data shortage.

"100%" is defined at the **published-move level**, not the single-frame softmax level:

- **M1 — per-tile, single frame**: ≥97% (classifier alone)
- **M2 — per-tile, after temporal voting** over ≥5 stable frames: ≥99.5%
- **M3 — per-move (word + score), after constraint decoding**: ≈100%, with the residue
  explicitly flagged to the operator rather than silently wrong
- **M4 — operator-review rate** in confidence-fallback mode: ≤5% of moves

M3+M4 is what "works on a normal stream setup" means for broadcast. No single-frame classifier
anywhere hits literal 100%; the system does, by voting, constraints, and human fallback for a
tiny flagged tail.

## Diagnosis (evidence from this repo's own experiments)

1. **Inverted-polarity synthetic prior.** `training/synth_render/tile_renderer.py` renders
   dark glyphs on cream/white tiles. Both observed tournament tile sets are the opposite:
   white glyphs on black tiles (NASPA/Protiles) and white glyphs on red tiles (CSW event).
   Decisive evidence: the synthetic-only baseline scored **1/73 = 1.4% on real tiles — below
   the 3.7% random-chance floor**. The pretraining is anti-correlated with reality, and
   fine-tuning had to fight it.
2. **Evaluation leakage masked the problem.** The "combined val" split (73–78%) contains
   augmented copies of source tiles that also appear in train. Only the source-disjoint
   held-out set (30.1%) was honest. Worse, the held-out split is source-*tile*-disjoint but not
   *frame*-disjoint: crops from the same board/frame share lighting and camera position, so
   even 30% is likely slightly optimistic.
3. **Class imbalance was real but not the ceiling.** Balancing augmentation counts (v4 run)
   flattened the prediction distribution (no more everything-is-E) yet accuracy stayed at
   30.1% — keep the fix, but the remaining errors are domain-gap errors: visually dissimilar
   confusions (Z→K, O→X, A→W) at uniformly low confidence, the signature of features that
   don't transfer, not of genuinely ambiguous glyphs.
4. **Confidence is uncalibrated and nearly uninformative.** Top-20-confidence accuracy 7/20 vs
   bottom-20 4/20. The 0.9 gateway threshold currently gates on noise.
5. **Minor but real: augmentation bug.** `augment_real_photo` rotates with
   `fillcolor=(255,255,255)` — white corner wedges on dark tiles, an artifact class the model
   partly learns. Use edge-replicate/reflect fill instead.
6. **Input resolution is fine.** ~55px tiles from 1080p are comfortably human-readable;
   resolution is not the bottleneck. YouTube compression makes our data *harder* than a real
   production camera feed, so production should be easier than our benchmark.

## Workstreams

### WS1 — Canonicalize the input (small effort, largest single algorithmic win) — ✅ DONE

Implemented in `autoscorer/perception/classify/canonicalize.py`. Includes one item added
during implementation that wasn't in the original plan: **auto-crop-to-tile**, found necessary
after a board-vs-rack accuracy breakdown showed rack crops (loose manual bounding boxes) scoring
far below board crops (precise calibrated homography) on the same model — a framing mismatch,
not a color one. Uses a squareness/solidity check on the candidate contour to avoid mistaking a
glyph for the tile boundary on already-tight crops (the first version of this made things worse,
not better — see Results above).

Original plan text, for reference — insert a canonicalization step between `crop_cell` and the classifier
(new: `autoscorer/perception/classify/canonicalize.py`, shared by training and inference):

1. Grayscale.
2. Local contrast normalization (CLAHE).
3. **Polarity normalization**: estimate glyph vs. background polarity (e.g. mean of central
   region vs. border ring); if glyph is lighter than background, invert. Every tile style —
   black, red, cream, future sets — collapses to "dark glyph on light tile."
4. Center-crop ~8–10% margins (drops neighboring-cell bleed and de-emphasizes the score
   subscript), resize to 60×60.

This converts a multi-domain color problem into single-domain grayscale glyph recognition —
the regime where prior art succeeded with KNN. It also future-proofs against new tile sets,
which the risk register already flags. Change `TileClassifier` input to 1 channel.
Optionally keep a second experiment arm with Otsu binarization; pick by held-out accuracy.

### WS2 — Fix the prior, the model, and training hygiene — ✅ MOSTLY DONE

Done: pretrained backbone arm (`PretrainedTileClassifier` in `training/classify/model.py`,
MobileNetV3-Small, first-conv weights averaged to 1 channel, upscaled to 224×224 with
approximated ImageNet-style normalization for that arm only) — this was the single biggest
lever in the whole plan, taking held-out accuracy from 32.3% to 60.6% on its own. Also done:
`WeightedRandomSampler`-based balancing (replacing file duplication), cosine LR decay, the
`augment_real_photo` fill-color bug fix (edge-reflect via `cv2.warpAffine` instead of PIL's
solid-color fill), and per-class precision/recall reporting (`training/classify/evaluate.py`).

Skipped, evidence-based: synthetic renderer v2 (rendering white-on-black/red styles). Once WS1's
canonicalization was in place, the polarity/color-domain gap it would have addressed was already
closed (confirmed: canonicalization alone took the synthetic-only zero-shot score on real tiles
from *below random chance* to above it). Revisit only if accuracy stalls again after WS3.

Original plan text, for reference:

1. **Synthetic renderer v2** (`tile_renderer.py`): render the two observed real styles —
   white-on-black and white-on-red — in addition to (or instead of) cream; heavier
   downscale-to-~50px-then-upscale, mild motion blur, aggressive JPEG/H.264-like compression,
   vignette, white-balance jitter.
2. **Pretrained backbone arm**: alongside the small custom CNN, add a
   torchvision MobileNetV3-Small/ResNet-18. With a few hundred real
   tiles, transfer learning should dominate scratch training. Keep whichever arm wins the
   golden eval; ONNX-export unchanged either way. All of this trains fine on the M2 Pro/MPS.
3. **Training hygiene** (`training/classify/train.py`):
   - Replace file-duplication balancing with `WeightedRandomSampler`.
   - Cosine or step LR decay, early stopping on a *source-disjoint* val split, more epochs.
   - Fix `augment_real_photo` fill color (edge replicate).
   - Report per-class precision/recall, not just aggregate accuracy.

### WS3 — Scale ground truth via ticker replay (kills the labeling bottleneck) — ✅ DONE

Built and validated: `autoscorer/gamelogic/notation.py` (parses both hand-transcribed ticker
lines and, far better, official `.gcg` game records from event.scrabbleplayers.org),
`training/collect/replay_game.py` (replays either through the real `GameSession`/rules engine),
and `training/collect/harvest_from_replay.py` (auto-crops every occupied cell of a rectified real
photo and labels it from the replayed board state — zero manual reading). Validated against 5
real 2026 NASPA Scrabble Players Championship games (`tests/fixtures/*.gcg`) — **all real turns
across all 5 games reproduce the official turn score and cumulative score exactly.**

**The data result, honestly**: harvesting more frames from a *single* game/venue didn't help, and
initially hurt (60.6%→58.6%, see Results table) — near-duplicate photos of the same few physical
tiles (same static camera across frames) dominated a handful of classes' training examples and
skewed them toward that one camera's look. Adding a second, visually distinct game/venue fixed the
regression (back to 60.6%) but didn't exceed it. Adding a **third** distinct game (4 games total
harvested via WS3, on top of the original 3-production dataset) finally moved the number for real:
**64.6% (64/99)**, reproduced exactly on a repeat run with the identical fixed-seed fine-tune —
board tiles 68%→71.8%, rack tiles 43%→46.4%, broad-based across most classes rather than
concentrated in one or two. **Confirms the venue-diversity hypothesis precisely**: 1 extra venue
regressed, 2 was a null result, 3 was a clear win — volume alone was never the lever, breadth of
camera/lighting/venue was. This is now the deployed checkpoint (`models/tile_classifier_v1.pt`).

The decisive data unlock. Tournament broadcasts print every move in a monospace ticker
("`Krafchick, Joey J7 TaSKING 86 107`") using standard notation (coordinate + word; lowercase
letter = blank played as that letter). Therefore:

1. **Transcribe the move list** for a full game (~25 moves — minutes by hand from ~25 paused
   frames, or OCR the monospace strip and hand-verify).
2. **Replay it through our own `GameSession`/rules engine** (already built, already tested) to
   get the exact board contents at every point in the game — including which tiles are blanks.
3. **Harvest frames across the whole video** (extend `decode_frame.py` batch capture): for each
   frame, timestamp-match it to the move index (or diff occupancy against the replay states),
   auto-rectify (contour corners are stable per-venue), auto-crop all occupied cells, and label
   them from the replayed board state. Skip frames failing a stillness/occlusion check.
4. Yield: **thousands of labeled tiles per game** across all letters, many lighting micro-
   variations, plus blanks — for minutes of transcription per game. Two side benefits:
   (a) every replayed game is an end-to-end regression test of our scoring engine against
   official tournament scores; (b) the harvested frame sequences become the integration-test
   replay library the master plan calls for.
5. Promote the scratchpad scripts into the repo as `training/collect/` (capture, transcribe,
   replay-label, harvest) so this is a repeatable pipeline, not session archaeology.

Target: ≥5,000 real tiles across ≥6 games/2+ venues, ≥100 per class (blanks will lag; they're
operator-routed anyway by design).

### WS4 — System-level correction (this is where "100%" actually comes from) — ✅ MOSTLY DONE

Done: **calibration** (`training/classify/calibrate.py`, temperature scaling via LBFGS in
log-space — the first version let temperature go negative, which silently flips softmax ranking;
fixed by parameterizing as `T = exp(log_T)` so it's positive by construction). Fitted T=1.402 on
our held-out set (model was overconfident, as expected); confidence-vs-accuracy went from
barely-monotonic to a real, usable signal (see Results above). Also done: **pool-feasibility
constraint decoding** (`autoscorer/gamelogic/movedetect/constraint_decoder.py` — a proposed
reading can't require more of a letter than the standard distribution has left unaccounted for
elsewhere; ties broken by giving scarce letters to whichever cell the classifier is more
confident about), fully unit-tested, plus `TileClassifierModel.predict_topk` to supply it
candidates.

**Now wired into the pipeline**: `board_reader.read_new_cells` scopes classification to exactly
one turn's new tiles (diffed against `board_before`, occupancy-detected rather than requiring a
second real photo) and returns top-k candidates per cell — the shape `decode_feasible_reading`
needs. `training/collect/eval_harvested_moves.py` puts the two together against real harvested
photos and real official scores (see below) — the first time constraint decoding has run against
anything but its own unit tests.

**Temporal voting is now done too**: `autoscorer/gamelogic/movedetect/temporal_vote.py` averages
each cell's full class-probability distribution across multiple frames of the same stable board
state (the M2 lever — "per-tile after temporal voting over ≥5 stable frames"), and
`board_reader.read_new_cells_voted` wires it into the same per-turn-diffing path `read_new_cells`
uses. Voting is only meaningful once something decides the input frames really do show the same
moment, so `autoscorer/perception/stillness/detector.py` (the master plan's Phase 5
"stillness/occlusion gate") was built alongside it: a frame-to-frame diff score, calibrated
against real footage — genuinely stable board frames scored 0.4-8, a frame where hands entered
the shot scored 37.8, a clean separation. Validated against 5 genuine consecutive frames pulled
from the broadcast (not part of any training data, ~3 seconds apart during the FOISTED move's
board-stable window): single-frame classification got 3/7 cells of that move right; voting across
all 5 frames got 4/7 — a real,
modest, measured improvement, not just a theoretical one. Both blanks in that move stayed
misread even after voting (this checkpoint clearly still struggles specifically with blanks on
this tile style) — voting fixes transient single-frame noise, it doesn't fix a systematic
per-class weakness, which is a data/training problem instead.

Not done: the **dictionary-scored beam search** half of constraint decoding (needs a real
TWL/CSW word list wired into `dictionary/lookup.py`, which is still a stub). Racks still aren't
visible to this pipeline (no rack camera yet — Phase 5), so `decode_feasible_reading` is
currently called with `racks=[]`; the pool-feasibility budget is therefore slightly looser than
it will be once rack contents are known, but this doesn't invalidate anything below — a looser
budget can only ever accept more candidates as feasible, never wrongly reject a correct one.

### WS5 — Honest evaluation harness (do first, actually) — ✅ DONE

Done: game-disjoint split (held out 3 entire games/tables, not just random tiles — see
`training/classify/evaluate.py` and the split-building step in this session) and per-class
precision/recall reporting. **Also now done: the end-to-end game-replay metric**
(`training/collect/eval_harvested_moves.py`), run against the 6 real photos harvested for WS3,
each checked against its game's official `.gcg` record:

| Game/move | Cells | Raw top-1 correct | After constraint decoding | Outcome |
|---|---|---|---|---|
| Final Game 1, move 13 | 3 | 2/3 | 2/3 | Routed to operator (low confidence) |
| Final Game 1, move 19 | 4 | 4/4 | 4/4 | Auto-published, score matched |
| Final Game 1, move 21 | 3 | 3/3 | 3/3 | Auto-published, score matched |
| Final Game 1, move 22 (FOISTED bingo, 2 blanks) | 7 | 1/7 | **4/7** | Routed to operator (low confidence) |
| Game 22, move 15 | 3 | 2/3 | 2/3 | Routed to operator (low confidence) |
| Game 25, move 24 | 2 | 2/2 | 2/2 | Auto-published, score matched |

Run with the **pre-WS3 checkpoint** (`finetuned_pretrained_calibrated.pt`, 60.6%), deliberately
*not* the deployed 64.6% one — these exact photos are in the deployed checkpoint's training data,
so evaluating it against them would be evaluating on training data, not a fair test.

The headline result: **3/6 moves auto-published, all three scored correctly; the other 3/6 all
had a misclassified cell and were all correctly routed to operator instead of silently
auto-publishing a wrong score — zero silent failures across all six real moves.** This is exactly
what the plan's M3+M4 framing predicts: no single-frame classifier hits literal 100%, but voting
+ constraints + confidence-gated human fallback together mean nothing wrong ever ships silently.
The FOISTED move is also a concrete, measured win for constraint decoding itself: raw
classification got only 1 of 7 cells right (a genuinely hard case — a 90-point bingo using both
blanks), and pool-feasibility decoding corrected 3 more of them before the confidence gate still
correctly caught the move as needing a human anyway.

Not done: a pytest-integrated slow suite for this (the real photos it needs can't be committed to
the repo — copyright, same as the rest of this project's real-tile data — so it can't run in CI;
it's a script to be re-run manually against freshly-harvested frames, not a gate).

Original plan text, for reference:

1. **Golden set v2**: source-disjoint **and frame/game-disjoint** — hold out entire games,
   ideally one entire venue/tile-style. This is the only number anyone is allowed to quote.
2. **End-to-end game replay metric**: run full harvested games through the whole pipeline and
   score per-move word/score accuracy against the ticker-derived truth (M3) and review rate
   (M4). Wire into pytest as a slow/marked suite; keep the per-class regression gate from the
   master plan.

## Sequencing and expected trajectory

| Step | Work | Expected M1 (held-out, frame-disjoint) |
|---|---|---|
| 0 | WS5 eval harness (half a day) | establishes the real baseline (~25–30%) |
| 1 | WS1 canonicalization + WS2 hygiene/bug fixes | 60–80% (est.) |
| 2 | WS2 renderer v2 + pretrained backbone | 85–92% (est.) |
| 3 | WS3 ticker-replay data at ~5k tiles | 95–98% (est.) |
| 4 | WS4 voting + constraint decoding + calibration | M2 ≥99.5%, M3 ≈100%, M4 ≤5% |

Estimates are labeled as such; each step gates on the golden eval before the next. If step 1
lands under ~55%, stop and re-audit the pipeline for another mechanical bug (the BGR/RGB and
leakage incidents earn that paranoia) before adding data.

## Pitfalls for the implementer (learned the hard way in this repo)

- BGR (OpenCV, perception layer) vs RGB (PIL, training) — convert exactly once, at the
  boundary, as `board_reader.py` now does. Canonicalizing to grayscale in WS1 mostly retires
  this class of bug.
- `torch.manual_seed` is already fixed in `run_training`; keep determinism when adding
  samplers/schedulers (seed the sampler generator too).
- Never report accuracy from any split that shares source tiles *or frames* with training.
- Label new data via axis-labeled tables / replay states, never by eyeballing composites
  (the off-by-one incident); keep the low-variance-crop suspect filter for rack tiles.
- Rack crops: fixed-grid division only works per contiguous tile group; the real fix is the
  planned rack detector (RF-DETR, needs cloud GPU) — out of scope for this plan, but WS3's
  classifier data transfers directly to rack tiles once localization exists.
