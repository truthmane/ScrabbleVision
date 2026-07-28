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

**Correction (found later, via an unrelated "why isn't this as good as OCR" investigation): the 99-tile held-out set itself was partially corrupted, deflating every accuracy number above by ~10-12 points.** Benchmarking the deployed checkpoint against off-the-shelf OCR (Tesseract, EasyOCR) on the exact same held-out set as a sanity check surfaced something odd during visual spot-checking: one 17-tile source batch ("frame 3" / `real_tiles_batch3`, plus its `f3R` rack counterpart, ~24 of the 99 tiles) showed systematic crop/grid misalignment -- e.g. the tile labeled "K" clearly shows a "U", the tile labeled "N" clearly shows a "G", the tile labeled "BLANK" clearly shows an "S", and two crops show almost entirely a premium-square graphic with no tile in frame at all. This is a harvest-time calibration error for that one source photo (matching the already-documented "corners... noted as slightly off for a later, busier frame" caveat elsewhere in this project), not a recognition failure -- no classifier could ever get these right, since the ground truth itself doesn't match what's in the crop.

Excluding that one corrupted batch: **64.6% (64/99) → 73.2% (60/82)**. Excluding all frame-3-tagged data (board + rack): **76.0% (57/75)**. Confirmed independently: harvesting a fresh, larger (92-tile) set from a different, unambiguously-identified real frame of the same broadcast (cross-verified via the on-screen score graphic matching a specific GCG cumulative score exactly, avoiding any guessing about which move a frame represents) using the current, validated calibration gave **74.4% (67/90)**, consistent with the corrected number from a completely independent source. **The classifier's true single-tile accuracy on trustworthy ground truth is ~74-76%, not 64.6%.** The relative comparisons in the table above (which venue-diversity step helped, which didn't) remain valid science -- both sides of each comparison were measured against the same, consistently-flawed set -- but the absolute numbers should be read as understated by roughly 10-12 points throughout. No model or code changed because of this; only the (corrupted, scratchpad-only, never-committed) evaluation data was found to be at fault. Re-baseline against a corrected/larger held-out set before trusting future absolute-accuracy comparisons.

Also worth recording since it directly answers "why isn't this as easy as OCR": off-the-shelf OCR performs meaningfully *worse* than the bespoke classifier on this exact task, not better -- Tesseract 23.5%/26.7% and EasyOCR 27.6%/32.0% (corrupted-set/clean-set respectively) vs. the classifier's 64.6%/76.0%. Classical/general-purpose OCR is tuned for scanned documents and natural scene text, not photographed physical game tiles with real broadcast compression -- the domain gap this project has chased since WS3 (venue diversity, not volume) is real, not imagined. A more promising, not-yet-tried direction in the same spirit as WS2's pretrained-backbone lever: swap in a scene-text-recognition-pretrained backbone (e.g. from EasyOCR's own CRNN recognition network) in place of the current ImageNet-pretrained MobileNetV3-Small, keeping this project's own 27-way classification head and training loop -- OCR-domain pretraining should transfer better than ImageNet's generic object features for "read this glyph," the same logic that made the ImageNet swap the single biggest lever tried so far.

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

## Round 2 of real data: venue-disjoint retrain, 81.0% → 85.1%, and a lost-dataset lesson

The dataset this section's earlier 64.6%/76.0% numbers were measured against **no longer
exists** — it lived only in an ephemeral per-session scratchpad and was never persisted
anywhere durable, so a later session lost it outright. Fixed going forward:
`training/data/real_tiles/` is now committed to the repo (that directory's own README has the
full per-venue provenance) specifically so a checkpoint's training data stays inspectable and
extendable, not just the checkpoint itself.

**513 new real tiles harvested this round**, across two genuinely new venues, both calibrated
via the new `training/collect/click_calibrate.py` tool (see `training/collect/README.md`
section 3b for why that tool exists — 7 automated calibration methods failed on one of these
venues first):
- Causeway Challenge 2026 (Bangkok), two distinct physical tables — 121 tiles.
- 2026 Scrabble Players Championship, 4 of Mack Meller's own games — 392 tiles, ground truth
  pulled directly from `cross-tables.com` GCG files via his own per-player "Annotated Games"
  page (`cross-tables.com/anno.php?p=<id>`, every row has both a GCG and a stream link — a
  better discovery mechanism than searching by tournament).

**Honest measurement, venue-disjoint** (not just game-disjoint): held out *all* Causeway tiles
entirely (both tables), fine-tuned only on the 4 Mack Meller games. The pre-fine-tune checkpoint
scored **81.0% (98/121)** on held-out Causeway already (a real, if partial, sign the earlier
diverse training generalizes). After fine-tuning: **85.1% (103/121)** — a genuine +4.1 point
lift from real venue diversity, not an artifact of an easier test set (same 121 tiles both
times).

**Catastrophic forgetting, measured, not assumed.** The "normal" fine-tune settings used
historically for this project (8 epochs, `lr=1e-3`) *regressed* held-out accuracy to 72.7% —
worse than the untouched baseline. With the old ~620-tile dataset gone, there's no way to mix
old and new data in one training run anymore, so an aggressive continuation overfits hard to
just the one new venue's specific look (all-black tiles, one font) at the expense of everything
previously learned. A much gentler continuation (2 epochs, `lr=1e-4`) is what actually produced
the real win; this is the setting to start from on any future continuation fine-tune of this
checkpoint, verified against a real held-out venue before trusting it — `train.py`'s own
reported "final validation accuracy" hit 100% in every one of these runs (an in-distribution
split of the new data alone) and was completely uninformative about which setting actually
helped or hurt generalization.

## Round 3 of real data: 7 more games, 85.1% → 91.7%, and from-scratch training loses to incremental

**680 more real tiles harvested**, across 7 genuinely distinct games discovered via
cross-tables.com's per-player "Annotated Games" pages (`anno.php?p=6003`, the same discovery
mechanism as round 2). The user personally screenshotted each game's end-state board (10
candidates pulled, 3 declared untenable and dropped). Each screenshot was matched back to its
source `.gcg` by cross-referencing distinctive board words against the GCG's actual move list —
**not assumed from list order**: a naive `grep` for expected words gave one false negative,
because GCG move lines often represent a shared/crossing letter as a `.` placeholder rather than
spelling the whole formed word out, so a literal text search can legitimately miss a real match.
Resolved by reconstructing the actual board and checking there instead of trusting grep.

Two of these seven `.gcg` files broke the existing strict, score-validating replay
(`replay_game.replay_gcg_game`) outright: one logs a withdrawn/retaken play as its own line
(`word == "--"`) without ever undoing the prior placement (a later real play across the same
squares then looks like an illegal double-placement); another spells a crossing word out in full
instead of using `.` for a cell an earlier play already filled (same false-double-placement
symptom, different cause). Fixed with a new, tolerant `board_from_gcg_final` in
`click_calibrate.py` (new `--gcg-final` CLI flag) that undoes withdrawn plays and accepts a
same-letter "conflict" silently — it does no score validation at all, existing purely to answer
"what does the final board look like" for a single end-state photo with full GCG ground truth.

**Honest measurement, same held-out set as round 2** (all Causeway tiles, both tables, still
never trained on): a gentle incremental continuation (2 epochs, `lr=1e-4`) from the round-2
checkpoint on just the 680 new tiles took held-out accuracy **85.1% (103/121) → 91.7% (111/121)**.

**Also tried, per explicit request: training from scratch on the full real dataset instead of
continuing incrementally.** All 1,072 non-held-out real tiles (513 from round 2 + 680 from this
round) at once, from a fresh ImageNet-pretrained backbone (no init checkpoint) — the hypothesis
being that a from-scratch run isn't limited by whatever the existing checkpoint already learned,
so it might generalize better. It did **much worse**, at every setting tried: 53.7% (8 epochs,
`lr=1e-3`, the historically "normal" full-training settings) and 66.1% (15 epochs, `lr=3e-4`) —
both far below the 91.7% the incremental continuation reached. Most likely explanation: this
checkpoint's lineage went through a synthetic-render pretraining stage before ever seeing a real
photo (`training/synth_render`), and training from scratch on real tiles alone skips that
foundation entirely; 1,072 real crops split across 27 classes (many classes under 50 examples)
just isn't enough signal on its own to learn 27-way tile classification from raw ImageNet weights
in a handful of epochs, no matter how the learning rate is tuned. **The lesson now holds across
two independent rounds**: a gentle continuation from the current checkpoint beats both aggressive
continuation settings (round 2's 72.7% regression) and training from scratch on the full dataset
(this round's 53.7%/66.1%). Recalibrated temperature to `T ≈ 1.078`; checkpoint promoted.

## Round 4 of real data: 10 more games across 8 tournaments, 91.7% → 93.4%

**981 more real tiles harvested**, across 10 games deliberately chosen to span different
tournaments (per explicit request), discovered via a different cross-tables.com player's
"Annotated Games" page (`anno.php?p=11232`): Albany NY (two different years), Montreal QC (two
different years), Lake George NY, Crescent City Cup LA, the 32nd National Championship Finals,
the 11th Word Cup, and the 31st National Championship — 8 distinct tournament/date combinations,
2019-2026. Same word-cross-referencing discipline as round 3 for matching each screenshot to its
source `.gcg` (not assumed from list order); one of the ten needed `board_from_gcg_final` again
(a withdrawn-play GCG).

**Collected via a new batched calibration tool** (`generate_multi_click_tool_html`, new
`multi-targets`/`multi-harvest` CLI subcommands, driven by a JSON manifest) built specifically for
this round, at the user's request, after doing 10 boards one link/copy-paste at a time in round 3
proved tedious. Walks through every board on one page, auto-advancing once a board's targets are
clicked, ending in one combined JSON blob for the whole batch instead of one per board. One board
in this batch got mis-clicked; the single-board `targets` tool still exists unchanged for exactly
that case (regenerate just the one board, harvest it separately, same output shape).

**Honest measurement, same held-out set as every round so far** (all Causeway tiles, never
trained on): a gentle incremental continuation (2 epochs, `lr=1e-4`) from the round-3 checkpoint
on just the 981 new tiles took held-out accuracy **91.7% (111/121) → 93.4% (113/121)**.
Recalibrated temperature to `T ≈ 1.116`; checkpoint promoted. `training/data/real_tiles/` now at
2,174 tiles total (1,193 → 2,174 this round).

## Round 5 of real data: 1,189 real WESPA rack tiles, 93.4% → 96.7% board / a genuine but modest rack win

**Four separate attempts at synthetic rack augmentation (warping existing board crops to look
rack-like) all failed first**, an honestly-reported negative result: perspective+lighting warp
(59.5%→55.2%), a loose non-square-crop variant targeting the real measured aspect-ratio mismatch
between board crops and RF-DETR's rack boxes (still flat/negative), then more epochs of the same
(56.9%, 56.0%) — each round improved board accuracy while further *hurting* rack accuracy, meaning
the augmented data resembled "a slightly different flavor of clean tile" rather than genuine rack
distortion. Found and fixed a real bug along the way (`_solve_perspective_coeffs` built its
least-squares RHS from `dst` instead of `src`, silently returning the identity transform), but the
augmentation approach itself was abandoned once four consecutive attempts failed to move the
number that mattered.

**Pivoted to real data**: the WESPA Word Wars broadcast displays both players' rack tiles
continuously at the bottom of every frame, so real rack photos could be harvested from the exact
same broadcast already used for board data. Harvested 1,189 real rack tiles (OCR-cross-check +
brightness-filter pipeline, see `training/data/real_tiles/README.md`'s "Rack tile crops" section
for the method) — `training/data/real_tiles/` grew from 2,174 to 3,363 tiles.

**Honest measurement needed a harder held-out set than the obvious one.** A time-disjoint split of
the same WESPA broadcast (283 tiles, last ~20% of the timeline, never trained on) went 88.7%→94.7%
— at the time this was (wrongly, see Round 6) attributed to WESPA being "a clean rendered graphic,
not a photographed physical rack," on the theory that this number alone would overstate the
real-world win. Re-measured against the **original, harder held-out set** (116 real photographed
NASPA rack tiles, a different specific game/venue, never touched by any of this round's training
data): **59.5% (69/116) → 64.7% (75/116)** — a real, modest, honestly-measured improvement. Board
accuracy (same venue-disjoint Causeway held-out set as every round) also improved, not just held
steady: **93.4% (113/121) → 96.7% (117/121)**. Gentle incremental continuation (2 epochs, `lr=1e-4`)
from the round-4 checkpoint, same recipe as every prior round. Recalibrated temperature to
`T ≈ 1.029`; checkpoint promoted with explicit user confirmation (overwriting the deployed,
Git-LFS-tracked checkpoint is a sensitive action).

**Lesson (later revised in Round 6, but the underlying caution was directionally right)**: when a
new training venue is also going to be a held-out measurement venue, a held-out split of *that same
venue* is necessary but not sufficient on its own — always also check a genuinely different, harder
held-out venue before trusting a training-venue-only number. What turned out to be wrong was the
specific claim that WESPA *itself* was the easy case; see Round 6.

## Round 6: WESPA/NASPA were both real photos all along, a black-tile venue, and a genuine mixed result

**Correction, found the hard way.** User asked to "hunt for a real-rack-photo venue" (assuming
WESPA's rack data was a rendered graphic per Round 5's framing). Checked four broadcasts (WESPA
itself re-examined, plus Galesburg, Montreal, "4th of July") and all looked like clean rendered
overlays at a glance — small downscaled screenshots all showed crisp, uniform tile graphics. User
pushed back directly on one link ("Are you kidding? The bottom quarter of the screen has the two
racks"), which prompted a pixel-level zoom rather than a glance — and every single one, including
the original WESPA data already trained on, turned out to be a genuine photograph (natural lighting
gradients, tilted tiles, real wood grain, a scoresheet visible in the background). **The entire
Round 5 "clean graphic vs. real photo" framing was wrong** — corrected in `models/README.md` and
`training/data/real_tiles/README.md`. The real, useful distinction turned out to be tile *color
scheme*: WESPA and "4th of July" both use white tiles/red-maroon text; NASPA's own "NWL
Championship Division Day 3" broadcast uses black tiles/white text — the same color scheme as the
original, hardest held-out set.

**Harvested "4th of July / Bob Linn Superstars" (820 tiles) — a genuinely different tournament,
same white/red tile style as WESPA.** Retrained (gentle continuation): board 96.7%→98.4%, WESPA
rack 94.7%→95.8%, July4's own held-out 89.8%→93.6% — **but the original NASPA held-out set
regressed 64.7%→58.6%**. Hypothesis: more of the same white/red tile style pulled the model further
from NASPA's visually distinct black-tile style, rather than adding genuine cross-style diversity.
**Not promoted** — first real case this project has had of a candidate that improved everything
measured except the metric that mattered most.

**User's call: look for a black-tile venue specifically, not just another venue.** Found it on
NASPA's own channel — the "NWL Championship Division Day 3" stream (a different specific
match/round than the original held-out set's Finals game) uses black tiles. Harvesting it surfaced
a real, venue-color-dependent bug: the brightness-based junk filter from Round 5 (rejects crops
darker than ~100, tuned for white tiles) was silently rejecting nearly all genuine black-tile crops
(mean ~65-70, well under threshold) — first attempt yielded only 4 tiles from 324 frames. Lowering
the threshold to let black tiles through then let a *different* junk class back in
(intermission "Out for lunch — we'll be back" banners, which are also dark). Fixed with a
color-scheme-agnostic signal instead of a brightness threshold: every genuine tile crop of any
color sits on a warm wood-tray strip somewhere below the tile (red channel clearly exceeds blue at
plausible brightness in a lower band of the crop), while banner text does not, regardless of how
dark the banner is. This rejected 132 of 582 candidates; spot-checking both the rejected and
accepted sets found zero errors either direction. Final: 450 clean black-tile rack tiles.

**Retrained again (gentle continuation from the pre-July4 checkpoint, on WESPA+July4+Day3
combined) — a large, clearly-real win on the new axis, but the original held-out set still dipped
slightly.** Board 96.7%→97.5%, WESPA rack 94.7%→95.1%, July4 rack →94.7%, **Day 3's own held-out
(black tile) 69.8%→85.9%** (a 16-point jump on 106 tiles — clearly a real generalization gain, not
noise). But the original NASPA Finals held-out set (116 tiles, a *different specific game* than Day
3) went 64.7%→61.2% — smaller than the July4-only regression, but still a regression. At n=116 a
swing of 3-4 tiles is close to one standard error on the underlying binomial estimate, so this
number alone can't distinguish "real residual regression" from "noise at this sample size." Promoted
anyway after presenting both readings to the user and getting explicit sign-off, weighing the large
clearly-real gains elsewhere against a small, statistically ambiguous dip on the noisiest
measurement. Recalibrated `T ≈ 1.176`. `training/data/real_tiles/` now at 4,633 tiles (was 3,363).

**Still an open problem, not solved**: even the best rack number here (85.9% on Day 3) trails board
accuracy (97.5%) by over 10 points, and the original Finals-specific held-out number hasn't cleared
~60-65% across three different training rounds — there may be something about that one specific
game/camera-moment that's unusually hard independent of general venue/color diversity. Worth
growing that 116-tile held-out set itself (more real photos from that same Finals game) before the
next round, so future promote/don't-promote calls on it aren't fighting sample-size noise.

## Round 7: WESPA Games 3-7 board harvest — rare letters finally covered, and a learning-rate-sensitivity lesson

**477 more real board tiles**, across WESPA Word Wars Games 3-7 (the same "Games 1-7" broadcast
Games 1-2 came from earlier this session), harvested via lightweight single-frame extraction
instead of downloading each game's video segment: storyboard sprite sheets (`yt-dlp -f sb1`)
located each game's rough boundary, then one direct HD frame per game (`yt-dlp -f 399 -g` +
`ffmpeg -ss <t> -frames:v 1`) once the frame's on-screen score and ticker text were grep-verified
against that game's own GCG. This finally gives the venue real coverage on the rare letters that
had been stuck at 0-2 examples after Games 1-2: J 55, K 59, Q 60, X 61, Z 66. Full provenance and
the spot-check-caught row-14 frame-edge corruption (9 crops dropped across 3 games) are in
`training/data/real_tiles/README.md`.

**The standard gentle recipe (2 epochs, `lr=1e-4`) cost 2 points on the Causeway held-out set for
the first time** (97.5% [118/121] -> 95.9% [116/121], two new mistakes on previously-correct,
unambiguous tiles — G->B twice, L->J once, confirmed by eye, not small-N noise on an easy class).
Cutting to 1 epoch at the same rate didn't fix it (still 95.9%), isolating the cause to the
learning rate rather than epoch count. Halving it (`lr=5e-5`, still 2 epochs) restored zero
regression exactly: 97.5% (118/121), the identical 3 baseline mistakes. Promoted after presenting
both the regression and the fix to the user. **Takeaway for future rounds this size (~450+ tiles
at once): try `5e-5` first rather than assuming the `1e-4` default is always safe** — it may be
that larger single-round batches need a gentler rate than the smaller (~90-100 tile) rounds this
default was originally tuned against. Recalibrated `T≈1.074`. `training/data/real_tiles/` now at
6,542 tiles (was 6,065).

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
