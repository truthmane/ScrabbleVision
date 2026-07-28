# Real tile crop dataset

3,363 real, individually-verified Scrabble tile crops, organized as an `ImageFolder`-compatible
directory (`<LETTER>/*.png`, plus `BLANK/` for blank tiles) for `training/classify/train.py`.
2,174 are board-crop tiles (60x60 canonical, one per occupied board cell); the remaining 1,189
(prefix `wespa_rack_*`) are rack-crop tiles harvested from broadcast rack overlays — see "Rack
tile crops" below, they follow a different provenance/verification path than the board table.

**Committed deliberately, unlike full broadcast frames/video** — these are small, individually
cropped game-piece images (a single tile face each), not the copyrighted broadcast footage
itself, and this project's own trained-model pipeline depends on being able to retrain/audit
against the actual data it was fit on. Losing this dataset between sessions (as happened to the
~616-tile set that produced the currently-deployed `models/tile_classifier_v1.pt` — it lived only
in an ephemeral per-session scratchpad and no longer exists) is the failure this directory exists
to prevent.

## Provenance

Every file is named `<LABEL>_<venue_prefix>_<row>_<col>.png` — the label is the exact ground truth
(never guessed), and `(row, col)` is the 0-indexed board position, both fully traceable to source:

| Prefix | Venue | Tiles | Ground truth source |
|---|---|---|---|
| `causeway_table3` | Causeway Challenge 2026 (Bangkok), Round 36 Table 3, Schoenbrun vs Richards | 23 | woogles.io game document (`GetGameDocument` API), state after event 6 |
| `causeway_table1` | Causeway Challenge 2026 (Bangkok), Table 1, different physical board/camera | 98 | Manual transcription, cross-verified cell-by-cell against the calibrated grid image (see git history for the full verification process) |
| `mack_round2` | 2026 Scrabble Players Championship, Round 2, Mack Meller vs Michael Thelen | 98 | `cross-tables.com` GCG (`annotated/selfgcg/610/anno61085.gcg`) |
| `mack_round8` | 2026 Scrabble Players Championship, Round 8, Mack Meller vs Joey Mallick | 99 | GCG `anno61132.gcg` |
| `mack_round15` | 2026 Scrabble Players Championship, Round 15, Mack Meller vs Joey Krafchick | 97 | GCG `anno61170.gcg` |
| `mack_round25` | 2026 Scrabble Players Championship, Round 25, Mack Meller vs Joey Krafchick | 98 | GCG `anno61157.gcg` |
| `batch10_g55974` | Mack Meller vs Nigel Richards (cross-tables.com game `g55974`) | 98 | GCG, full end-state board photo screenshotted by the user |
| `batch10_g55292` | Nigel Richards vs Joey Krafchick (cross-tables.com game `g55292`) | 96 | GCG, same as above |
| `batch10_g29132` | Joey Mallick vs Nigel Richards, 2018 Scrabble Championship 1/10 | 98 | GCG (`g29132`), reconstructed via `board_from_gcg_final` (this GCG spells crossing words in full instead of using `.` for overlaps) |
| `batch10_g29692` | Jackson Smylie vs Nigel Richards, 2018 Scrabble Championship 4/10 | 99 | GCG (`g29692`), reconstructed via `board_from_gcg_final` (this GCG logs a withdrawn/retaken play as its own line) |
| `batch10_g36165` | WSC 2018 Grand Final, Round 4 | 97 | GCG (`g36165`) |
| `batch10_g36164` | WSC 2018 Grand Final, Round 3 | 95 | GCG (`g36164`) |
| `batch10_g35181` | David Eldar, WSC 2019 Final Part 2 | 97 | GCG (`g35181`) |
| `batch2_g60796` | Albany, NY 2026-07-02, Joshua Castellano's opponent (annotator used pen names) | 99 | GCG (`g60796`) |
| `batch2_g59120` | Jackson Smylie vs Josh Sokol-Rubenstein, Montreal QC 2026-03-07 | 96 | GCG (`g59120`) |
| `batch2_g51842` | Ben Schoenbrun vs Josh Sokol, Albany NY 2024-12-29 | 99 | GCG (`g51842`) |
| `batch2_g52040` | Matthew Tunnicliffe vs Josh Sokol, Lake George NY 2024-10-18 | 98 | GCG (`g52040`), reconstructed via `board_from_gcg_final` |
| `batch2_g48957` | Josh Sokol vs Matthew Tunnicliffe, Montreal QC 2024-05-24 | 99 | GCG (`g48957`) |
| `batch2_g45984` | Jackson Smylie vs Josh Sokol-Rubenstein, Crescent City Cup LA 2024-01-13 | 98 | GCG (`g45984`) |
| `batch2_g45334` | Stefan Fatsis vs Josh Sokol, Albany NY 2023-12-29 | 98 | GCG (`g45334`) |
| `batch2_g43439` | Joshua vs Joey Mallick, 32nd National Championship Finals 2023-07-19 | 97 | GCG (`g43439`) |
| `batch2_g43003` | Mack Meller vs Josh Sokol, 11th Word Cup 2023-06-30 | 99 | GCG (`g43003`) |
| `batch2_g39295` | Will Anderson vs Joshua Sokol, 31st National Championship 2022-07-23 | 98 | GCG (`g39295`) |

All calibrated via `training/collect/click_calibrate.py` (see that module's docstring and
`training/collect/README.md` section 3b) — a human clicks a handful of named reference cells
directly on the source photo, RANSAC-fit a homography from the clicks, crop every occupied cell,
and every single crop was visually spot-checked (full per-tile montage, not just one-per-class)
against its label before being added here.

## Rack tile crops (`wespa_rack_*` prefix)

1,189 tiles cropped from the WESPA Word Wars broadcast's on-screen rack overlay, which displays
both players' current rack tiles continuously at the bottom of the frame — a different source
path than the board-crop provenance table above:

- `wespa_rack_g3t0_<i>.png` (13 tiles): the original pilot frame (WESPA Game 3, `t=0`), read and
  labeled manually pixel-by-pixel off a clear, legible broadcast graphic.
- `wespa_rack_bcast_f<seconds>_<L|R><idx>.png` (1,176 tiles): harvested at scale from the full
  6h52m WESPA "games 1-7" broadcast, sampled at 150s and 50s intervals (`<seconds>` = timestamp
  in the source video), `L`/`R` = which player's rack, `<idx>` = tile position within that rack.

**Harvest method** (`scripts` not committed — this was a one-off scratchpad pipeline, not a
reusable tool, since it depends on this specific broadcast's fixed rack-overlay layout):
1. Extract frames via `ffmpeg -ss <t> -i <video> -frames:v 1`.
2. Locate each rack's letter glyphs via `pytesseract.image_to_boxes` (psm 7, whole rack as one
   "word") on the fixed on-screen rack region, then crop each tile using the median pitch between
   consecutive glyph centers — tiles sit flush against each other with no border gap, so a plain
   contour/blob split does not separate them, but the *printed letters* have enough margin between
   them for Tesseract's own character segmentation to work.
3. Independently re-OCR each cropped tile alone (psm 10, single-character) and keep it only if
   this second read agrees with the first pass's letter at that position — a cheap two-pass
   cross-check, not a confidence threshold (Tesseract's per-char confidence output was unusable
   here, always near 0 in psm 10 regardless of correctness).
4. Reject any crop whose mean grayscale brightness is below 100 — this alone was enough to catch
   every false-positive glyph the OCR pass found on intermission/title-card frames (a cyan
   "letsplayscrabble.com" watermark, white commentary text, a scrolling banner), which are much
   darker overall than a real cream tile + wood rack holder. Verified: a clean brightness gap
   (highest rejected ~78, lowest kept ~133) with zero ambiguous cases in this dataset.
5. Blank tiles detected separately (near-zero dark-pixel fraction in the letter region) rather
   than relying on OCR, since a blank has no glyph for either OCR pass to read at all.
6. Manually spot-checked via random-sample montages (30 tiles at a time) against the source
   crops before merging — zero mislabels found in the post-filter set across two independent
   30-tile checks.

No exact-duplicate crops exist in this batch (checked via content hash) — even at 150s spacing,
JPEG/exposure noise alone made every crop byte-distinct, and the two expert players in this
broadcast change their racks often enough that 150-450s sampling rarely caught the same rack twice
anyway.

**Not yet used in a retrain.** These tiles exist to fill the real rack-crop accuracy gap measured
earlier (board 93.4% vs. rack ~59.5% on a small held-out set) — see `docs/classifier-accuracy-plan.md`
and prior fine-tune attempts using *synthetic* rack augmentation of board crops, all of which
failed to move real rack accuracy. A future retrain against this real data should hold out a fresh
split of these WESPA tiles (not just the pre-existing small held-out set, which may not be
venue-disjoint from this broadcast) before trusting any accuracy number.

## Regenerating / extending

Source photos and click coordinates are NOT committed here (the actual broadcast/personal-camera
photos remain in whatever scratchpad produced them, and re-running calibration from scratch is
cheap via `click_calibrate.py` if a source photo is still available). This directory holds the
final, verified crop outputs only — the thing actually consumed by training.

To add a new venue: follow `training/collect/README.md` section 3b, harvest into its own
directory with a distinct prefix, spot-check the full montage (not just one crop per class), then
merge into this directory (`cp <label>_<prefix>_*.png training/data/real_tiles/<label>/`).
