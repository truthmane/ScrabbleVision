# Real tile crop dataset

5,880 real, individually-verified Scrabble tile crops, organized as an `ImageFolder`-compatible
directory (`<LETTER>/*.png`, plus `BLANK/` for blank tiles) for `training/classify/train.py`.
2,174 are board-crop tiles (60x60 canonical, one per occupied board cell); the remaining 3,706 are
rack-crop tiles (prefixes `wespa_rack_*`, `july4_rack_bcast_*`, `naspa_day3_rack_bcast_*`)
harvested from broadcast footage of real physical racks — see "Rack tile crops" below, they follow
a different provenance/verification path than the board table. (July4 and Day3 were each harvested
in two passes at interleaved timestamps — 100s then a second 100s pass offset by 50s — for roughly
double the density of the first pass; see the provenance table below for the exact per-venue split.)

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

## Rack tile crops (all genuine real photographs of physical racks)

2,459 tiles cropped from broadcast footage of real physical racks, across three distinct
tournament venues/productions — a different source path than the board-crop provenance table
above. **Correction worth stating plainly**: an earlier version of this README described the
WESPA tiles as coming from "a clean rendered graphic overlay." That was wrong — direct pixel-level
zoom on all three venues below shows genuine photographs (natural lighting gradients, tilted
tiles, real wood grain, background clutter), not rendered graphics. The mistake came from judging
small downscaled screenshots instead of zooming in; caught after the user pushed back directly on
one of them.

| Prefix | Venue | Tile style | Tiles | Source video |
|---|---|---|---|---|
| `wespa_rack_g3t0_<i>` | WESPA Word Wars (Albany NY), Game 3 pilot frame | white tile, red/maroon text | 13 | manually read pixel-by-pixel |
| `wespa_rack_bcast_f<s>_<L\|R><i>` | WESPA Word Wars, full "games 1-7" broadcast | white tile, red/maroon text | 1,176 | `youtube.com/watch?v=YojCdbthGQY` |
| `july4_rack_bcast_f<s>_<L\|R><i>` | "4th of July / Bob Linn Superstars" tournament, Day 1 | white tile, red/maroon text | 1,554 | `youtube.com/live/ROL1VARMt9U` |
| `naspa_day3_rack_bcast_f<s>_<L\|R><i>` | 2026 Scrabble Players Championship, NWL Div. Day 3 (NASPA's own channel) | **black tile, white text** | 963 | `youtube.com/watch?v=ncHZby73h4I` |

`<s>` = timestamp in seconds in the source video; `L`/`R` = which player's rack; `<i>` = tile
position within that rack.

**Harvest method** (scripts not committed — a one-off scratchpad pipeline per broadcast, not a
reusable tool, since each depends on that specific broadcast's fixed rack-overlay layout and tile
color scheme):
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
4. Two junk filters, both needed because they catch different failure modes and neither alone is
   color-scheme-agnostic: a brightness floor (rejects crops far darker than any real tile of that
   venue's color — but this threshold is tile-color-dependent, roughly 100 for white tiles vs. as
   low as 20 for black tiles, so it must be tuned per venue, not reused blindly) and a **wood-tray
   strip check** (`has_wood_tray_strip`: scans a lower band of the crop for any row where red
   clearly exceeds blue at plausible brightness — every genuine tile crop, of any tile color, sits
   on a warm wood-tray background somewhere below the tile, while intermission/announcement banner
   text does not, regardless of how dark the banner itself is). The wood-strip check was added
   specifically because the NASPA Day 3 harvest (black tiles) needed such a low brightness floor
   that "Out for lunch — we'll be back" / "See you tomorrow @ 9:00" banner crops started passing
   the brightness filter alone; the wood-strip check rejected all of them (132 of an initial 582
   candidates) while a spot-check of both the rejected and accepted sets found zero errors either
   direction.
5. Blank tiles detected separately (near-zero dark-pixel fraction in the letter region) rather
   than relying on OCR, since a blank has no glyph for either OCR pass to read at all.
6. Manually spot-checked via random-sample montages (24-30 tiles at a time) against the source
   crops before merging — zero mislabels found in the post-filter set across every venue's checks.

No exact-duplicate crops exist in any of these batches (checked via content hash) — even at
50-150s sampling spacing, JPEG/exposure noise alone made every crop byte-distinct, and expert
players change their racks often enough that repeat sampling rarely caught the same rack twice.

**Retrained and honestly re-measured across three rounds** (full numbers, including the mid-series
regression that motivated hunting for a black-tile venue, are in `models/README.md`). Net result
after all three: board accuracy 93.4%→97.5%, and — measured against the harder, non-WESPA-style
held-out sets rather than trusting the easiest number — the NASPA Day 3 held-out set (black tile,
same color scheme as the hardest original benchmark) went 69.8%→85.9%. This checkpoint is now
`models/tile_classifier_v1.pt`.

## Regenerating / extending

Source photos and click coordinates are NOT committed here (the actual broadcast/personal-camera
photos remain in whatever scratchpad produced them, and re-running calibration from scratch is
cheap via `click_calibrate.py` if a source photo is still available). This directory holds the
final, verified crop outputs only — the thing actually consumed by training.

To add a new venue: follow `training/collect/README.md` section 3b, harvest into its own
directory with a distinct prefix, spot-check the full montage (not just one crop per class), then
merge into this directory (`cp <label>_<prefix>_*.png training/data/real_tiles/<label>/`).
