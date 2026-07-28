# NASPA Finals rack held-out set — evaluation only, NEVER training data

435 real rack tile crops (black tile / white text), all from the same single broadcast: the 2026
Scrabble Players Championship "Day 5 | Best-of-Five Finals | NWL" match (Orry Swift vs. Mack
Meller), `youtube.com/watch?v=hC-e2HikEZU`.

**This directory must never be merged into `training/data/real_tiles/` or otherwise used as
training data.** Its entire value is as a fixed, honest, never-trained-on benchmark for measuring
real rack-crop accuracy on the hardest domain this project has found (a physical black-tile rack,
photographed at an angle, under real venue lighting — not a clean board-crop or a broadcast
graphic). Every retrain in this project's history has been measured against this set (or an
earlier, smaller version of it) specifically because it is the one number that resisted easy wins:
59.5% → 64.7% → 58.6% (regression) → 61.2%, across three separate training rounds, none of which
came close to the ~97% board accuracy achieved elsewhere. See `docs/classifier-accuracy-plan.md`'s
"Round 5/6" sections and `models/README.md` for the full history.

## Provenance

- `<LABEL>_orig116_*.png` (116 tiles): the original held-out set, assembled early in this project
  from several earlier scratchpad harvesting passes against this same broadcast (`rack_tiles/`,
  `ws3d/harvested_racks/`, `ws3d/harvested_racks_g22/`, `ws3d/harvested_racks_g25/` — the "g22"/"g25"
  suffixes are historical directory names from that harvesting pass, not meaningful game IDs here).
  This set lived only in ephemeral per-session scratchpad for a long time and was at real risk of
  being lost between sessions (the same failure mode `training/data/real_tiles/` was committed to
  prevent) — committing it here fixes that.
- `<LABEL>_naspa_finals_bcast_f<seconds>_<L|R><idx>.png` (319 tiles): harvested at scale by sampling
  the full ~8h23m broadcast every 100s and running the same OCR-cross-check + wood-tray-strip
  filter pipeline used for the training-side NASPA Day 3 harvest (see
  `training/data/real_tiles/README.md`'s "Rack tile crops" section for the exact method — this
  directory's tiles used identical code, just pointed at this video and this output directory
  instead of `training/data/real_tiles/`). Spot-checked via a 30-tile random montage, zero
  mislabels found.

No exact-duplicate crops exist across the full 435 (checked via content hash).

## Regenerating / extending

The harvest script itself is not committed (same reasoning as the training-side rack harvests: a
one-off scratchpad pipeline tuned to this broadcast's specific layout/color scheme, not a reusable
tool). To add more tiles from this same broadcast: extract more frames via
`ffmpeg -ss <t> -i <video> -frames:v 1`, run the same OCR-cross-check + wood-tray-strip-filter
pipeline described in `training/data/real_tiles/README.md`, spot-check, then merge into the
appropriate `<LABEL>/` subdirectory here — never into `training/data/real_tiles/`.
