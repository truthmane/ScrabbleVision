# NASPA Finals rack held-out set — evaluation only, NEVER training data

319 real rack tile crops (black tile / white text), all from the same single broadcast: the 2026
Scrabble Players Championship "Day 5 | Best-of-Five Finals | NWL" match (Orry Swift vs. Mack
Meller), `youtube.com/watch?v=hC-e2HikEZU`.

**This directory must never be merged into `training/data/real_tiles/` or otherwise used as
training data.** Its entire value is as a fixed, honest, never-trained-on benchmark for measuring
real rack-crop accuracy on the hardest domain this project has found (a physical black-tile rack,
photographed at an angle, under real venue lighting — not a clean board-crop or a broadcast
graphic). See `docs/classifier-accuracy-plan.md`'s "Round 5/6" sections and `models/README.md` for
the full retrain history.

**The original 116-tile version of this set was thrown out (2026-07-28) after the user spot-checked
it and confirmed what an isolated re-measurement had flagged**: the deployed checkpoint scored only
61.2% on those 116 tiles but 90.3% on the 319 below (harvested from the identical broadcast/domain)
— a ~29-point gap that meant one of the two measurements was wrong, not that the domain itself is
uniquely hard. Visual inspection found the old 116 contained real harvest-time defects — crops
showing empty wood-grain background with no tile at all, and crops labeled for one letter but
actually centered on/dominated by the adjacent tile — the same failure mode as an earlier
"corrupted eval batch" incident already documented in this project for board-crop data. **Every
accuracy number quoted anywhere in this project's history against "the 116-tile NASPA held-out
set" (59.5%, 64.7%, 58.6%, 61.2%) should be read as measured against this now-discredited data and
is not necessarily a reliable reflection of real model accuracy on this domain.**

## Provenance

- `<LABEL>_naspa_finals_bcast_f<seconds>_<L|R><idx>.png` (319 tiles): harvested at scale by sampling
  the full ~8h23m broadcast every 100s and running the same OCR-cross-check + wood-tray-strip
  filter pipeline used for the training-side NASPA Day 3 harvest (see
  `training/data/real_tiles/README.md`'s "Rack tile crops" section for the exact method — this
  directory's tiles used identical code, just pointed at this video and this output directory
  instead of `training/data/real_tiles/`). Spot-checked via a 30-tile random montage, zero
  mislabels found.

No exact-duplicate crops exist (checked via content hash).

## Regenerating / extending

The harvest script itself is not committed (same reasoning as the training-side rack harvests: a
one-off scratchpad pipeline tuned to this broadcast's specific layout/color scheme, not a reusable
tool). To add more tiles from this same broadcast: extract more frames via
`ffmpeg -ss <t> -i <video> -frames:v 1`, run the same OCR-cross-check + wood-tray-strip-filter
pipeline described in `training/data/real_tiles/README.md`, spot-check, then merge into the
appropriate `<LABEL>/` subdirectory here — never into `training/data/real_tiles/`.
