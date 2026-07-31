# Real-photo data collection workflow

How the real tile crops used to fine-tune `models/tile_classifier_v1.pt` were collected, and
how to repeat the process (this is also the prerequisite for WS3 in
`docs/classifier-accuracy-plan.md` — scaling ground truth via ticker replay).

## 1. Capture a frame

There's no standalone capture script for this step — it's done interactively via a browser tool
(this repo was built using Claude's browser-automation tools, but any browser + devtools works):

1. Open a broadcast video (e.g. a NASPA or "Let's Play Scrabble" YouTube VOD) and pause on a
   frame with a clear, well-lit board.
2. Force the highest available resolution (on YouTube, via the player's own quality settings or
   `document.getElementById('movie_player').setPlaybackQuality('hd1080')` in the console).
3. Draw the current video frame to a canvas and export it, e.g. in the browser console:
   ```js
   const v = document.querySelector('video');
   const c = document.createElement('canvas');
   c.width = v.videoWidth; c.height = v.videoHeight;
   c.getContext('2d').drawImage(v, 0, 0);
   const dataUrl = c.toDataURL('image/jpeg', 0.85);
   ```
4. Save that data URL to a file. If you captured it through an agent tool that saves large
   results to a JSON file instead of returning them inline, `decode_frame.py` handles unwrapping
   that specific format; otherwise just base64-decode `dataUrl` directly.

## 2. Decode the captured frame

```
python -m training.collect.decode_frame <tool_result.json> <out.jpg>
```

## 3. Calibrate and rectify

Use `autoscorer.perception.calibration.homography.calibrate_from_corners` (or
`calibrate_from_aruco` if markers are present) to rectify the raw frame to the canonical
900x900 grid. Contour-based auto-detection of the board's outer edge works well for boards
photographed against a plain/dark background; otherwise pick the four corners manually by eye.
See `docs/classifier-accuracy-plan.md`'s Results section and the git history around the first
data-collection pass for worked examples of both.

If this rectified frame becomes a venue's `reference_board_path` (see `VenueProfile`), save it
under `~/.autoscorer/venue_references/`, not a session-scoped scratchpad or anywhere inside a git
worktree -- both get cleaned up between sessions and will silently break
`load_reference_board()` for that venue until someone notices and regenerates it.

### 3b. When automated/eyeballed calibration won't converge: `click_calibrate.py`

Some venues defeat every automated corner-detection method — a board that's small on-screen,
tilted, or shot at an angle can produce a homography that looks fine at a glance but is off by
most of a cell width once you crop individual tiles (measured directly on a real 2026 Causeway
Challenge frame: 7 different automated methods — color-mask contours, Hough-line rotation/pitch
measurement, least-squares homography fits, even a ground-truth-driven grid-search optimization —
all converged on "center cell perfect, edges drift into background," the signature of real
non-uniform perspective distortion none of those methods can model). Guessing corner pixels from
static screenshots doesn't fix this either — human visual/spatial judgment does, just not through
a screenshot round-trip.

**The key realization this tool acts on: you never need a human to identify *letters*, or even to
have a game's ground truth on hand, just to locate the grid.** Every standard board has the same 8
Triple Word Score squares + center/start square in the same fixed positions in every game — so the
*default* calibration targets need no ground truth at all, work on a totally empty board, and only
need clicking **once per physical venue/camera setup**, not once per game:

```
# Step 1: generate the clicker. No game data needed for the default (fixed-square) targets.
python -m training.collect.click_calibrate targets FRAME.jpg --out clicker.html

# Open clicker.html in a browser. It shows the frame and asks you to click each of
# the 9 named squares in order, by standard notation ("Click H8 (center star)",
# "Click A1 (TWS (red/pink))", ...) -- a board's own printed row/column labels make
# this faster and less error-prone than hunting for a specific letter among tiles.
# It then displays a JSON blob once you're done. Save that as clicks.json.

# Step 2: fit a homography from the clicks (RANSAC over however many points you
# gave it, so one imprecise click doesn't wreck the whole fit), harvest every
# occupied cell (this step DOES need ground truth, to label the crops), and build
# a spot-check montage automatically.
python -m training.collect.click_calibrate harvest FRAME.jpg \
    --woogles-doc game_document.json --clicks clicks.json \
    --out-dir harvest/ --prefix my_venue
# (or --gcg some_game.gcg --move 25 instead of --woogles-doc)
```

Since the fixed-square calibration only depends on the camera, not the game, do step 1 once per
venue and reuse the resulting `clicks.json` (and the homography it fits) across every game/frame
from that same camera — that's the actual point for scaling data collection.

If a frame happens to have every premium square covered by tiles (or a venue's board doesn't use
standard coloring), fall back to occupied-cell targets instead: add `--use-occupied-cells
--woogles-doc game_document.json` (or `--gcg`/`--move`) to the `targets` command — picks a handful
of spread-out occupied cells via farthest-point sampling instead of the fixed squares.

Always look at `harvest/_spotcheck_<prefix>.jpg` before adding anything to training data — the
`harvest` command also prints each click's reprojection error, so a target that came back
suspiciously high can be re-clicked without redoing the whole set.

**Pulling ground truth from a woogles.io broadcast** (a good source independent of cross-tables.com/GCG
files, useful when an event isn't cross-tables-indexed): any broadcast's per-game "Review" link is
`woogles.io/anno/<id>`; the full move-by-move document, including the final board as a base64 byte
array, is one POST away with no auth and no scraping:

```
curl -X POST https://woogles.io/api/omgwords_service.GameEventService/GetGameDocument \
    -H 'Content-Type: application/json' -d '{"gameId": "<id-from-the-anno-URL>"}'
```

`board_from_woogles_document` in `click_calibrate.py` decodes that response directly into a real
`BoardState`, so it plugs into the exact same `harvest_board_cells` used by GCG-sourced games.

## 4. Read ground truth precisely — don't eyeball a composite

```
python -m training.collect.make_table <rectified.jpg> <out_table.png> <r1> <r2> <c1> <c2>
```

Produces an axis-labeled contact sheet (row/col numbers overlaid) of every cell in the given
range. Read letters off *this*, not off a raw board photo or a composite grid you built by hand
— the very first collection pass had an off-by-one row error caught exactly this way, and
subsequent reads were cross-checked against words visible in the source image as a sanity check.

Once you have a `{(row, col): letter}` mapping, crop board cells directly with
`autoscorer.perception.calibration.homography.crop_cell(rectified_image, row, col)` and save
each to `<letter>_<label>.png`.

## 5. Rack tiles need a different approach than board cells

Real physical rack tiles are **not evenly spaced** — players leave gaps, per the earlier session
finding that a fixed 7-way division only works when a rack happens to be completely full and
contiguous. Contour detection doesn't cleanly separate touching tiles either (they merge into
one blob). The working approach is manual per-group bounding boxes:

```
python -m training.collect.crop_rack <rack_crop.jpg> <out_dir> <prefix> \
    <x1> <x2> <letters,comma,separated> [<x1> <x2> <letters> ...]
```

Each `(x1, x2, letters)` group is evenly divided into `len(letters)` slots — use one group per
*contiguous* cluster of tiles in the rack image (a rack showing "BATE  ELS" with a gap in the
middle needs two groups, not one covering the whole rack).

This is a real, unsolved gap for a *fully automated* rack-reading pipeline: it needs real object
detection (RF-DETR, per the earlier architecture discussion) trained on bounding-box-labeled
rack images, which doesn't exist yet. Manual per-group cropping is a stand-in for data collection
purposes only.

## 6. Verify before trusting it

Build a contact sheet of the extracted crops with their labels overlaid and eyeball it for
mismatches before adding anything to a training set — cheap insurance against exactly the kind
of labeling bug this workflow is designed to catch early instead of baking into a model.
