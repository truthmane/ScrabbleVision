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
