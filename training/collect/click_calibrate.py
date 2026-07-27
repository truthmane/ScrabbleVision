"""Repeatable venue calibration when corner/homography detection can't get
pixel-accurate on its own (see docs/classifier-accuracy-plan.md's Causeway
Challenge writeup: 7 automated methods all converged on "center cell
perfect, edges drift" -- a signature of a board too small on-screen for
any of those methods, not a bug to keep chasing).

The insight this module acts on: **we never need a human to identify
letters, or even to have a game's ground truth on hand, just to locate
where the grid is.** Every standard 15x15 Scrabble board has the same 8
Triple Word Score squares and center/start square in the same fixed
positions, in every game, on every board of that design -- so the default
calibration targets (`pick_fixed_board_targets`) need no ground truth at
all, work on a completely empty board, and only need clicking ONCE per
physical venue/camera setup, not once per game. (A fallback,
`pick_calibration_targets`, targets spread-out *occupied* cells instead,
for when premium squares happen to be covered or a venue's board doesn't
use standard coloring -- that one does need a game's ground truth, a
woogles.io game document or a replayed .gcg, same as
`harvest_from_replay.py` already uses.) Either way, a sighted human
clicking a handful of named reference points directly on the photo solves
the geometry in under a minute, far faster and more precisely than any
automated corner-detection attempted here first.

Two-step, repeatable workflow (same shape for any future venue/frame):

    1. `python -m training.collect.click_calibrate targets FRAME.jpg --out clicker.html`
       Opens/writes a self-contained HTML page prompting for the 8 TWS
       squares + center, by standard notation ("H8", "A1", ...) -- a
       physical board's own printed row/column labels make this faster and
       less error-prone than hunting for a specific letter among a dozen
       tiles. A human opens it in a browser, clicks each in order, and
       copies the JSON blob the page displays once done. (Pass
       `--use-occupied-cells --woogles-doc game_document.json` instead if
       premium squares aren't usable in this particular frame.)

    2. `python -m training.collect.click_calibrate harvest FRAME.jpg
       --woogles-doc game_document.json --clicks clicks.json
       --out-dir harvest/ --prefix venue_name`
       Fits a homography from the clicks (RANSAC over all of them, so one
       stray click doesn't wreck the fit), rectifies, crops every
       occupied cell via the existing `harvest_from_replay.harvest_board_cells`
       (same file-naming convention, so downstream tooling doesn't need to
       know which calibration method produced a given crop), and writes a
       labeled spot-check montage -- inspect that before ever adding the
       crops to training data, per this project's own hard-won batch3
       lesson. This step always needs ground truth (to know what letter
       goes in each crop's filename), even when step 1 didn't.
"""
from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from autoscorer.gamelogic.board import BoardState, Coord, Tile
from autoscorer.gamelogic.notation import format_square
from autoscorer.perception.calibration.homography import (
    CANONICAL_CELL_PX,
    BoardCalibration,
)
from training.collect.harvest_from_replay import BLANK_LABEL, harvest_board_cells


@dataclass(frozen=True)
class CalibrationTarget:
    row: int
    col: int
    label: str  # the letter/BLANK actually sitting at this cell -- a
    # confirmation hint only, not the primary instruction; see `notation`.

    @property
    def notation(self) -> str:
        """Standard tournament square notation ("H8", "A2", ...) -- this
        project's own convention (`notation.format_square`) for showing a
        coordinate to a human. The primary click instruction: counting a
        board's own printed row/column labels (nearly every tournament
        board has them) is faster and less error-prone than hunting for a
        specific letter among a dozen tiles, especially rotated/tilted or
        photographed at low resolution."""
        return format_square((self.row, self.col))


TRIPLE_WORD_SCORE_SQUARES: Tuple[Coord, ...] = (
    (0, 0), (0, 7), (0, 14),
    (7, 0),         (7, 14),
    (14, 0), (14, 7), (14, 14),
)
"""Fixed on every standard 15x15 Scrabble board, in every game -- unlike an
occupied cell's letter, these never depend on which game or move you
happen to have a frame from."""

CENTER_SQUARE: Coord = (7, 7)


def pick_fixed_board_targets(board: Optional[BoardState] = None) -> List[CalibrationTarget]:
    """The 8 Triple Word Score squares plus the center/start square -- 9
    points, always in the same place on any standard board, regardless of
    game state. This is the preferred calibration target set: unlike
    `pick_calibration_targets` (which needs a game's ground truth just to
    know where its targets ARE), these need no ground truth to locate at
    all, work equally well on a completely empty board (arguably better,
    since an empty premium square's own color/print is maximally
    unobstructed), and -- most importantly for scaling data -- only need
    clicking ONCE per venue/camera setup, not once per game, since a fixed
    camera sees these same 9 physical points in every frame it ever
    captures (**as long as the camera hasn't physically moved/reframed
    between the calibrated frame and the one you harvest** -- verified
    directly on a real Causeway broadcast that this doesn't always hold
    across an entire video; recalibrate whenever you switch to a
    genuinely different frame unless you've confirmed the framing is
    identical).

    `board`, if given, makes the click prompt honest when a target happens
    to be covered by an already-placed tile (common on an endgame board,
    where several premium squares are typically occupied by then) --
    shows the real sitting letter instead of claiming you'll see red/pink
    printing that a tile is actually covering. Omit it (the default) only
    when you genuinely have no ground truth yet, e.g. calibrating from a
    still-empty board before a game has even started."""
    targets = []
    for row, col in TRIPLE_WORD_SCORE_SQUARES:
        tile = board.get((row, col)) if board is not None else None
        label = (BLANK_LABEL if tile.is_blank else tile.letter) if tile else "TWS (red/pink)"
        targets.append(CalibrationTarget(row=row, col=col, label=label))
    center_tile = board.get(CENTER_SQUARE) if board is not None else None
    center_label = (BLANK_LABEL if center_tile.is_blank else center_tile.letter) if center_tile else "center star"
    targets.append(CalibrationTarget(row=CENTER_SQUARE[0], col=CENTER_SQUARE[1], label=center_label))
    return targets


def pick_calibration_targets(board: BoardState, count: int = 8) -> List[CalibrationTarget]:
    """Fallback for when `pick_fixed_board_targets` won't work -- e.g. every
    premium square happens to be covered by a tile in the only frame you
    have, or a venue's board doesn't use standard premium-square coloring.
    Greedy farthest-point sampling over the board's own occupied cells --
    spreads targets across the whole grid (corners, edges, and middle all
    represented) regardless of this specific board's layout, so it works
    for any game rather than assuming e.g. a clean row-0/row-14/col-0/col-14
    extreme exists. Starts from the occupied cell closest to the true
    board center, since that's the single most valuable calibration point
    (least sensitive to any residual lens/perspective distortion) -- matches
    the STATUS.md finding that center-anchored fits were consistently the
    most reliable part of every method tried.
    """
    occupied = list(board.occupied_cells())
    if not occupied:
        raise ValueError("board has no occupied cells to calibrate against")
    count = min(count, len(occupied))

    center = (7.0, 7.0)
    first = min(occupied, key=lambda rc: (rc[0] - center[0]) ** 2 + (rc[1] - center[1]) ** 2)
    chosen = [first]
    remaining = [rc for rc in occupied if rc != first]

    while len(chosen) < count and remaining:
        best_cell = max(
            remaining,
            key=lambda rc: min((rc[0] - c[0]) ** 2 + (rc[1] - c[1]) ** 2 for c in chosen),
        )
        chosen.append(best_cell)
        remaining.remove(best_cell)

    targets = []
    for row, col in chosen:
        tile = board.get((row, col))
        label = BLANK_LABEL if tile.is_blank else tile.letter
        targets.append(CalibrationTarget(row=row, col=col, label=label))
    return targets


def _decode_tile_byte(v: int) -> Tuple[str, bool]:
    is_blank = v > 128
    letter = chr(64 + (v - 128 if is_blank else v))
    return letter, is_blank


def board_from_woogles_document(doc_path: Path, through_event: Optional[int] = None) -> BoardState:
    """Decodes a woogles.io `GetGameDocument` response into a `BoardState`.

    By default (`through_event=None`) decodes the FINAL board (`board.tiles`,
    a base64 byte array, row-major, index = row*15+col: 0 empty, 1..26 A..Z,
    >128 a blank playing that letter -- verified against this project's own
    real Causeway Challenge 2026 game, reproducing known words VAMPLATES/
    TRISKELE/CHAEBOLS exactly).

    **The final board is usually a bad calibration/harvest choice**: a
    completely full board is visually cluttered for a human clicking
    targets, and tournament tiles are sometimes physically "squared up"
    after the game ends for the closing camera shot -- neither is
    representative of what the system sees mid-broadcast, and harvesting
    only from one maximally-dense endgame frame gives very homogeneous
    (not diverse) training data. Pass `through_event` (0-indexed, exclusive
    of that index -- i.e. `through_event=6` replays events 0-5) to
    reconstruct board state after an early, sparser move instead, by
    replaying each `TILE_PLACEMENT_MOVE` event's own `row`/`column`/
    `direction`/`played_tiles` fields. A `played_tiles` byte of 0 means
    that position is a hook through a tile an EARLIER move already placed
    (not a gap to fill) -- skip it, don't overwrite, but still advance
    position; verified this distinction matters on the real game (move 3's
    LENSED hooks through an existing tile mid-word, 6 position bytes but
    only 5 real new placements).
    """
    doc = json.loads(Path(doc_path).read_text())

    if through_event is None:
        raw = base64.b64decode(doc["board"]["tiles"])
        cells: Dict[Coord, Tile] = {}
        for i, v in enumerate(raw):
            if v == 0:
                continue
            row, col = divmod(i, 15)
            letter, is_blank = _decode_tile_byte(v)
            cells[(row, col)] = Tile(letter=letter, is_blank=is_blank)
        return BoardState(cells)

    cells = {}
    for event in doc["events"][:through_event]:
        if event["type"] != "TILE_PLACEMENT_MOVE":
            continue
        row, col = event["row"], event["column"]
        dr, dc = (0, 1) if event["direction"] == "HORIZONTAL" else (1, 0)
        for v in base64.b64decode(event["played_tiles"]):
            if v != 0:
                letter, is_blank = _decode_tile_byte(v)
                cells[(row, col)] = Tile(letter=letter, is_blank=is_blank)
            row += dr
            col += dc
    return BoardState(cells)


_ROTATE_CODES = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def load_frame(image_path: Path, rotate: int = 0) -> np.ndarray:
    """Loads a frame, optionally pre-rotated (0/90/180/270) -- some venues'
    raw camera footage sits upside-down or sideways relative to reading
    orientation (e.g. a camera mounted opposite the dealer), which makes
    letters awkward, not impossible, for a human to identify when
    clicking. Rotating up front is simpler than asking a human to read
    upside-down text. Must be applied identically at both the `targets`
    and `harvest` steps -- both go through this one function so that
    can't drift apart."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image at {image_path}")
    code = _ROTATE_CODES[rotate]
    return image if code is None else cv2.rotate(image, code)


def _image_to_base64_data_uri(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("could not encode image for embedding")
    return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}"


def generate_click_tool_html(image: np.ndarray, targets: List[CalibrationTarget], out_html: Path) -> None:
    """Self-contained (image embedded as base64, no server/relative-path
    dependency) HTML page: walks a human through clicking each target in
    order, one at a time, so there's never ambiguity about which click
    corresponds to which known cell. Reads native image pixels back via
    `naturalWidth`/`getBoundingClientRect`, so it's correct regardless of
    how the browser has scaled the displayed image -- no manual scale
    constant to get wrong (the exact class of arithmetic mistake this
    tool exists to eliminate).
    """
    data_uri = _image_to_base64_data_uri(image)
    targets_json = json.dumps(
        [{"row": t.row, "col": t.col, "label": t.label, "notation": t.notation} for t in targets]
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>calibration clicker</title>
<style>
  body {{ margin:0; background:#181818; color:#eee; font-family: system-ui, sans-serif; }}
  #bar {{ position:sticky; top:0; background:#000; padding:12px 16px; font-size:18px; z-index:10; }}
  #bar b {{ color:#7ee787; }}
  #wrap {{ position:relative; display:inline-block; }}
  img {{ display:block; max-width:none; }}
  .mark {{ position:absolute; width:14px; height:14px; border:2px solid #ff5555;
           border-radius:50%; margin-left:-7px; margin-top:-7px; pointer-events:none;
           box-shadow:0 0 4px #000; }}
  .mark span {{ position:absolute; left:16px; top:-6px; color:#ff5555; font-weight:bold;
                background:#000; padding:0 3px; border-radius:3px; white-space:nowrap; }}
  #done {{ padding:16px; }}
  textarea {{ width:90%; height:200px; font-family:monospace; font-size:13px; }}
  #rotate-btn {{ margin-left:12px; }}
  #img {{ transform-origin: center center; }}
</style></head>
<body>
<div id="bar">Click <b><span id="target-notation">?</span></b>
  (<span id="target-label">?</span>)
  (<span id="target-idx">0</span> / <span id="target-total">0</span>) &mdash;
  zoom with ctrl/cmd+scroll or your browser's native zoom before clicking for precision.
  <button id="rotate-btn" type="button">&#8635; Rotate view</button>
  <span style="color:#aaa">(if the board looks upside-down/sideways -- click before you start clicking targets)</span>
</div>
<div id="wrap"><img id="img" src="{data_uri}"></div>
<div id="done" style="display:none">
  <h3>All targets clicked. Copy this JSON for the harvest step:</h3>
  <textarea id="output" readonly></textarea>
</div>
<script>
const targets = {targets_json};
let idx = 0;
let viewRotation = 0;  // degrees, CSS-clockwise; purely visual, click math below inverts it
const results = [];
const img = document.getElementById('img');
const wrap = document.getElementById('wrap');

document.getElementById('rotate-btn').addEventListener('click', () => {{
  if (idx > 0) {{
    alert('Rotate the view before your first click, not partway through -- ' +
          'reload the page to start over if you need to change it now.');
    return;
  }}
  viewRotation = (viewRotation + 90) % 360;
  img.style.transform = 'rotate(' + viewRotation + 'deg)';
}});

function updateBar() {{
  if (idx >= targets.length) {{
    document.getElementById('bar').style.display = 'none';
    document.getElementById('done').style.display = 'block';
    document.getElementById('output').value = JSON.stringify(
      results.map((r,i) => ({{row: targets[i].row, col: targets[i].col, x: r[0], y: r[1]}})),
      null, 2
    );
    return;
  }}
  document.getElementById('target-notation').textContent = targets[idx].notation;
  document.getElementById('target-label').textContent = targets[idx].label;
  document.getElementById('target-idx').textContent = idx + 1;
  document.getElementById('target-total').textContent = targets.length;
}}

img.addEventListener('click', (e) => {{
  if (idx >= targets.length) return;
  const rect = img.getBoundingClientRect();
  // rect is the ROTATED element's on-screen box; its center is invariant
  // under rotation, so recover the click's offset from that center, then
  // invert the CSS rotation to get back to the image's own (unrotated)
  // layout coordinates, then scale by naturalWidth/offsetWidth -- offsetWidth
  // is the pre-transform CSS layout size, unaffected by the rotation, unlike
  // rect.width/height which swap for a 90/270 rotation.
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const dx = e.clientX - cx;
  const dy = e.clientY - cy;
  const rad = -viewRotation * Math.PI / 180;
  const localX = dx * Math.cos(rad) - dy * Math.sin(rad);
  const localY = dx * Math.sin(rad) + dy * Math.cos(rad);
  const layoutW = img.offsetWidth, layoutH = img.offsetHeight;
  const scaleX = img.naturalWidth / layoutW;
  const scaleY = img.naturalHeight / layoutH;
  const nativeX = (localX + layoutW / 2) * scaleX;
  const nativeY = (localY + layoutH / 2) * scaleY;
  results.push([nativeX, nativeY]);

  const mark = document.createElement('div');
  mark.className = 'mark';
  mark.style.left = (e.clientX - rect.left) + 'px';
  mark.style.top = (e.clientY - rect.top) + 'px';
  const lbl = document.createElement('span');
  lbl.textContent = (idx+1) + ':' + targets[idx].notation;
  mark.appendChild(lbl);
  wrap.appendChild(mark);

  idx += 1;
  updateBar();
}});

updateBar();
</script>
</body></html>
"""
    Path(out_html).write_text(html)


def fit_homography_from_clicks(
    clicks: List[Tuple[int, int, float, float]],
) -> Tuple[BoardCalibration, List[float]]:
    """`clicks` is (row, col, click_x, click_y) tuples. Uses RANSAC (not
    plain least-squares) specifically so one imprecise click among many
    doesn't drag the whole fit off -- the more targets clicked, the more
    robust this is. Returns the calibration plus each point's reprojection
    error in canonical pixels, so a suspiciously large one can be flagged
    and re-clicked rather than silently trusted.
    """
    if len(clicks) < 4:
        raise ValueError(f"need at least 4 clicks for a homography fit, got {len(clicks)}")

    src = np.array([[x, y] for _, _, x, y in clicks], dtype=np.float32)
    dst = np.array(
        [[col * CANONICAL_CELL_PX + CANONICAL_CELL_PX / 2, row * CANONICAL_CELL_PX + CANONICAL_CELL_PX / 2]
         for row, col, _, _ in clicks],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0)
    if homography is None:
        raise ValueError("could not compute a homography from the given clicks")

    errors = []
    for s, d in zip(src, dst):
        projected = homography @ np.array([s[0], s[1], 1.0])
        projected = projected[:2] / projected[2]
        errors.append(float(np.linalg.norm(projected - d)))

    return BoardCalibration(homography=homography), errors


def harvest_labeled_cells(
    image: np.ndarray, board: BoardState, calibration: BoardCalibration, out_dir: Path, prefix: str,
) -> int:
    rectified = calibration.rectify(image)
    # PNG (lossless), not JPEG -- the rectified image is read back by
    # harvest_board_cells to produce every training crop; JPEG's chroma
    # subsampling/quantization would bake compression artifacts into
    # every single one of them.
    rectified_path = Path(out_dir) / f"_rectified_{prefix}.png"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(rectified_path), rectified)
    return harvest_board_cells(rectified_path, board, Path(out_dir), prefix)


def build_spotcheck_montage(harvest_dir: Path, out_path: Path, cell_px: int = 100) -> None:
    """One crop per label, in a grid, so a human (or a future Claude
    session) can visually confirm alignment in a single glance before
    trusting any of it for training -- this project's batch3 corruption
    was only caught by doing exactly this, by hand, once; this makes it
    automatic and repeatable every time."""
    harvest_dir = Path(harvest_dir)
    by_label: Dict[str, Path] = {}
    for f in sorted(harvest_dir.glob("*.png")):
        label = f.name.split("_", 1)[0]
        by_label.setdefault(label, f)

    labels = sorted(by_label)
    if not labels:
        raise ValueError(f"no harvested crops found in {harvest_dir}")

    cols = 7
    rows = (len(labels) + cols - 1) // cols
    pad = 10
    canvas = np.full((rows * (cell_px + pad + 15), cols * (cell_px + pad), 3), 255, dtype=np.uint8)
    for i, label in enumerate(labels):
        r, c = divmod(i, cols)
        img = cv2.imread(str(by_label[label]))
        img = cv2.resize(img, (cell_px, cell_px))
        y0, x0 = r * (cell_px + pad + 15) + 20, c * (cell_px + pad) + pad // 2
        canvas[y0:y0 + cell_px, x0:x0 + cell_px] = img
        cv2.putText(canvas, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(out_path), canvas)


def _board_from_args(args: argparse.Namespace) -> Optional[BoardState]:
    if args.woogles_doc:
        return board_from_woogles_document(args.woogles_doc, through_event=args.woogles_through_event)
    if args.gcg:
        from training.collect.replay_game import read_gcg_moves, replay_gcg_game
        moves = read_gcg_moves(args.gcg)
        turns = replay_gcg_game(moves)
        return turns[args.move - 1].board_after
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    targets_cmd = sub.add_parser("targets", help="generate the HTML clicker tool")
    targets_cmd.add_argument("image", type=Path)
    targets_cmd.add_argument("--woogles-doc", type=Path, default=None)
    targets_cmd.add_argument("--woogles-through-event", type=int, default=None,
                              help="reconstruct board state after events[:N] instead of the final (often over-cluttered) board")
    targets_cmd.add_argument("--gcg", type=Path, default=None)
    targets_cmd.add_argument("--move", type=int, default=None)
    targets_cmd.add_argument("--use-occupied-cells", action="store_true",
                              help="target spread-out occupied cells (needs --woogles-doc/--gcg) instead of "
                                   "the default: the 8 fixed Triple Word Score squares + center, which need "
                                   "no ground truth at all and only need clicking once per venue/camera")
    targets_cmd.add_argument("--count", type=int, default=8, help="only used with --use-occupied-cells")
    targets_cmd.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                              help="pre-rotate the frame so letters read naturally -- must match --rotate at harvest time")
    targets_cmd.add_argument("--out", type=Path, required=True)

    harvest_cmd = sub.add_parser("harvest", help="fit homography from clicks and harvest labeled crops")
    harvest_cmd.add_argument("image", type=Path)
    harvest_cmd.add_argument("--woogles-doc", type=Path, default=None)
    harvest_cmd.add_argument("--woogles-through-event", type=int, default=None,
                              help="must match the value used at the targets step")
    harvest_cmd.add_argument("--gcg", type=Path, default=None)
    harvest_cmd.add_argument("--move", type=int, default=None)
    harvest_cmd.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                              help="must match --rotate used for the targets step -- clicks were made against that rotated image")
    harvest_cmd.add_argument("--clicks", type=Path, required=True)
    harvest_cmd.add_argument("--out-dir", type=Path, required=True)
    harvest_cmd.add_argument("--prefix", required=True)

    args = parser.parse_args()

    if args.command == "targets":
        board = _board_from_args(args)
        if args.use_occupied_cells:
            if board is None:
                raise SystemExit("--use-occupied-cells needs --woogles-doc or (--gcg and --move)")
            targets = pick_calibration_targets(board, count=args.count)
        else:
            # board may be None here -- fine, pick_fixed_board_targets falls
            # back to generic premium-square descriptions in that case.
            targets = pick_fixed_board_targets(board)
        image = load_frame(args.image, rotate=args.rotate)
        generate_click_tool_html(image, targets, args.out)
        print(f"wrote {args.out} with {len(targets)} targets:")
        for t in targets:
            print(f"  {t.notation} ({t.label})")
        return

    if args.command == "harvest":
        board = _board_from_args(args)
        if board is None:
            raise SystemExit("harvest needs ground truth: --woogles-doc or (--gcg and --move)")

        clicks_data = json.loads(Path(args.clicks).read_text())
        clicks = [(c["row"], c["col"], c["x"], c["y"]) for c in clicks_data]
        calibration, errors = fit_homography_from_clicks(clicks)
        print("reprojection errors (canonical px, out of a 60px cell):")
        for (row, col, _, _), err in zip(clicks, errors):
            flag = "  <-- check this one" if err > 15 else ""
            print(f"  ({row},{col}): {err:.1f}{flag}")

        image = load_frame(args.image, rotate=args.rotate)
        count = harvest_labeled_cells(image, board, calibration, args.out_dir, args.prefix)
        montage_path = Path(args.out_dir) / f"_spotcheck_{args.prefix}.jpg"
        build_spotcheck_montage(args.out_dir, montage_path)
        print(f"harvested {count} cells to {args.out_dir}")
        print(f"spot-check montage: {montage_path} -- inspect before adding to training data")


if __name__ == "__main__":
    main()
