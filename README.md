# ScrabbleVision

An ML-powered auto-annotator for livestreamed Scrabble games: watches a board camera, detects
plays, scores them with a real rules engine, and publishes to a stream overlay / operator review
UI — with a manual-entry mode that needs no camera at all.

## Status at a glance

- **Manual-entry mode**: complete. Type in moves, get scoring, tile-pool tracking, a stream
  overlay, and an operator review UI — no camera required.
- **Computer-vision pipeline**: complete and running end-to-end against real broadcast video
  (board rectification → occupancy detection → tile classification → constraint/lexicon decoding
  → the `GameWatcher` state machine → scoring → publish gateway → overlay/operator UI).
- **Tile classifier**: currently **93.4%** held-out accuracy on a venue-disjoint real-tile set
  (2,174 real training crops from 15+ distinct venues/tournaments). Full accuracy history and
  methodology: [`docs/classifier-accuracy-plan.md`](docs/classifier-accuracy-plan.md).
- **Not yet done**: cross-camera sync (board + rack cameras), cross-frame voting on rack reads,
  end-of-game GCG bonus/penalty line, PASS/EXCHANGE detection from the board camera alone
  (structurally needs a clock or rack camera — nothing changes on the board for these).

For the detailed build history, every measured accuracy number, and the debugging narrative
behind each fix, see [`docs/classifier-accuracy-plan.md`](docs/classifier-accuracy-plan.md) and
the module docstrings in `autoscorer/gamelogic/movedetect/game_watcher.py` and
`autoscorer/perception/capture/run_watcher.py`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+. Computer-vision features additionally need `opencv-python`, `torch`, and
`torchvision` (not in the base install — see `training/classify/model.py` and
`autoscorer/perception/` for what each piece needs).

## Usage

### Manual-entry mode (no camera)

Start the API server:

```bash
uvicorn autoscorer.api.main:app --reload
```

- `POST /moves` — submit a move
- `GET /pending` / `POST /pending/{turn_number}/decision` — operator review queue
- `GET /state` — current game state
- `GET /overlay`, `GET /overlay/field` — stream overlay pages
- `GET /operator` — operator review UI
- `GET /export/gcg` — export the game so far as a `.gcg` file
- `WS /ws/overlay` — live overlay updates

### Watching a real video

```bash
python -m autoscorer.perception.capture.run_watcher VIDEO.mp4 \
  --venue wespa_word_wars --player1 Alice --player2 Bob --mode autonomous
```

`--venue` names a profile under `configs/venues/<name>.json` (camera calibration, occupancy
thresholds, an empty-board reference photo — see that directory for existing venues and how to
add one). `POST /watch` on the API server does the same thing against a live session, streaming
updates to the overlay in real time.

### Measuring accuracy against a real game

```bash
python -m autoscorer.eval.run_game_eval VIDEO.mp4 --gcg GAME.gcg \
  --venue wespa_word_wars --player1 Alice --player2 Bob \
  --baseline tests/baselines/wespa_word_wars_game1.json
```

Reports turns detected, first divergence from the real GCG, cell/letter accuracy, stalls, and
more — see `autoscorer/eval/metrics.py`. Exits non-zero on a regression against the baseline.

### Training / extending the tile classifier

`training/classify/train.py` fine-tunes the classifier; `training/collect/click_calibrate.py` is
the tool for turning a new real board photo + its known GCG into labeled training crops (see
`training/collect/README.md` for the full collection workflow, and
`training/data/real_tiles/README.md` for what's already collected). `training/detect/README.md`
covers the separate rack-tile detector.

## Running tests

```bash
pytest
```

`tests/slow/` covers full-game scenarios with no external assets needed. A handful of tests need
a real, uncommitted video asset (gated behind an env var — see `pyproject.toml`'s markers) and are
skipped otherwise.

## Project layout

- `autoscorer/gamelogic/` — rules engine, board/tile/scoring models, the `GameWatcher` move-
  detection state machine, GCG parsing/export, the lexicon-constrained decoder.
- `autoscorer/perception/` — board calibration/rectification, occupancy detection, video capture.
- `autoscorer/api/` — FastAPI app (manual entry, `/watch`, overlay, operator UI).
- `autoscorer/eval/` — the accuracy-measurement harness (alignment, metrics, CLI).
- `training/` — classifier/detector training, synthetic data rendering, real-data collection
  tools.
- `configs/venues/` — per-venue camera calibration profiles.
- `models/` — saved checkpoints (see `models/README.md` for what each one is and how accurate).
