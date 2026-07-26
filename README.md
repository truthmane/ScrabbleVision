# ScrabbleVision
An ML-powered auto-annotator for livestreamed Scrabble games

## Status

- **Manual-entry mode (Phases 1-2)**: done. Rules engine, tile pool,
  overlay, and operator UI all work today from typed-in moves, with no
  camera required -- see `autoscorer/api/`.
- **Perception components (Phases 3-5)**: built and validated
  individually against real broadcast photos -- board rectification,
  occupancy detection, a 27-class tile classifier (64.6% held-out,
  `docs/classifier-accuracy-plan.md`), an RF-DETR rack-tile detector
  (`training/detect/README.md`), temporal voting, constraint decoding,
  and a stillness/occlusion gate.
- **The state machine that sequences them into a system**
  (`autoscorer/gamelogic/movedetect/game_watcher.py`): built and tested
  against synthetic multi-turn frame sequences, **and now run against real
  continuous broadcast footage** via
  `python -m autoscorer.perception.capture.run_watcher` --
  `autoscorer/perception/capture/` reads frames from a video file;
  `configs/venues/` persists per-venue camera calibration (corners,
  motion/occupancy thresholds, an empty-board reference photo) between
  sessions instead of losing it at the end of every chat.
- **Real-footage result**: ran the full pipeline (capture -> stillness
  gate -> temporal-voted classification -> constraint decoding ->
  validation -> scoring -> publish gateway) against ~2 minutes of a real
  WESPA Word Wars broadcast, both mid-game and from an actual game start.
  **Zero wrong moves ever auto-published** across both runs, even while
  chasing down two real perception bugs along the way -- both now fixed:
  occupancy thresholds tuned for a different venue's graphics, and a
  sponsor-logo overlay on the center square that produced 61/61 false
  placement detections on the game-start clip. Root cause: the
  `gradient` occupancy signal is computed from the current cell crop
  alone (`cv2.Sobel`, no reference comparison), so a busy static graphic
  scores high purely from its own texture, indistinguishable from a real
  tile -- no reference-matching trick can fix a signal that never looks
  at the reference. Fixed by raising `occupancy_gradient_threshold` for
  this venue (relying on `diff`, which does compare to reference, and
  cleanly separates logo from tile) plus correcting the reference photo.
  Re-ran the exact clip that produced 61 false detections: **0/0** after
  the fix, with a small accepted trade-off on a separate mid-game clip
  (spurious cells 12->15, missed stayed 0). Full details in
  `configs/venues/wespa_word_wars.json`'s notes. Also added, and tested,
  multi-reference occupancy support (`detect_occupancy` can now check a
  cell against several empty-board photos, for a venue whose empty state
  genuinely has more than one appearance) -- not needed for the WESPA fix
  itself, but real, reusable infrastructure for the next venue that does.
  The confidence-gated safety net did exactly the job it was designed
  for throughout -- even while these bugs were live, nothing wrong ever
  reached the stream.
- **Wired into the actual product, not just a script.** `GameWatcher` now
  has a delegated mode (`session=`): every detected move flows through
  `GameSession.submit_move`, the exact call path a human operator's
  typed-in move already uses. The FastAPI app has a new `POST /watch`
  endpoint (`autoscorer/api/main.py`) that starts a video against the
  app's own live session in a background thread, broadcasting overlay
  updates as moves publish. **Verified live**: started the real server,
  pointed `/watch` at the real WESPA game-start clip, and watched the
  existing, unmodified operator UI (`/operator`) fill up with 61
  individually-reviewable pending moves in real time -- all correctly
  low-confidence, none wrongly auto-published, all approvable/rejectable
  through the same buttons a human operator already uses for manual
  entry. This is the connective tissue between "a validated CV pipeline"
  and "a product you can point a browser at."
- **GCG export** (`autoscorer/gamelogic/notation.py`'s `export_gcg`, served
  live at `GET /export/gcg`): renders the running `GameState`'s move
  history as GCG, the plain-text format tournament streaming graphics
  already consume -- a real integration point, not a new format invented
  for this project (the same module already *parsed* official `.gcg`
  files for WS3's ground-truth replay, so this is the reverse direction of
  something already validated). Covers PLAY/EXCHANGE/PASS lines with
  correct rack-before-move, position/direction, blank-as-lowercase-letter
  rendering, and running cumulative score per player; round-trips through
  the existing parser in tests. Deliberately does **not** produce the
  final end-of-game rack bonus/penalty line -- this engine has no
  end-of-game detection to trigger it, and real examples of that line
  don't reduce to one obvious formula (see `notation.py`'s module
  docstring), so it's left honestly absent rather than guessed at.
- **Ran against a complete real game end-to-end for the first time, and
  fixed the real bug it found.** Downloaded a full ~37-minute WESPA
  Word Wars game (Game 1 of the same broadcast) via `yt-dlp` and ran it
  through `run_watcher.py`. Two operational findings along the way, both
  resolved: the CLI's default confidence-gated mode never advances the
  board in an unattended run (real single-frame confidence essentially
  never clears 0.9, even after temporal voting -- voting fixes
  frame-to-frame noise, not the classifier's genuine per-crop
  uncertainty on a clean static tile), fixed by using `--mode autonomous`
  for offline validation; and the structural blank-tile check (correct
  and unconditional by design) has no automated resolution path in an
  unattended run, handled with a validation-only fallback to the
  classifier's best non-blank guess.
  With those out of the way, the run surfaced a real, previously-unknown
  architectural bug: **a settled read was committed the moment it was
  first seen**, so a player who placed part of a multi-tile word and
  then paused to think -- fully satisfying "hands not moving" while
  genuinely mid-turn -- got that partial placement committed as if it
  were the whole move. This is exactly what split one real move ("HUIA")
  into two turns attributed to two different players, and later caused
  the detector to permanently jam once enough such fragments piled up
  into a diff too scattered to validate as a legal placement.
  **Fixed** in `game_watcher.py`: a settled read is now only committed
  once the *identical* set of new-cell coordinates is confirmed by a
  second, independent settled observation -- a player still placing
  tiles will show a growing/different set on the next read and correctly
  keep waiting, while a genuinely finished turn re-confirms itself
  immediately. Covered by a new regression test that reproduces the
  exact partial-placement scenario. 227 tests passing (was 226).
  Re-ran the full game with the fix: turns 1-8 all matched the real GCG
  cleanly (HUIA came through as one complete move, both bingos scored
  exact matches), confirming the fix -- but the run stalled again,
  revealing a **second, distinct bug**: `read_new_cells_voted` decided
  occupancy from only the first frame of its temporal-voting window, so a
  hand hovering near (not over) the board -- too small a disturbance for
  the coarse whole-frame stillness gate to reject -- could nudge one
  frame's occupancy signal for a specific cell over threshold, and that
  one noisy frame got treated as ground truth for the whole window.
  Confirmed directly against the actual footage: every cell that flagged
  this way disagreed across the window's own 5 frames, while every
  genuine tile read occupied in all 5, every time. **Fixed** by requiring
  unanimous occupancy agreement across the whole window instead of
  trusting frame zero -- covered by a new regression test that reproduces
  a single noisy frame inside an otherwise-clean window. 228 tests
  passing.
- **Not yet done**: a third full-game run with both fixes hasn't been
  completed yet; cross-camera synchronization between a board camera and
  rack camera(s); batching classifier calls for real per-frame speed
  (currently ~5s/settled-frame on CPU, dominated by repeated
  single-image inference on occupied cells); the GCG end-of-game
  bonus/penalty line. See `game_watcher.py`'s and `run_watcher.py`'s
  module docstrings for the exact scope lines.
