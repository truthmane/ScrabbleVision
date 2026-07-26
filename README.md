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
  Re-ran again with both fixes: got twice as far as any previous attempt
  (cleanly through turns 1-9, past both former blockers) before hitting a
  **third, narrower issue** -- one marginal/borderline-visibility cell
  kept flickering in and out of an otherwise-stable reading indefinitely,
  and the exact-set-equality confirmation check never fired while it was
  unsettled, even though every other cell in the placement was solid.
  **Fixed** by tracking confirmation per cell instead of on the whole
  reading as one atomic set: a cell only needs to survive two consecutive
  settled observations on its own, so a stable subset can commit without
  waiting forever on one cell that never fully stabilizes, while a
  genuinely new cell still has to reappear once before it's trusted.
  Covered by a new regression test (a stable cell + a cell that appears
  once then disappears, confirming the stable one still commits on its
  own). 229 tests passing (one pre-existing flaky test in the tiny-
  classifier-training suite unrelated to this change, confirmed by
  re-running clean 4/4 times).
  Re-ran a fourth time with a small refinement (only clear per-cell
  confirmation state on a genuine commit, not on any failed attempt, so
  a confirmed cell doesn't have to re-wait two full observations every
  time a transient neighbor spoils one retry) and got twice as far again
  (cleanly through turn 9) before hitting a **fourth, deeper issue**:
  `board_before` can genuinely have more than one turn's worth of
  unaccounted-for tiles new at once -- e.g. a slow multi-tile word (a
  real 7-letter bingo, "RAGBOLT") still being placed in one part of the
  board while an unrelated, already-stable cell elsewhere is also new.
  Both passed per-cell confirmation, but were treated as ONE combined
  placement, which can never validate as a single legal line since
  they're nowhere near each other -- so it failed the same way, forever,
  every retry. **Fixed** by clustering confirmed cells by orthogonal
  adjacency before validating -- a word's tiles are always contiguous, so
  two unrelated cells that happen to stabilize at the same time now form
  two separate candidate clusters, and only one is acted on per turn (the
  rest stay exactly as confirmed as they are and get picked up, against
  an updated board, on a later call). Covered by a new regression test:
  two disconnected, independently-legal single-cell placements confirmed
  in the same observation, verifying they commit as two separate turns
  rather than one failed, merged one. 230 tests passing.
  Re-ran a fifth time and the clustering fix itself had a bug, caught
  immediately: it only grouped new cells adjacent to each OTHER, so a
  real word hooking through an existing tile in the middle (the actual
  WESPA move "ARB.RIZE" -- the "." is a pre-existing letter) got split
  back into two separate turns instead of the one 7-cell bingo it really
  was. **Fixed**: clustering now bridges through a run of already-
  occupied cells (old or new), not just other new cells -- the same
  notion of "connected" `validate_placement`/`word_resolver` already use.
  New regression test: new cells on both sides of an existing middle
  tile now correctly cluster as one. 231 tests passing (the same known
  flaky test above, unrelated).
  Re-ran a sixth time with all five fixes: cleanly through turns 1-11,
  including two adjacent single-cell turns that would previously have
  merged -- real, direct proof the clustering-by-adjacency design works
  correctly on genuine, non-bridged cases too, not just the bridged one.
  Then hit a **sixth issue, a different category entirely**: a screen
  region (row 0, just above where "RAGBOLT" was being placed) developed
  a borderline occupancy diff (38-44, against a 38.0 threshold) even
  though it was visually confirmed empty. Tested the "just use a fresher
  reference photo" fix directly: a reference from mid-game improved the
  earlier region but made a *different* region worse near the end of the
  same game (diff climbed to 46) -- proof that lighting drifts
  *continuously* over a 30+ minute broadcast, so no single fixed
  reference point can cover a whole game. **Fixed properly**: `GameWatcher`
  now continuously refreshes the reference's still-unplayed cells from
  real frames as the game progresses (gated on the cell being BOTH
  unplayed in `board_before` AND not currently reading as occupied, so a
  not-yet-committed real tile can never get silently baked into the
  background). New regression test simulates two increments of gradual
  drift and confirms the second increment -- big enough to spuriously
  cross the threshold against the original reference -- reads correctly
  as still-empty once the first increment has been absorbed. 232 tests
  passing.
- **Not yet done**: cross-camera synchronization between a board
  camera and rack camera(s); the GCG end-of-game bonus/penalty line.
  See `game_watcher.py`'s and `run_watcher.py`'s module docstrings for
  the exact scope lines.
- **Classifier calls batched.** `read_new_cells_voted` previously called
  the classifier once per (cell, frame) pair -- for a 7-cell turn over a
  5-frame settled window, 35 separate forward passes. Added
  `TileClassifierModel.predict_topk_batch()`, which stacks every crop
  into one tensor for a single forward pass; this is mathematically
  identical to the one-at-a-time calls (the model runs in `eval()` mode,
  so BatchNorm uses its stored running statistics rather than the
  current batch's -- no image's result can depend on what else shares
  its batch). `board_reader.py` now collects every (cell, frame) crop
  across the whole window before calling the classifier once, instead of
  inside the per-cell loop. New regression test proves batched output is
  numerically identical to individual `predict_topk` calls (`abs=1e-5`),
  plus an empty-input test. Measured directly against a real 7-cell turn
  from the Game 1 broadcast: 2.81s total (down from the ~5s/frame,
  per-crop-call baseline this was written to fix). 234 tests passing.
- **First full end-to-end completion of a complete real game.** Re-ran
  with all six fixes (adding the blended-reference refinement above) and
  the pipeline processed the entire ~37-minute broadcast start to finish
  -- no hang, no crash, a first for this project. Comparing the 11
  detected turns against the real GCG: turns 1, 2, 5, 6, 9 (QAT, HUIA,
  CROWNET, ARB.RIZE, WYND) scored an exact match; turns 3, 4, 7, 8
  (FISCS, ..OJO, INENTED, .YGA) got the right position and cell count
  with a score off by a few points (one misread letter somewhere); turns
  10-11 broke down (RAGBOLT came back with only 6 of 7 cells) and most
  of the remaining ~16 real moves in the game weren't cleanly captured.
  **This validates all six architectural fixes above -- the pipeline
  reliably handles a real game's first ~9 moves with strong accuracy.**
  What remains past that point is a different category of problem: a
  classifier/occupancy accuracy ceiling, not turn-detection logic.
  Root-caused the specific RAGBOLT miss rather than just noting it:
  pulled the exact frame and measured occupancy directly -- the dropped
  cell (RAGBOLT's "T") scored diff=38.96 against the venue's 38.0
  threshold, while its neighboring letters scored 52-70. Thin-glyph
  letters like T carry less ink and so naturally score lower diff,
  leaving little margin before ordinary noise pushes a real tile below
  threshold. Deliberately did not chase this with a threshold change:
  the residual noise measured earlier tonight overlaps the same 30s-40s
  band, so lowering the threshold to rescue thin letters would likely
  resurrect the false-positive problems already fixed this session --
  a genuine calibration ceiling for this broadcast, not a quick fix.
