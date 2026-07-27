# ScrabbleVision
An ML-powered auto-annotator for livestreamed Scrabble games

## Status

- **Manual-entry mode (Phases 1-2)**: done. Rules engine, tile pool,
  overlay, and operator UI all work today from typed-in moves, with no
  camera required -- see `autoscorer/api/`.
- **Perception components (Phases 3-5)**: built and validated
  individually against real broadcast photos -- board rectification,
  occupancy detection, a 27-class tile classifier (**~74-76% held-out on
  trustworthy ground truth** -- the long-reported 64.6% figure turned out
  to be deflated ~10-12 points by a corrupted harvest batch in the
  held-out set itself, found and corrected; see
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
  camera and rack camera(s); cross-frame temporal voting on rack reads
  (the rack path now has a stillness gate, see below, but no voting --
  matching detected tiles across frames first would be needed, since a
  rack has no fixed grid to vote per-position against like board cells
  do); the GCG end-of-game bonus/penalty line. See `game_watcher.py`'s
  and `run_watcher.py`'s module docstrings for the exact scope lines.
- **Rack-specific stillness gate.** `record_rack` used to read whatever
  single frame was handed to it, with no check that the rack was
  actually settled -- a hand still moving tiles around mid-exchange
  could get read as if it were a finished rack. Added a per-player
  rolling frame buffer gated through the same `stable_window` check the
  board path already uses (`still_frame_count` consecutive low-motion
  frames required before a read happens); `record_rack` returns `None`
  while still waiting, same as it already did for "no change," so
  callers don't see a new contract, just a real read attempted less
  often and only once actually settled. Cross-frame *voting* on rack
  reads (like the board path's `read_new_cells_voted`) still isn't
  built -- would need to match detected tiles across frames first,
  since a rack has no fixed grid to vote per-position against. New
  tests: settling now requires repeated calls (mirrors the board path's
  own `_settle` test helper), and a rack alternating between a clean
  frame and one with a large hand-sized occlusion never settles across
  6 calls. 235 tests passing (was 234).
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
  **Confirmed end-to-end**: re-ran the entire ~37-minute Game 1
  broadcast with this change and got byte-for-byte identical detection
  output (same 11 events, same final 52-cell board, same scores
  376/252) in **1293.5s, down from the pre-batching run's 2059.0s** --
  a real 37% wall-clock reduction on the exact same real video, not
  just a synthetic microbenchmark.
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
- **The "classifier/occupancy accuracy ceiling" diagnosis above was
  wrong, and an audit corrected it with a measurement, not a guess.**
  Requested an accuracy/full-game audit; instead of trusting the prior
  session's own conclusion, re-ran the real Game 1 log and counted
  failure reasons directly: **163 consecutive settled observations, every
  single one rejected with the identical reason**,
  `'new tiles must lie in a single row or column'`. Not a letter-accuracy
  problem at all -- three real logic defects in `GameWatcher`'s commit
  path: (1) `_cluster_cells` groups new cells by connectivity, not
  collinearity, so an L-shaped/staircase group of adjacent cells clusters
  as one un-validatable blob; (2) the watcher tried only the *first*
  confirmed cluster per observation and gave up entirely on failure,
  so the same bad blob got re-picked forever; (3) nothing ever expired a
  persistently-bad cell -- the adaptive reference's healing gate
  (`not occupancy[coord]`) structurally excludes any cell that has
  already crossed the occupancy threshold, so a false-positive cell can
  never recover on its own. A fourth, independent jam existed on
  blanks: the blank-detected path returned before clearing confirmation,
  so a cluster containing a blank (Game 1 move 19, `PIOTE?? 14B
  PInOTa.E`, two blanks) re-triggered identically forever regardless of
  classifier accuracy -- reproduced in a **36-second synthetic test with
  a perfect (p=1.0) classifier**, proving it was never about letter
  accuracy at all.
- **Fixed with a real search, not a threshold tweak.** New module
  `autoscorer/gamelogic/movedetect/placement_search.py`:
  `enumerate_candidate_placements` makes every candidate collinear and
  contiguous-through-the-board *by construction* (splitting a cluster
  into maximal per-row/per-column runs, still bridging through existing
  tiles for hooked words), so the two dominant rejection reasons become
  structurally unreachable. `GameWatcher`'s commit loop now tries every
  ranked candidate (largest first) instead of one, and a cell that fails
  `FAILURE_QUARANTINE_THRESHOLD` (3) times is quarantined -- excluded
  from consideration -- for `QUARANTINE_TTL_OBSERVATIONS` (20)
  observations, which is what stops one bad cell from blocking every
  other placement on the board. A blank failure is attributed only to
  the actual blank cell(s), not the whole candidate -- found by testing:
  without this, a 7-cell word with one blank in it would quarantine its
  five perfectly good letters right alongside the blank. A candidate
  smaller than the largest one its cluster could have produced (a
  truncation) never auto-publishes, regardless of confidence or publish
  mode -- measured risk: dropping one cell from each of 302 real plays
  across 13 fixtures still passes pure geometry 39% of the time, so a
  truncated word could otherwise silently commit a wrong score. A
  `STALL_OBSERVATIONS` (30) watchdog is a backstop beyond per-cell
  quarantine, force-quarantining every confirmed cell with any failure
  history if nothing commits for that long.
- **Built the measurement harness this whole audit exposed was
  missing** (`autoscorer/eval/`): before this, every accuracy claim in
  this README was established by hand-reading a log. `gcg_truth.py`
  reuses the already-validated GCG replay machinery for ground truth;
  `alignment.py` is a monotone sequence alignment (sequences can skip,
  merge, or split turns, so this isn't a zip) over cell-set Jaccard
  similarity; `metrics.py` builds a `GameEvalReport` with everything
  reported separately (turns detected, first divergence, cell F1, letter
  accuracy, exact score matches, stalls, provenance) rather than
  collapsed into one number; `run_game_eval.py` is the CLI that runs the
  real pipeline against a video and a `.gcg` and checks for regressions
  against a committed baseline JSON -- the video itself can never be
  committed (copyright), but the baseline now is, so today's numbers are
  a contract future work has to improve against, not a one-off printout.
  `tests/slow/test_synthetic_full_game.py` reproduces the whole thing
  without any video or GPU: a `NoisyOracleClassifier`
  (`tests/support/noisy_oracle.py`) nearest-neighbor identifies synthetic
  crops against a registry of everything it actually rendered, then
  injects controlled per-cell error at a chosen accuracy -- at `p=1.0`
  the pipeline must complete all 21 real moves with zero jam; at `p=0.72`
  (the deployed checkpoint's real measured accuracy) it must never get
  permanently stuck, even though it won't get every turn right.
- **Measured, honest before/after on the real Game 1 broadcast**:
  `longest_stall` **86 → 3** settled observations (the actual jam is
  fixed, not just shortened); `detected_turns` **11 → 24**; turns cleanly
  matched to the real GCG **11 → 15**; `exact_score_matches` **5 → 6**;
  final board cells correct **39 → 50 of 95**; wall clock **1349s → 384s**
  (3.5x faster, almost entirely because the pipeline no longer burns
  hundreds of observations retrying one permanently-stuck blob). The one
  metric that got honestly worse: `cell_f1_micro` **0.981 → 0.935** --
  expected and accepted, not a regression to chase: the old pipeline
  never got far enough into the game to be wrong about the harder later
  content, so its cell-level average was measured over an easier, shorter
  slice. New stalls are shorter (2-3 observations, not 163) and a
  different, more benign reason: `'placement is not connected to any
  existing tile'` -- real occupancy/classification uncertainty in a
  harder part of the board, not a logic bug. Remaining gap (missed/split
  turns, wrong letters) is squarely WS2 (lexicon-constrained decoding)
  and WS3 (occupancy signal) territory, not more turn-detection logic.
  **275 tests passing** (was 235; +40 new: alignment, metrics, notation,
  placement_search, the synthetic full-game test, and new GameWatcher
  regression tests for candidate fallback, quarantine, truncation, the
  stall watchdog, and phony-word safety), one pre-existing flaky test
  documented as before, unrelated to any of this.
- **Lexicon-constrained decoding (WS2).** `autoscorer/gamelogic/dictionary/lexicon.py`
  loads a word list -- vendored default (ENABLE, public domain, 168,551
  words after filtering) or a licensed override (`configs/lexicons/`,
  gitignored, resolved via name/path/env var) -- as a plain `frozenset`
  (a DAWG/GADDAG buys nothing here: the only queries are exact membership
  and a 26-way blank-letter check, never rack-based move generation).
  **Phonies are legal, scoring plays**, so the lexicon is a pure
  re-ranker, never a filter: `autoscorer/gamelogic/movedetect/lexicon_decoder.py`'s
  `decode_with_lexicon` is a position-ordered beam search over each
  cell's full temporal-voted distribution, scored by pool feasibility
  (reusing `constraint_decoder`'s machinery) plus how many of the words
  the reading forms are real -- it can rank a phony above an in-lexicon
  alternative, never substitute one, and always returns a reading (never
  invents one, never drops a cell). `GameWatcher` gained an optional
  `lexicon` parameter, wired through `run_watcher_on_video`/
  `run_game_eval.py`'s `--lexicon` and `VenueProfile.lexicon`.

  Caught two real bugs by testing, not by reasoning about the design in
  the abstract: (1) beam truncation at each cell originally used only
  the cross-word penalty accumulated so far, computing the *main* word's
  lexicon check in a separate pass after the whole beam already finished
  -- meaning the correct answer could get pruned before the score that
  would have favored it was ever computed (a needed letter, ranked
  alphabetically outside the beam width, silently vanished). (2) The
  blank branch scored "maybe this is a blank" as a neutral 0 log-probability,
  which is *always* better than any real, honestly-confident letter guess
  (`log(confidence)` is always negative below 1.0) -- a single, cleanly
  classified tile was losing every time to a low-confidence "it might be
  a blank" interpretation. Both are now regression-tested directly.

  **Measured, not guessed** (E1-E5, zero pixels, all 13 real GCG
  fixtures, 302 real plays): **E1 coverage** -- 471 distinct words
  formed; the vendored ENABLE default covers 79.4% (374/471), the real
  licensed CSW24 list covers 99.8% (470/471, missing only `DORMENT`).
  **E2 simulated-noise recovery** (corrupting each true label at the
  deployed checkpoint's real confusion shape, p=0.72): cell accuracy is
  roughly flat with or without a lexicon (73.3% raw either way), but
  **per-move exact match improves substantially** -- 89/302 raw →
  96/302 with ENABLE → **100/302 with CSW24** -- because a lexicon's
  value shows up at the word level (every cell has to be simultaneously
  right for a real word to form), not the per-cell average. **E3 blank
  recovery**: pool-feasibility alone recovers 4/26 real blank letters
  (15.4%); adding CSW24 recovers **13/26 (50.0%)**, with 2/26 genuinely
  ambiguous (more than one letter keeps every word valid -- irreducibly
  operator work). **E4 phony safety**: 0 violations across 3 corrupted
  trials -- confirmed the lexicon never turns an already-correct
  *evidenced* letter reading into a wrong one (blanks are excluded from
  this specific check on purpose: a blank's true letter carries zero
  raw evidence to begin with, so "correctly identified as blank" isn't
  the same claim as "the letter guess was already right" -- that's what
  E3 measures honestly, not this one). **E5 truncation/contamination**,
  now against the real lexicon instead of a generic-dictionary stand-in:
  of 1253 single-cell drops from real plays, 487 (38.9%) still pass pure
  geometry, and 174 of those (13.9% of all drops) are also lexicon-valid
  -- the lexicon filters roughly 6 in 7 of the geometrically-passable
  truncations. Of 444 spurious one-cell-extended supersets, all 444
  (100%) pass geometry, but only 13 (2.9%) are also lexicon-valid --
  tighter than an earlier generic-list estimate (~11%), since a real
  tournament dictionary, despite being broad, still rejects almost any
  random letter appended to a real word.
- **Confirmed against the real Game 1 broadcast, with the real CSW24
  lexicon wired in (`--lexicon csw24`), not just synthetic corruption.**
  Re-ran the same real video against the post-WS1 baseline:
  `first_divergence_index` **3 → 7** (four more turns now match the real
  GCG cleanly before anything diverges); `exact_score_matches` **6 → 9**;
  `letter_accuracy` on correctly-located cells **72.4% → 87.3%**
  (46/58 → 48/55); final board cells correct **50 → 56 of 95**;
  `cell_f1_micro` **0.935 → 0.940** (no regression this time -- the
  earlier accepted dip from WS1 alone has now partly reversed).
  `longest_stall` stays at 3 (unaffected, as expected -- the lexicon
  changes letter/word decoding, not turn detection). The CLI's own
  regression check reports **no regressions vs. the WS1 baseline**.
  Baseline JSON updated to this new, better state.
- **WS3 occupancy signal work -- items 1-2 landed and are a clean win,
  after a self-inflicted evaluation bug wasted most of an evening
  chasing a phantom regression.** `read_new_cells_voted`'s hard
  unanimous-AND (every frame in a settled window must agree a cell is
  occupied) is now a two-tier system: unanimous stays HARD; a cell
  agreed on by all but one frame, *and* orthogonally adjacent to a HARD
  cell, is SOFT -- usable only as an optional in-line extension of an
  already-confirmed HARD run, never as a placement anchor on its own
  (targets a real measured case: WESPA "RAGBOLT"'s final T scored diff
  38.96 against a 38.0 threshold, a bare miss in an otherwise clean
  window). Also added `crop_cell_inset` -- a 5% occupancy-only inset
  crop trimming grid-line/neighbor-bleed noise at each cell's edge,
  deliberately never touching `crop_cell` itself (the classifier's
  trained input distribution) -- after a naive 15% inset was measured to
  erase a blank tile's entire detectable signal (no glyph means no
  interior texture; its only signal is its own edge).

  **The real bug of this whole session: `run_game_eval`'s CLI defaulted
  to `sample_fps=2.0`, but every prior baseline run (and the WESPA
  profile's own stillness-gate calibration, `still_frame_count=5`) used
  `0.2`.** At 5 frames, that's 25 seconds of real stillness at the
  right rate, but only 2.5 seconds at the wrong one -- fast enough that
  a player's pause between individual tile placements reads as
  "settled," so a single already-placed tile satisfies the
  two-observation confirmation rule and multi-tile words fragment. This
  was invisible because `Provenance` never recorded the sampling rate,
  and it produced a very convincing-looking regression (divergence and
  exact-match numbers dropping, cell F1 dropping) that consumed several
  hours before being traced to the actual cause by comparing stall-frame
  indices between old and new logs.

  **Re-running current code at the correct 0.2 fps settled it
  cleanly**: `first_divergence_index` **7** (matches the old baseline
  exactly) and `exact_score_matches` **9** (also exact), with
  `cell_f1_micro` **1.000** (up from the old baseline's 0.940) and final
  board cells correct **70/95** (up from 56/95). WS3 items 1-2 are a
  real improvement on every axis measured, once measured under the
  conditions the pipeline was actually calibrated for.

  **Fixed so this class of bug can't recur silently**: `Provenance` now
  records `sample_fps` and `publish_mode`; `check_for_regressions`
  refuses to compare two reports captured at different rates instead of
  reporting a phantom pass/fail; the CLI's `--sample-fps` default is now
  `0.2` to match every calibration run for this venue. Also caught and
  fixed a second, smaller bug while wiring this in:
  `GameEvalReport.to_json_dict()` enumerated provenance fields by hand
  and had silently dropped the new ones on the first attempt to add
  them. The committed baseline is regenerated from a run with complete,
  honest provenance. **307 tests passing.**
