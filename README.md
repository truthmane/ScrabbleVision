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
- **Not yet done**: this hasn't run against a *complete* real game
  end-to-end (only short clips so far, now the next planned validation
  step); cross-camera synchronization between a board camera and rack
  camera(s); batching classifier calls for real per-frame speed (currently
  ~5s/settled-frame on CPU, dominated by repeated single-image inference
  on occupied cells); the GCG end-of-game bonus/penalty line above. See
  `game_watcher.py`'s and `run_watcher.py`'s module docstrings for the
  exact scope lines.
