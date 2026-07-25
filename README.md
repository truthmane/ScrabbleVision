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
  against synthetic multi-turn frame sequences. `autoscorer/perception/capture/`
  reads frames from a video file; `configs/venues/` persists per-venue
  camera calibration between sessions.
- **Not yet done**: running any of the above against real continuous
  video (every real-photo validation so far has been single-frame);
  cross-camera synchronization between a board camera and rack camera(s).
  See `game_watcher.py`'s module docstring for the exact scope line.
