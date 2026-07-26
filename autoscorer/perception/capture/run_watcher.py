"""The runnable entry point this project has been missing: point it at a
video file and a saved venue profile, and it drives `GameWatcher` frame by
frame via `VideoFrameSource`, printing every move it detects. Everything
this calls (capture, calibration, stillness, classification, constraint
decoding, scoring, the publish gateway) was previously only ever exercised
from a script or a unit test -- this is the first place they run together
against a continuous stream, not a hand-picked single frame.

**Board-camera only.** This CLI has no concept of a paired rack-camera
video source -- `GameWatcher.record_rack` exists and works (see its own
docstring for scope), but wiring a second synchronized video stream
through this runner is real, separate, not-yet-done work. Board-only
tracking already gets you PLAY-move detection, scoring, and the full
publish-gateway safety net; it just can't classify EXCHANGE/PASS turns or
tighten constraint decoding with real rack contents.

Turn alternation is simple and deliberate: `player1_id` moves first, and
the two alternate after every PLAY-carrying event -- whether it
auto-published or is only pending operator review, since either way a
real move was genuinely detected on the board (alternating only on
auto-published moves would desync every subsequent player attribution
after the very first move that needed a human, in any mode other than
pure AUTONOMOUS). This is exactly as much "whose turn is it" logic as
vision alone can ever provide (see game_watcher.py's own docstring on why
PASS detection is out of scope) -- a real deployment with a game clock
could drive this more precisely, but strict alternation is the correct
assumption for any game with no passes or challenges, and this project
has no clock signal to do better with.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional

from autoscorer.api.session import GameSession
from autoscorer.gamelogic.movedetect.game_watcher import GameWatcher, WatcherEvent
from autoscorer.gamelogic.models import MoveType
from autoscorer.gamelogic.notation import format_square
from autoscorer.gamelogic.publish import PublishGateway, PublishMode
from autoscorer.perception.calibration.venue_profile import load_venue_profile
from autoscorer.perception.capture.video_source import VideoFrameSource
from training.classify.infer import TileClassifierModel

PUBLISH_MODES = {
    "autonomous": PublishMode.AUTONOMOUS,
    "manual": PublishMode.MANUAL,
    "confidence_fallback": PublishMode.AUTONOMOUS_WITH_CONFIDENCE_FALLBACK,
}


def format_event(event: WatcherEvent, player_id: Optional[str]) -> Optional[str]:
    """A human-readable one-line summary, or None for a no-op event (still
    settling, or settled with nothing new -- most calls, and not worth
    printing one line per sampled frame for). A stall watchdog event has
    no `scored_move` (nothing ever committed) but must still be printed --
    it's precisely the kind of event a jammed watcher previously produced
    silently, forever, with nothing in this CLI's output to show for it."""
    if event.is_stall:
        squares = ", ".join(sorted(format_square(c) for c in event.attempted_cells))
        return f"[STALLED] {player_id}: {event.reason} squares=[{squares}]"
    if event.scored_move is None:
        return None

    candidate = event.scored_move.candidate
    move_type = candidate.move_type.value
    status = "NEEDS OPERATOR" if event.needs_operator else "AUTO-PUBLISHED"
    conf = f"conf={event.confidence:.2f}"

    if move_type == MoveType.PLAY.value and event.scored_move.move_score is not None:
        score = event.scored_move.move_score.total
        cells = candidate.new_cells
        return f"[{status}] {player_id}: PLAY {cells} score={score} {conf} (turn {candidate.turn_number})"
    return f"[{status}] {player_id}: {move_type} {conf} (turn {candidate.turn_number})"


def run_watcher_on_video(
    video_path: Path,
    venue_name: str,
    classifier_path: Path,
    player1_id: str,
    player2_id: str,
    sample_fps: Optional[float] = 2.0,
    mode: PublishMode = PublishMode.AUTONOMOUS_WITH_CONFIDENCE_FALLBACK,
    confidence_threshold: float = 0.9,
    max_frames: Optional[int] = None,
    device: str = "cpu",
    session: Optional[GameSession] = None,
    on_event=None,
    on_frame_event=None,
) -> List[WatcherEvent]:
    """Runs `GameWatcher` against every sampled frame of `video_path`,
    alternating `player1_id`/`player2_id` after each PLAY-carrying event
    (whether it auto-published or is only pending operator review --
    either way a real move was detected on the board, and the *next*
    frame belongs to the other player regardless of when the pending one
    gets approved). Returns every event that carried a detected move
    (empty settle/no-op events are filtered out -- callers wanting the
    raw per-frame trace should pass `on_frame_event` instead of relying
    on the return value).

    Pass `session` (an existing `GameSession`, e.g. the FastAPI app's live
    one) to run in delegated mode -- every detected move then flows
    through the same `submit_move` path an operator's typed-in move
    already uses, landing in that session's real `/pending` list and
    history, not a throwaway one. `mode`/`confidence_threshold` are
    ignored in that case (the session's own gateway already has them).

    `on_event`, if given, is called with every move-carrying event as it
    happens -- the hook `main.py`'s `/watch` endpoint uses to broadcast an
    overlay update the moment a move actually publishes, without this
    function needing to know anything about WebSockets. Do not repurpose
    this for anything that needs to see EVERY frame's event (including
    settle/no-op/stall ones) -- `/watch`'s semantics depend on only ever
    seeing move-carrying events here. Use `on_frame_event(frame_index,
    event)` for that instead: it fires for every single event, before the
    move-carrying filter, which is what a stall/health report needs (a
    permanently jammed watcher previously produced nothing at all for
    `on_event` to see).
    """
    profile = load_venue_profile(venue_name)
    classifier = TileClassifierModel(classifier_path, device=device)

    watcher_kwargs = dict(
        calibration=profile.calibration(),
        reference_board=profile.load_reference_board(),
        classifier=classifier,
        motion_threshold=profile.motion_threshold,
        still_frame_count=profile.still_frame_count,
        occupancy_diff_threshold=profile.occupancy_diff_threshold,
        occupancy_gradient_threshold=profile.occupancy_gradient_threshold,
    )
    if session is not None:
        watcher = GameWatcher(session=session, **watcher_kwargs)
    else:
        gateway = PublishGateway(mode=mode, confidence_threshold=confidence_threshold)
        watcher = GameWatcher(publish_gateway=gateway, **watcher_kwargs)

    current_player, other_player = player1_id, player2_id
    events: List[WatcherEvent] = []

    with VideoFrameSource(video_path, sample_fps=sample_fps) as source:
        for i, frame in enumerate(source):
            if max_frames is not None and i >= max_frames:
                break
            event = watcher.observe_board_frame(frame, player_id=current_player)

            if on_frame_event is not None:
                on_frame_event(i, event)

            if event.is_stall:
                line = format_event(event, current_player)
                if line:
                    print(line)

            if event.scored_move is None:
                continue

            events.append(event)
            line = format_event(event, current_player)
            if line:
                print(line)
            if on_event is not None:
                on_event(event)

            if event.scored_move.candidate.move_type == MoveType.PLAY:
                current_player, other_player = other_player, current_player

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--venue", required=True, help="Venue profile name (configs/venues/<name>.json)")
    parser.add_argument("--classifier", type=Path, default=Path("models/tile_classifier_v1.pt"))
    parser.add_argument("--player1", required=True)
    parser.add_argument("--player2", required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--mode", choices=list(PUBLISH_MODES), default="confidence_fallback")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    t0 = time.time()
    events = run_watcher_on_video(
        args.video_path, args.venue, args.classifier, args.player1, args.player2,
        sample_fps=args.sample_fps, mode=PUBLISH_MODES[args.mode],
        confidence_threshold=args.confidence_threshold, max_frames=args.max_frames, device=args.device,
    )
    elapsed = time.time() - t0

    auto_published = sum(1 for e in events if not e.needs_operator)
    routed = sum(1 for e in events if e.needs_operator)
    print(f"\n{len(events)} moves detected in {elapsed:.1f}s: {auto_published} auto-published, {routed} routed to operator")


if __name__ == "__main__":
    main()
