"""FastAPI backend: REST endpoints for manual move entry and operator
decisions, a WebSocket that pushes OverlayState to the stream overlay on
every applied move, and the two static pages (overlay + operator UI).

Single-process, single in-memory GameSession for now -- this is the Phase 2
"manual mode" product described in the plan: a human operator drives the
whole game, with correct scoring, exact pool tracking, and a working stream
overlay, entirely without a camera.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autoscorer.api.session import GameSession
from autoscorer.gamelogic.board import BOARD_SIZE, CENTER, PREMIUM_SQUARES
from autoscorer.gamelogic.models import MoveProcessingError, ScoredMove
from autoscorer.gamelogic.notation import export_gcg, parse_position
from autoscorer.gamelogic.publish import PublishMode
from autoscorer.overlay.state import build_overlay_state

app = FastAPI(title="AutoScorer")
session = GameSession()

_STATIC_DIR = Path(__file__).resolve().parent.parent


class ConnectionManager:
    """Tracks connected overlay WebSocket clients and broadcasts state to
    all of them. Kept separate from GameSession so the game logic stays
    ASGI-free and independently testable.
    """

    def __init__(self) -> None:
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, payload: dict) -> None:
        stale: List[WebSocket] = []
        for ws in self.active:
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


overlay_connections = ConnectionManager()


# --- request/response models -------------------------------------------------

class TilePlacement(BaseModel):
    row: int
    col: int
    letter: str
    is_blank: bool = False


class RackTile(BaseModel):
    letter: Optional[str] = None
    is_blank: bool = False


class MoveSubmission(BaseModel):
    player_id: str
    new_tiles: List[TilePlacement] = []
    rack_after: Optional[List[RackTile]] = None
    confidence: float = 1.0


class NotationMoveSubmission(BaseModel):
    player_id: str
    line: str  # e.g. "A1 NO" or "14B PInOTa.E"
    confidence: float = 1.0


class OperatorDecisionRequest(BaseModel):
    action: str  # "approve" | "reject"


class ModeChangeRequest(BaseModel):
    mode: PublishMode


class WatchRequest(BaseModel):
    video_path: str
    venue: str
    player1: str
    player2: str
    sample_fps: Optional[float] = 2.0
    classifier_path: str = "models/tile_classifier_v1.pt"
    max_frames: Optional[int] = None


class WatchStatus:
    """Tracks the single background video-watching run this process
    supports at a time -- one `GameSession` already means one live game,
    so one concurrent watcher is the right constraint, not an arbitrary
    limitation. A real multi-table deployment would need one session (and
    one watcher) per table, not more concurrency on this one.
    """

    def __init__(self) -> None:
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.events_seen = 0
        self.error: Optional[str] = None


watch_status = WatchStatus()


def _score_move_to_dict(scored: ScoredMove) -> dict:
    move_score = scored.move_score
    return {
        "turn_number": scored.candidate.turn_number,
        "player_id": scored.candidate.player_id,
        "move_type": scored.candidate.move_type.value,
        "new_cells": list(scored.candidate.new_cells),
        "score": {
            "total": move_score.total,
            "is_bingo": move_score.is_bingo,
            "words": [{"text": w.text, "score": w.score} for w in move_score.words],
        } if move_score is not None else None,
    }


# --- routes -------------------------------------------------------------------

@app.post("/moves")
async def submit_move(submission: MoveSubmission):
    new_tiles = [((t.row, t.col), t.letter, t.is_blank) for t in submission.new_tiles]
    rack_after = (
        [(t.letter, t.is_blank) for t in submission.rack_after]
        if submission.rack_after is not None else None
    )

    result = session.submit_move(
        submission.player_id, new_tiles=new_tiles, rack_after=rack_after, confidence=submission.confidence,
    )

    if isinstance(result.outcome, MoveProcessingError):
        raise HTTPException(status_code=422, detail=result.outcome.reason)

    if result.published:
        await overlay_connections.broadcast(build_overlay_state(session.game_state))

    return {"published": result.published, "move": _score_move_to_dict(result.outcome)}


@app.post("/moves/notation")
async def submit_move_notation(submission: NotationMoveSubmission):
    """Standard tournament notation, one line: `<position> <word>`, e.g.
    `A1 NO` or `14B PInOTa.E` (lowercase = blank played as that letter,
    '.' = a pre-existing cell the play hooks onto/through, not a new
    tile -- exactly the GCG convention `notation.py` already parses for
    replaying real game records, reused here so an operator never has to
    learn a second format). This is the fast path `submit_move`'s
    per-cell row/col/letter boxes were never meant to be for a human
    typing in real time.
    """
    parts = submission.line.strip().split()
    if len(parts) != 2:
        raise HTTPException(status_code=422, detail="expected '<position> <word>', e.g. 'A1 NO' or '14B PInOTa.E'")
    position, word = parts
    try:
        (row, col), direction = parse_position(position)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    new_tiles = []
    r, c = row, col
    for ch in word:
        if ch != ".":
            new_tiles.append(((r, c), ch.upper(), ch.islower()))
        if direction == "across":
            c += 1
        else:
            r += 1

    result = session.submit_move(submission.player_id, new_tiles=new_tiles, confidence=submission.confidence)

    if isinstance(result.outcome, MoveProcessingError):
        raise HTTPException(status_code=422, detail=result.outcome.reason)

    if result.published:
        await overlay_connections.broadcast(build_overlay_state(session.game_state))

    return {"published": result.published, "move": _score_move_to_dict(result.outcome)}


@app.get("/pending")
async def list_pending():
    return [
        {
            "turn_number": p.scored_move.candidate.turn_number,
            "move": _score_move_to_dict(p.scored_move),
            "has_frame": p.source_frame_jpeg is not None,
        }
        for p in session.list_pending()
    ]


@app.get("/pending/{turn_number}/frame")
async def pending_frame(turn_number: int):
    pending = session.pending.get(turn_number)
    if pending is None or pending.source_frame_jpeg is None:
        raise HTTPException(status_code=404, detail="no source frame for this pending move")
    return Response(content=pending.source_frame_jpeg, media_type="image/jpeg")


@app.get("/state/board")
async def get_board():
    """The live 15x15 grid, for the operator page's own board rendering
    -- separate from `/state` (which stays score/last-move summary only,
    matching what a stream overlay graphic needs) since a full grid is a
    different, heavier shape a broadcast overlay has no use for."""
    board = session.game_state.board
    cells = []
    for row in range(BOARD_SIZE):
        row_cells = []
        for col in range(BOARD_SIZE):
            tile = board.get((row, col))
            premium = "STAR" if (row, col) == CENTER else PREMIUM_SQUARES.get((row, col))
            row_cells.append({
                "letter": tile.letter if tile is not None else None,
                "is_blank": tile.is_blank if tile is not None else False,
                "premium": premium,
            })
        cells.append(row_cells)
    return {"cells": cells}


@app.post("/pending/{turn_number}/decision")
async def decide_pending(turn_number: int, decision: OperatorDecisionRequest):
    applied = session.decide(turn_number, decision.action)
    if applied is None:
        raise HTTPException(status_code=404, detail=f"no pending move for turn {turn_number}")
    if applied:
        await overlay_connections.broadcast(build_overlay_state(session.game_state))
    return {"applied": applied}


@app.post("/moves/undo")
async def undo_last_move():
    """Reverts the single most recently committed move -- the correction
    path for a wrong reading that already got approved (whether by an
    operator or auto-published), rather than merely rejected while still
    pending. Only ever undoes the tail: to fix an older mistake, undo
    repeatedly back to it, then resubmit the moves in between (see
    `GameState.undo_last`'s docstring for why an arbitrary earlier turn
    isn't supported)."""
    undone = session.undo_last_move()
    if undone is None:
        raise HTTPException(status_code=404, detail="no committed move to undo")
    await overlay_connections.broadcast(build_overlay_state(session.game_state))
    return {"undone": _score_move_to_dict(undone)}


@app.post("/mode")
async def set_mode(request: ModeChangeRequest):
    session.set_mode(request.mode)
    return {"mode": session.gateway.mode.value}


@app.get("/state")
async def get_state():
    return build_overlay_state(session.game_state)


@app.websocket("/ws/overlay")
async def overlay_websocket(ws: WebSocket):
    await overlay_connections.connect(ws)
    try:
        await ws.send_json(build_overlay_state(session.game_state))
        while True:
            # The overlay is a passive subscriber; we still need to await
            # incoming messages so we notice a disconnect promptly.
            await ws.receive_text()
    except WebSocketDisconnect:
        overlay_connections.disconnect(ws)


@app.get("/overlay")
async def overlay_page():
    return FileResponse(_STATIC_DIR / "overlay" / "web" / "overlay.html")


@app.get("/overlay/field")
async def overlay_field_page():
    """A single transparent value (?key=scores.p1, bag_count, last_word, ...)
    for use as an individual OBS Browser Source layered over an existing
    graphic, rather than the combined scorebug at /overlay. With no ?key=
    it renders a picker listing the fields currently available."""
    return FileResponse(_STATIC_DIR / "overlay" / "web" / "field.html")


@app.get("/operator")
async def operator_page():
    return FileResponse(_STATIC_DIR / "operator_ui" / "web" / "operator.html")


@app.post("/watch")
async def start_watching(request: WatchRequest):
    """Starts `run_watcher_on_video` against the app's own live `session`,
    in a background thread -- the connective tissue that makes a video
    file (today; a real camera, unchanged, once one exists) drive the
    exact same overlay/operator UI a human operator's typed-in moves
    already do. Every detected move goes through `session.submit_move`
    (see `GameWatcher`'s delegated mode), so it lands in the real
    `/pending` list and broadcasts to `/ws/overlay` the moment it
    publishes, same as a manual entry.

    One watch run at a time, matching one `GameSession` per process --
    returns 409 if a previous run hasn't finished.
    """
    if watch_status.running:
        raise HTTPException(status_code=409, detail="a watch run is already in progress")

    # training.detect imports rfdetr transitively, an optional dependency
    # this project's rack detection needs but board-only watching (all
    # run_watcher.py currently supports) does not -- imported lazily here
    # so a server with only the classifier installed can still serve
    # every other route.
    from autoscorer.perception.capture.run_watcher import run_watcher_on_video

    loop = asyncio.get_running_loop()

    def on_event(event) -> None:
        watch_status.events_seen += 1
        if event.needs_operator:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                overlay_connections.broadcast(build_overlay_state(session.game_state)), loop,
            )
        except RuntimeError:
            # The event loop this handler was bound to is gone (server
            # shutting down, or -- the only way this actually triggers in
            # practice today -- a test harness with a short-lived loop per
            # request). Either way, a missed overlay push for one move
            # must never abort the rest of the video; GET /state always
            # has the real, current answer regardless.
            pass

    def run() -> None:
        watch_status.running = True
        watch_status.events_seen = 0
        watch_status.error = None
        try:
            run_watcher_on_video(
                Path(request.video_path), request.venue, Path(request.classifier_path),
                request.player1, request.player2, sample_fps=request.sample_fps,
                max_frames=request.max_frames, session=session, on_event=on_event,
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced via /watch/status, not raised in this thread
            watch_status.error = str(exc)
        finally:
            watch_status.running = False

    watch_status.thread = threading.Thread(target=run, daemon=True)
    watch_status.thread.start()
    return {"started": True}


@app.get("/watch/status")
async def watch_status_endpoint():
    return {
        "running": watch_status.running,
        "events_seen": watch_status.events_seen,
        "error": watch_status.error,
    }


@app.get("/export/gcg", response_class=PlainTextResponse)
async def export_gcg_endpoint():
    """The current game's move history as GCG text -- the plain-text
    format tournament streaming graphics already know how to consume, so
    this is a real integration point, not a new format for this project to
    get adopted. Covers PLAY/EXCHANGE/PASS lines only; see `notation.py`'s
    module docstring for why the end-of-game rack bonus/penalty line is
    deliberately not produced.
    """
    return export_gcg(session.game_state)
