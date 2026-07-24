"""FastAPI backend: REST endpoints for manual move entry and operator
decisions, a WebSocket that pushes OverlayState to the stream overlay on
every applied move, and the two static pages (overlay + operator UI).

Single-process, single in-memory GameSession for now -- this is the Phase 2
"manual mode" product described in the plan: a human operator drives the
whole game, with correct scoring, exact pool tracking, and a working stream
overlay, entirely without a camera.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from autoscorer.api.session import GameSession
from autoscorer.gamelogic.models import MoveProcessingError, ScoredMove
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


class OperatorDecisionRequest(BaseModel):
    action: str  # "approve" | "reject"


class ModeChangeRequest(BaseModel):
    mode: PublishMode


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


@app.get("/pending")
async def list_pending():
    return [
        {"turn_number": p.scored_move.candidate.turn_number, "move": _score_move_to_dict(p.scored_move)}
        for p in session.list_pending()
    ]


@app.post("/pending/{turn_number}/decision")
async def decide_pending(turn_number: int, decision: OperatorDecisionRequest):
    applied = session.decide(turn_number, decision.action)
    if applied is None:
        raise HTTPException(status_code=404, detail=f"no pending move for turn {turn_number}")
    if applied:
        await overlay_connections.broadcast(build_overlay_state(session.game_state))
    return {"applied": applied}


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
