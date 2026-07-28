"""API-level integration tests: exercise the FastAPI app the way the
operator UI and overlay actually do, over HTTP and WebSocket, using an
in-process TestClient (no real network, no camera -- still hermetic).
"""
import random
import time

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from autoscorer.api import main as main_module
from autoscorer.api.session import GameSession
from autoscorer.gamelogic.board import BOARD_SIZE, CENTER, PREMIUM_SQUARES
from autoscorer.perception.calibration.homography import CANONICAL_SIZE, cell_bounds
from autoscorer.perception.calibration.venue_profile import VenueProfile, save_venue_profile
from training.classify.train import run_training, save_checkpoint
from training.synth_render.tile_renderer import SQUARE_COLORS, augment_tile, render_tile

FIRST_MOVE_TILES = [
    {"row": 7, "col": 6, "letter": "A"},
    {"row": 7, "col": 7, "letter": "N"},
    {"row": 7, "col": 8, "letter": "T"},
]

IDENTITY_CORNERS = ((0.0, 0.0), (CANONICAL_SIZE, 0.0), (CANONICAL_SIZE, CANONICAL_SIZE), (0.0, CANONICAL_SIZE))


@pytest.fixture(autouse=True)
def fresh_session():
    """Each test gets an isolated GameSession -- the module-level session
    is a process-wide singleton in the real app, so tests must reset it."""
    main_module.session = GameSession()
    main_module.overlay_connections.active.clear()
    yield


@pytest.fixture
def client():
    return TestClient(main_module.app)


def test_manual_mode_queues_move_for_review(client):
    resp = client.post("/moves", json={"player_id": "p1", "new_tiles": FIRST_MOVE_TILES})
    assert resp.status_code == 200
    body = resp.json()
    assert body["published"] is False

    pending = client.get("/pending").json()
    assert len(pending) == 1
    assert pending[0]["move"]["turn_number"] == 1


def test_approving_a_pending_move_publishes_it(client):
    client.post("/moves", json={"player_id": "p1", "new_tiles": FIRST_MOVE_TILES})

    resp = client.post("/pending/1/decision", json={"action": "approve"})
    assert resp.status_code == 200
    assert resp.json()["applied"] is True

    state = client.get("/state").json()
    assert state["scores"]["p1"] == (1 + 1 + 1) * 2
    assert client.get("/pending").json() == []


def test_rejecting_a_pending_move_leaves_state_untouched(client):
    client.post("/moves", json={"player_id": "p1", "new_tiles": FIRST_MOVE_TILES})
    resp = client.post("/pending/1/decision", json={"action": "reject"})
    assert resp.json()["applied"] is False

    state = client.get("/state").json()
    assert state["scores"] == {}


def test_deciding_unknown_turn_returns_404(client):
    resp = client.post("/pending/999/decision", json={"action": "approve"})
    assert resp.status_code == 404


def test_invalid_placement_returns_422(client):
    bad_tiles = [{"row": 0, "col": 0, "letter": "A"}, {"row": 0, "col": 1, "letter": "T"}]
    resp = client.post("/moves", json={"player_id": "p1", "new_tiles": bad_tiles})
    assert resp.status_code == 422
    assert "center" in resp.json()["detail"]


def test_autonomous_mode_publishes_immediately(client):
    client.post("/mode", json={"mode": "AUTONOMOUS"})
    resp = client.post("/moves", json={"player_id": "p1", "new_tiles": FIRST_MOVE_TILES})
    assert resp.json()["published"] is True
    assert client.get("/pending").json() == []
    assert client.get("/state").json()["scores"]["p1"] == (1 + 1 + 1) * 2


def test_undo_endpoint_reverses_the_last_published_move(client):
    client.post("/mode", json={"mode": "AUTONOMOUS"})
    client.post("/moves", json={"player_id": "p1", "new_tiles": FIRST_MOVE_TILES})

    resp = client.post("/moves/undo")
    assert resp.status_code == 200
    assert resp.json()["undone"]["player_id"] == "p1"

    state = client.get("/state").json()
    assert state["scores"].get("p1", 0) == 0


def test_undo_endpoint_404s_with_nothing_to_undo(client):
    resp = client.post("/moves/undo")
    assert resp.status_code == 404


def test_export_gcg_endpoint_returns_the_move_history_as_gcg_text(client):
    client.post("/mode", json={"mode": "AUTONOMOUS"})
    resp = client.post("/moves", json={"player_id": "p1", "new_tiles": FIRST_MOVE_TILES})
    assert resp.json()["published"] is True

    resp = client.get("/export/gcg")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")

    lines = resp.text.splitlines()
    assert lines[0] == "#player1 p1 p1"
    assert lines[1] == ">p1:  8G ANT +6 6"  # p1's rack is unknown here, so it renders empty


def test_overlay_field_page_is_served(client):
    resp = client.get("/overlay/field")
    assert resp.status_code == 200
    assert b"getPath" in resp.content  # sanity check it's the field-picker page, not a 404 page


def test_overlay_websocket_receives_initial_state_then_broadcast_on_publish(client):
    client.post("/mode", json={"mode": "AUTONOMOUS"})
    with client.websocket_connect("/ws/overlay") as ws:
        initial = ws.receive_json()
        assert initial["turn_number"] == 0

        client.post("/moves", json={"player_id": "p1", "new_tiles": FIRST_MOVE_TILES})

        update = ws.receive_json()
        assert update["turn_number"] == 1
        assert update["last_word"] == "ANT"
        assert update["scores"]["p1"] == (1 + 1 + 1) * 2


# --- /watch: GameWatcher driving the live session from a video ---------------

def _blank_board_image() -> np.ndarray:
    img = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE, 3), dtype=np.uint8)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            code = PREMIUM_SQUARES.get((row, col), "plain")
            color = SQUARE_COLORS[code]
            x1, y1, x2, y2 = cell_bounds(row, col)
            img[y1:y2, x1:x2] = color[::-1]
    return img


def _place_tile(board_img: np.ndarray, row: int, col: int, letter, rng: random.Random) -> None:
    tile = augment_tile(render_tile(letter, rng=rng), rng=rng)
    tile_arr = np.array(tile)[:, :, ::-1]
    x1, y1, x2, y2 = cell_bounds(row, col)
    board_img[y1:y2, x1:x2] = tile_arr


def _train_tiny_classifier(tmp_path, labels=("A", "N", "T", "BLANK")):
    rng = random.Random(0)
    for label in labels:
        letter = None if label == "BLANK" else label
        class_dir = tmp_path / label
        class_dir.mkdir(parents=True, exist_ok=True)
        base = render_tile(letter, rng=rng)
        for i in range(30):
            augment_tile(base, rng=rng).save(class_dir / f"{i:03d}.png")

    model, result = run_training(tmp_path, epochs=25, batch_size=8, device=torch.device("cpu"), seed=1)
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(model, result.classes, checkpoint_path)
    return checkpoint_path


def _write_video(path, frames, fps=10.0) -> None:
    size = (frames[0].shape[1], frames[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*"FFV1")  # lossless -- see test_run_watcher.py's note on mp4v
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_watch_endpoint_drives_the_live_session_from_a_video(client, tmp_path, monkeypatch):
    # The point of this test: a move detected from a *video file* ends up
    # in the exact same place a human operator's typed-in move would --
    # applied to session.game_state, visible via GET /state, and broadcast
    # to the overlay WebSocket -- via GameWatcher's delegated (session=)
    # mode, not a separate code path.
    client.post("/mode", json={"mode": "AUTONOMOUS"})

    checkpoint_path = _train_tiny_classifier(tmp_path)
    reference = _blank_board_image()
    rng = random.Random(7)

    still_frame_count = 3
    frames = [reference.copy() for _ in range(still_frame_count)]
    placed = reference.copy()
    _place_tile(placed, *CENTER, "A", rng)
    frames.append(reference.copy())
    frames.append(np.zeros_like(reference))
    # +1: a placement is only committed once the same set of new cells is
    # confirmed by a SECOND independent settled observation (see
    # GameWatcher's _pending_new_cells) -- still_frame_count identical
    # frames only reaches the first sighting, one more confirms it.
    frames += [placed.copy() for _ in range(still_frame_count + 1)]

    video_path = tmp_path / "test_game.mkv"
    _write_video(video_path, frames)

    reference_path = tmp_path / "reference.png"
    cv2.imwrite(str(reference_path), reference)
    profile = VenueProfile(
        name="test_venue", corners=IDENTITY_CORNERS, still_frame_count=still_frame_count,
        reference_board_path=str(reference_path),
    )
    save_venue_profile(profile, directory=tmp_path)

    import autoscorer.perception.capture.run_watcher as run_watcher_module
    original_loader = run_watcher_module.load_venue_profile
    monkeypatch.setattr(run_watcher_module, "load_venue_profile", lambda name: original_loader(name, directory=tmp_path))

    resp = client.post("/watch", json={
        "video_path": str(video_path), "venue": "test_venue",
        "player1": "Alice", "player2": "Bob",
        "sample_fps": None, "classifier_path": str(checkpoint_path),
    })
    assert resp.status_code == 200
    assert resp.json()["started"] is True

    # A second concurrent /watch should be rejected outright.
    assert client.post("/watch", json={
        "video_path": str(video_path), "venue": "test_venue",
        "player1": "Alice", "player2": "Bob", "classifier_path": str(checkpoint_path),
    }).status_code == 409

    for _ in range(200):
        if not client.get("/watch/status").json()["running"]:
            break
        time.sleep(0.1)
    status = client.get("/watch/status").json()
    assert status["running"] is False
    assert status["error"] is None
    assert status["events_seen"] == 1

    # The detected move landed in the same place a manual submission would --
    # applied to the live session, not stuck in some separate watcher-only
    # state nobody else can see.
    state = client.get("/state").json()
    assert state["scores"]["Alice"] == 2
    assert client.get("/pending").json() == []
