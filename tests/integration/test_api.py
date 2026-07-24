"""API-level integration tests: exercise the FastAPI app the way the
operator UI and overlay actually do, over HTTP and WebSocket, using an
in-process TestClient (no real network, no camera -- still hermetic).
"""
import pytest
from fastapi.testclient import TestClient

from autoscorer.api import main as main_module
from autoscorer.api.session import GameSession

FIRST_MOVE_TILES = [
    {"row": 7, "col": 6, "letter": "A"},
    {"row": 7, "col": 7, "letter": "N"},
    {"row": 7, "col": 8, "letter": "T"},
]


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
