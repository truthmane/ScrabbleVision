import pytest
from fastapi.testclient import TestClient

from autoscorer.api import main
from autoscorer.api.session import GameSession


def test_session_decide_rejects_unknown_action():
    session = GameSession()
    session.submit_move("p1", new_tiles=[((7, 7), "H", False), ((7, 8), "I", False)])
    with pytest.raises(ValueError):
        session.decide(1, "aprove")
    assert 1 in session.pending
    assert not session._rejected_signatures


def test_decision_endpoint_returns_422_for_unknown_action():
    client = TestClient(main.app)
    response = client.post("/pending/1/decision", json={"action": "aprove"})
    assert response.status_code == 422
