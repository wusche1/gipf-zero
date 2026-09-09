from fastapi.testclient import TestClient
import pytest
import time

import gipf_engine as ge
import server.app as api


@pytest.fixture(autouse=True)
def clean_server_state(monkeypatch):
    api.requests.clear()
    api.global_requests.clear()
    api.inflight = 0
    api.model = None
    api.model_stamp = 0
    api.model_info = {}
    yield
    api.requests.clear()
    api.global_requests.clear()


@pytest.fixture
def client():
    with TestClient(api.app) as test_client:
        yield test_client


def opening():
    return ge.State().serialize()


def terminal_state():
    state = opening()
    # A legal post-settlement reserve terminal: Black is now due to insert.
    state.update(reserves=[12, 0], captured=[0, 12], current_player=-1,
                 turn_player=1, phase="push", winner=1, ply=1)
    return state


def test_move_returns_a_legal_action(client, monkeypatch):
    def fake_infer(data, budget_ms):
        state = ge.State.from_dict(data)
        return {"action": state.legal_actions()[0], "elapsed_ms": 1,
                "model": "test", "kind": "baseline"}

    monkeypatch.setattr(api, "infer", fake_infer)
    payload = {"state": opening(), "budget_ms": 50}
    response = client.post("/api/move", json=payload)
    assert response.status_code == 200
    assert response.json()["action"] in ge.State.from_dict(payload["state"]).legal_actions()


def test_move_rejects_malformed_and_terminal_states(client):
    malformed = opening()
    malformed["board"] = malformed["board"][:-1]
    response = client.post("/api/move", json={"state": malformed, "budget_ms": 50})
    assert response.status_code == 422
    assert "37" in response.json()["detail"]

    response = client.post("/api/move", json={"state": terminal_state(), "budget_ms": 50})
    assert response.status_code == 422
    assert response.json()["detail"] == "Game is already over"


def test_post_body_limit_and_invalid_length_are_rejected(client):
    response = client.post("/api/move", content=b"x" * 8193, headers={"content-type": "application/json"})
    assert response.status_code == 413
    response = client.post("/api/move", content=b"{}", headers={"content-type": "application/json", "content-length": "nope"})
    assert response.status_code == 400


def test_pages_cors_preflight(client):
    response = client.options(
        "/api/move",
        headers={"Origin": "https://wusche1.github.io", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://wusche1.github.io"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_rate_limit_client_bookkeeping_is_bounded(client, monkeypatch):
    for i in range(5000):
        api.requests[f"client-{i}"].append(time.monotonic())
    monkeypatch.setattr(api, "infer", lambda data, budget_ms: {"action": 0})
    response = client.post("/api/move", json={"state": opening(), "budget_ms": 50}, headers={"cf-connecting-ip": "another-client"})
    assert response.status_code == 200
    assert len(api.requests) == 5001


def test_champion_schema_and_loading_metadata(tmp_path, monkeypatch):
    metadata = {"config": {"kind": "mlp", "width": 64, "blocks": 1}, "iteration": 3, "games": 12}
    assert api.champion_metadata(metadata) == {"name": "RL champion", "iteration": 3, "games": 12}
    with pytest.raises(ValueError):
        api.champion_metadata({"iteration": 3, "games": 12})
    with pytest.raises(ValueError):
        api.champion_metadata({"config": {}, "iteration": -1, "games": 12})

    champion = tmp_path / "champion.pt"
    champion.write_bytes(b"promoted checkpoint placeholder")
    monkeypatch.setattr(api, "CHAMPION", champion)
    sentinel = object()
    monkeypatch.setattr(api, "load_model", lambda path, device: (sentinel, metadata))
    monkeypatch.setattr(api, "choose_action", lambda model, state, device, simulations, budget_ms: (state.legal_actions()[0], {"simulations": 1}))
    result = api.infer(opening(), 50)
    assert result["action"] in ge.State().legal_actions()
    assert result["kind"] == "rl"
    assert api.model_info == {"name": "RL champion", "iteration": 3, "games": 12}
