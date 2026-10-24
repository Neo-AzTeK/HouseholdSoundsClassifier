"""Integration tests for web app endpoints including Thingy streaming."""

import struct
import numpy as np
import pytest
from fastapi.testclient import TestClient

from household_sounds.web.app import app, thingy_session


@pytest.fixture
def client():
    return TestClient(app)


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Household Sound Classifier" in response.text
    assert "source-thingy" in response.text
    assert "thingy-controls" in response.text


def test_backends_api(client):
    response = client.get("/api/backends")
    assert response.status_code == 200
    data = response.json()
    assert "active" in data
    assert "cpu" in data
    assert data["cpu"]["available"] is True


def test_thingy_status_and_mock_api(client):
    # Initial status
    response = client.get("/api/thingy/status")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data

    # Inject mock stream
    mock_resp = client.post("/api/thingy/mock", json={"node_id": 1, "sound": "tone", "chunks": 25})
    assert mock_resp.status_code == 200
    mock_data = mock_resp.json()
    assert mock_data["ok"] is True
    assert mock_data["node_id"] == 1
    assert mock_data["chunks_injected"] == 25

    # Verify status updated
    status_resp = client.get("/api/thingy/status")
    status_data = status_resp.json()
    assert "1" in status_data["nodes"] or 1 in status_data["nodes"]
    node1 = status_data["nodes"].get("1") or status_data["nodes"].get(1)
    assert node1["packets"] >= 25


def test_thingy_websocket_ingestion(client):
    with client.websocket_connect("/ws/thingy") as ws:
        ready_msg = ws.receive_json()
        assert ready_msg["type"] == "ready"

        # Test ping-pong
        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"
        assert "nodes" in pong

        # Test NDJSON message ingestion
        ws.send_json({"node_id": 0, "codes": [0] * 190})

        # Test raw 210-byte binary packet ingestion
        head = struct.pack("<IBBBBQI", 1, 0, 0xB5, 4, 190, 0, 0)
        packet = head + bytes(190)
        ws.send_bytes(packet)


def test_agc_configuration_and_prediction(client):
    from household_sounds.predict import Predictor
    from household_sounds.model import WINDOW_SAMPLES

    predictor = Predictor()
    assert predictor.use_agc is True

    # Test faint signal (RMS ~0.005) gets amplified with AGC
    faint_window = np.sin(np.linspace(0, 100, WINDOW_SAMPLES, dtype=np.float32)) * 0.007
    pred_agc = predictor.predict(faint_window)
    assert pred_agc["silence"] is False
    assert pred_agc["gain"] > 1.0

    # Test with AGC disabled
    predictor.use_agc = False
    pred_no_agc = predictor.predict(faint_window)
    assert pred_no_agc["gain"] == 1.0

    # Test POST /api/config
    cfg_resp = client.post("/api/config", json={"use_agc": False})
    assert cfg_resp.status_code == 200
    assert cfg_resp.json()["use_agc"] is False

    cfg_resp2 = client.post("/api/config", json={"use_agc": True})
    assert cfg_resp2.status_code == 200
    assert cfg_resp2.json()["use_agc"] is True
