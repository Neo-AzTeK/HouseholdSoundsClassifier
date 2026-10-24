"""Unit and integration tests for Dataset Studio, BLE Manager, and Threshold Tuning."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient

from household_sounds.dataset_studio import DatasetStudioManager
from household_sounds.predict import LABELS
from household_sounds.web.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_dataset_studio_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = DatasetStudioManager(data_dir=Path(tmpdir))

        # 1. Start take
        start_res = manager.start_recording(source="thingy", sample_rate=4031)
        assert start_res["ok"] is True
        assert manager.is_recording is True
        rec_id = start_res["recording_id"]

        # 2. Push audio samples (1 second at 4031 Hz)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 4031)).astype(np.float32)
        manager.push_audio(audio, source="thingy")

        # 3. Stop take
        stop_res = manager.stop_recording()
        assert stop_res["ok"] is True
        assert stop_res["recording_id"] == rec_id
        assert stop_res["duration_s"] >= 0.9
        assert len(stop_res["waveform"]) == 300
        assert (manager.audio_dir / f"{rec_id}.wav").is_file()

        # 4. Annotate segment
        labels = {"impacts": 1, "voices": 0, "machines": 0}
        ann_res = manager.annotate_segment(rec_id, start_s=0.1, end_s=0.9, labels=labels)
        assert ann_res["ok"] is True
        assert ann_res["segment_id"].startswith(rec_id)

        # 5. Summary verification
        summary = manager.get_dataset_summary()
        assert summary["total_segments"] == 1
        assert summary["label_counts"]["impacts"] == 1
        assert summary["label_counts"]["voices"] == 0

        # 6. Delete segment
        del_ok = manager.delete_segment(ann_res["segment_id"])
        assert del_ok is True
        summary_after = manager.get_dataset_summary()
        assert summary_after["total_segments"] == 0


def test_web_threshold_endpoints(client):
    # GET thresholds
    resp = client.get("/api/thresholds")
    assert resp.status_code == 200
    data = resp.json()
    assert "thresholds" in data
    assert "voices" in data["thresholds"]
    init_voices = data["thresholds"]["voices"]
    init_impacts = data["thresholds"]["impacts"]

    # POST new thresholds
    new_th = {"voices": 0.85, "impacts": 0.45}
    set_resp = client.post("/api/thresholds", json={"thresholds": new_th, "persist": False})
    assert set_resp.status_code == 200
    set_data = set_resp.json()
    assert set_data["thresholds"]["voices"] == 0.85
    assert set_data["thresholds"]["impacts"] == 0.45

    # POST reset thresholds
    reset_resp = client.post("/api/thresholds/reset", json={"persist": False})
    assert reset_resp.status_code == 200
    reset_data = reset_resp.json()
    assert reset_data["thresholds"]["voices"] == init_voices
    assert reset_data["thresholds"]["impacts"] == init_impacts


def test_web_ble_endpoints(client):
    status_resp = client.get("/api/ble/status")
    assert status_resp.status_code == 200
    st = status_resp.json()
    assert "state" in st
    assert "connected" in st


def test_web_studio_endpoints(client):
    # Test dataset summary endpoint
    sum_resp = client.get("/api/studio/dataset")
    assert sum_resp.status_code == 200
    data = sum_resp.json()
    assert "total_segments" in data
    assert "label_counts" in data

    # Test recordings listing
    rec_resp = client.get("/api/studio/recordings")
    assert rec_resp.status_code == 200
    assert "recordings" in rec_resp.json()
