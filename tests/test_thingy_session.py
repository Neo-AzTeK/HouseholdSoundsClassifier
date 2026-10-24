"""Tests for Thingy:53 audio streaming and session windowing."""

import struct
import numpy as np
import pytest

from household_sounds.model import HOP_SAMPLES, WINDOW_SAMPLES
from household_sounds.web.streaming import ThingySession, classify_thingy_window
from household_sounds.predict import Predictor, decode_codes


def test_decode_codes():
    # Test all-zeros code
    codes = [0, 0, 0]
    decoded = decode_codes(codes)
    assert decoded.shape == (3,)
    assert np.allclose(decoded, 0.0)

    # Test with bytes input
    byte_codes = bytes([0, 128, 16])
    decoded_bytes = decode_codes(byte_codes)
    assert decoded_bytes.shape == (3,)
    # code 128 has sign bit set, magnitude 0 -> -0.0
    assert np.allclose(decoded_bytes[0], 0.0)


def test_thingy_session_windowing():
    session = ThingySession()
    assert session.get_nodes_status() == {}

    # Each chunk is 190 samples
    chunk_samples = 190
    # To reach WINDOW_SAMPLES (4032), we need 4032 / 190 = ~21.2 chunks
    # 22 chunks = 4180 samples >= 4032
    windows_found = []
    for i in range(22):
        fake_samples = np.random.uniform(-0.1, 0.1, chunk_samples).astype(np.float32)
        windows = session.push_samples(node_id=0, samples=fake_samples)
        windows_found.extend(windows)

    assert len(windows_found) == 1
    assert windows_found[0].shape == (WINDOW_SAMPLES,)

    status = session.get_nodes_status()
    assert 0 in status
    assert status[0]["packets"] == 22
    assert status[0]["packet_rate"] > 0
    assert status[0]["buffer_samples"] == (22 * chunk_samples) - HOP_SAMPLES


def test_thingy_session_raw_packet():
    session = ThingySession()

    # Construct a valid 210-byte Thingy BLE packet
    # CHUNK_FORMAT = "<IBBBBQI" (20 bytes)
    # index=1, node=2, magic=0xB5, decim=4, count=190, t_first=0, period_q8=0
    head = struct.pack("<IBBBBQI", 1, 2, 0xB5, 4, 190, 0, 0)
    assert len(head) == 20
    payload = bytes(190)  # 190 zeros
    packet = head + payload
    assert len(packet) == 210

    node_id, windows = session.push_raw_packet(packet)
    assert node_id == 2
    assert len(windows) == 0  # not enough samples yet for a full 4032-sample window

    status = session.get_nodes_status()
    assert 2 in status
    assert status[2]["packets"] == 1


def test_thingy_session_roomsense_packet_resampling():
    session = ThingySession()
    # decim=3 (roomsense 5376 Hz)
    head = struct.pack("<IBBBBQI", 1, 0, 0xB5, 3, 190, 0, 0)
    payload = bytes(190)
    packet = head + payload
    assert len(packet) == 210

    node_id, windows = session.push_raw_packet(packet)
    assert node_id == 0
    # Resampled 190 samples at ratio 3/4 should give 143 samples in buffer
    assert session.buffers[0].size == 143
    status = session.get_nodes_status()
    assert 0 in status
    assert status[0]["packets"] == 1
    assert status[0]["buffer_samples"] == 143


def test_classify_thingy_window():
    predictor = Predictor()
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    res = classify_thingy_window(predictor, window, node_id=1)
    assert res["source"] == "thingy"
    assert res["node_id"] == 1
    assert res["type"] == "prediction"
    assert "scores" in res
    assert "voices" in res
