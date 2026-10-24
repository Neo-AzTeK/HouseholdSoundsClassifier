"""End-to-end test simulating Thingy BLE forwarder streaming into sound-web."""

import json
import struct
import numpy as np
import pytest
from fastapi.testclient import TestClient

from household_sounds.web.app import app


def test_thingy_to_webui_broadcast():
    client = TestClient(app)

    # 1. Connect Web UI client
    with client.websocket_connect("/ws/audio") as ui_ws:
        ui_ws.send_json({"type": "start", "source": "thingy"})
        init_msg = ui_ws.receive_json()
        assert init_msg["type"] == "ready"
        assert init_msg["source"] == "thingy"

        # 2. Connect Thingy forwarder (gateway.py / sound-bridge)
        with client.websocket_connect("/ws/thingy") as thingy_ws:
            ready = thingy_ws.receive_json()
            assert ready["type"] == "ready"

            # 3. Stream 25 chunks of 440 Hz pure tone on Node 0 (approx 1.18 sec of audio)
            dt = 1.0 / 4031.25
            t0 = 0.0
            for i in range(25):
                t = np.arange(190) * dt + t0
                samples = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
                t0 += 190 * dt

                # Compress into 8-bit companded codes
                int_samples = np.clip(np.round(samples * 32768.0), -32767, 32767).astype(np.int32)
                signs = np.where(int_samples < 0, 128, 0).astype(np.uint8)
                mags = np.abs(int_samples)
                q = np.clip(np.round(8.0 * np.log2(mags + 1.0)), 0, 127).astype(np.uint8)
                codes = (signs | q).tolist()

                # Send raw BLE packet
                head = struct.pack("<IBBBBQI", i, 0, 0xB5, 4, 190, 0, 0)
                packet = head + bytes(codes)
                thingy_ws.send_bytes(packet)

        # 4. Web UI should receive at least one prediction event for Node 0
        prediction = ui_ws.receive_json()
        assert prediction["type"] == "prediction"
        assert prediction["source"] == "thingy"
        assert prediction["node_id"] == 0
        assert "scores" in prediction
        assert set(prediction["scores"].keys()) == {"voices", "machines", "collisions", "steps"}
        assert "latency_ms" in prediction
        assert prediction["latency_ms"] >= 0.0
