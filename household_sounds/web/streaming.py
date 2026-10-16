"""Turn browser Float32 PCM chunks into overlapping classifier windows."""

from __future__ import annotations

from typing import Any
import numpy as np

from household_sounds.model import HOP_SAMPLES, STREAM_RATE, WINDOW_SAMPLES, simulate_thingy_stream
from household_sounds.predict import Predictor


class AudioSession:
    def __init__(self, sample_rate: int) -> None:
        if not 8_000 <= sample_rate <= 192_000:
            raise ValueError("sample rate must be between 8,000 and 192,000 Hz")
        self.sample_rate = sample_rate
        self.window_samples = round(sample_rate * WINDOW_SAMPLES / STREAM_RATE)
        self.hop_samples = round(sample_rate * HOP_SAMPLES / STREAM_RATE)
        self.buffer = np.empty(0, dtype=np.float32)

    def push_bytes(self, payload: bytes) -> list[np.ndarray]:
        """Return complete one-second windows; retain overlap for the next chunk."""
        if not payload or len(payload) % 4:
            raise ValueError("audio messages must contain Float32 PCM samples")
        if len(payload) > 4 * self.sample_rate:
            raise ValueError("audio chunk exceeds one second")
        samples = np.frombuffer(payload, dtype="<f4")
        if not np.isfinite(samples).all():
            raise ValueError("audio chunk contains a non-finite sample")
        self.buffer = np.concatenate((self.buffer, np.clip(samples, -1.0, 1.0)))
        windows: list[np.ndarray] = []
        while self.buffer.size >= self.window_samples:
            windows.append(self.buffer[:self.window_samples].copy())
            self.buffer = self.buffer[self.hop_samples:]
        return windows


class ThingySession:
    """Manages audio buffers, decoding, and windowing for incoming Thingy:53 BLE streams."""

    MAGIC_STREAM = 0xB5
    STREAM_CHUNK_SIZE = 210

    def __init__(self) -> None:
        import time
        from collections import deque
        self._time = time
        self.buffers: dict[int, np.ndarray] = {}
        self.stats: dict[int, dict[str, Any]] = {}
        self._packet_times: dict[int, deque[float]] = {}

    def get_nodes_status(self) -> dict[int, dict[str, Any]]:
        now = self._time.monotonic()
        res = {}
        for node_id, s in self.stats.items():
            times = self._packet_times.get(node_id)
            if times:
                while times and now - times[0] > 2.0:
                    times.popleft()
                rate = len(times) / 2.0
            else:
                rate = 0.0
            res[node_id] = {
                "node_id": node_id,
                "packets": s.get("packets", 0),
                "last_seen_s_ago": round(now - s.get("last_seen", now), 2),
                "packet_rate": round(rate, 1),
                "rms": s.get("rms", 0.0),
                "buffer_samples": self.buffers.get(node_id, np.empty(0)).size,
            }
        return res

    def push_raw_packet(self, data: bytes) -> tuple[int, list[np.ndarray]]:
        """Parse 210-byte Thingy BLE packet and return (node_id, list_of_windows)."""
        from household_sounds.predict import decode_codes

        if len(data) != self.STREAM_CHUNK_SIZE or data[5] != self.MAGIC_STREAM:
            magic_hex = f"{data[5]:#x}" if len(data) > 5 else "none"
            raise ValueError(f"Invalid Thingy stream packet: len={len(data)}, magic={magic_hex}")
        node_id = int(data[4])
        count = int(data[7])
        codes = data[20 : 20 + count]
        samples = decode_codes(codes)
        return node_id, self.push_samples(node_id, samples)

    def push_message(self, message: dict[str, Any]) -> tuple[int, list[np.ndarray]]:
        """Parse NDJSON / dict message {'node_id': ..., 'codes' or 'samples': ...}."""
        from household_sounds.predict import decode_codes

        node_id = int(message.get("node_id", 0))
        if "codes" in message:
            samples = decode_codes(message["codes"])
        elif "samples" in message:
            samples = np.asarray(message["samples"], dtype=np.float32)
            if samples.size and float(np.max(np.abs(samples))) > 2.0:
                samples = samples / 32768.0
        else:
            raise ValueError("Message must contain 'codes' or 'samples'")
        return node_id, self.push_samples(node_id, samples)

    def push_samples(self, node_id: int, samples: np.ndarray) -> list[np.ndarray]:
        from collections import deque
        now = self._time.monotonic()
        if node_id not in self.buffers:
            self.buffers[node_id] = np.empty(0, dtype=np.float32)
            self.stats[node_id] = {"packets": 0, "last_seen": now, "rms": 0.0}
            self._packet_times[node_id] = deque(maxlen=100)

        self._packet_times[node_id].append(now)
        self.stats[node_id]["packets"] += 1
        self.stats[node_id]["last_seen"] = now
        if samples.size > 0:
            rms = float(np.sqrt(np.mean(samples * samples)))
            self.stats[node_id]["rms"] = round(rms, 6)

        self.buffers[node_id] = np.concatenate((self.buffers[node_id], samples))

        windows: list[np.ndarray] = []
        while self.buffers[node_id].size >= WINDOW_SAMPLES:
            windows.append(self.buffers[node_id][:WINDOW_SAMPLES].copy())
            self.buffers[node_id] = self.buffers[node_id][HOP_SAMPLES:]
        return windows


def classify_browser_window(predictor: Predictor, window: np.ndarray, sample_rate: int) -> dict[str, Any]:
    stream = simulate_thingy_stream(window, sample_rate)
    result = predictor.predict(stream)
    result["source"] = "browser"
    for label, score in result["scores"].items():
        result[label] = score
    result["type"] = "prediction"
    return result


def classify_thingy_window(predictor: Predictor, window: np.ndarray, node_id: int) -> dict[str, Any]:
    """Classify 4,032-sample window directly from Thingy (native ~4031 Hz stream, no simulation needed)."""
    result = predictor.predict(window)
    result["source"] = "thingy"
    result["node_id"] = node_id
    result["type"] = "prediction"
    for label, score in result["scores"].items():
        result[label] = score
    return result
