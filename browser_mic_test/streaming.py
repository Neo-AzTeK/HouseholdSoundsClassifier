"""Turn browser Float32 PCM chunks into overlapping classifier windows."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "household_sound_model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

try:
    from audio_model import HOP_SAMPLES, STREAM_RATE, WINDOW_SAMPLES, simulate_thingy_stream
    from predict import Predictor
except ModuleNotFoundError:
    from household_sound_model.audio_model import HOP_SAMPLES, STREAM_RATE, WINDOW_SAMPLES, simulate_thingy_stream
    from household_sound_model.predict import Predictor

CHECKPOINT = MODEL_DIR / "artifacts" / "household_sound.pt"


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


def classify_browser_window(predictor: Predictor, window: np.ndarray, sample_rate: int) -> dict[str, float]:
    stream = simulate_thingy_stream(window, sample_rate)
    return predictor.scores(stream)
