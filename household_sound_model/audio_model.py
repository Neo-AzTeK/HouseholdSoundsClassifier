"""Audio preprocessing and a compact five-output neural network."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy import signal
import torch
from torch import nn

LABELS = ("voices", "machines", "collisions", "steps", "music")
SOURCE_RATE = 16_125
STREAM_RATE = SOURCE_RATE / 4
WINDOW_SAMPLES = 4_031
HOP_SAMPLES = 2_016
N_FFT = 256
STFT_HOP = 80
FRAMES = 48
# Reject essentially silent decoded streams before classification. This level is
# below 231 of 233 held-out loudest test windows after Thingy simulation.
SILENCE_RMS = 5e-4
PREPROCESSING = {
    "version": 1,
    "source_rate": SOURCE_RATE,
    "stream_rate": STREAM_RATE,
    "window_samples": WINDOW_SAMPLES,
    "hop_samples": HOP_SAMPLES,
    "n_fft": N_FFT,
    "stft_hop": STFT_HOP,
    "frames": FRAMES,
}

# Copied from thingy-ranging/thingy_ranging/src/stream_dsp.c. The FIR is at 16,125 Hz.
FIR_Q15 = np.array([
    1, 0, -3, -6, -9, -6, 2, 16, 28, 30, 15, -16, -54, -79, -71, -21,
    60, 140, 176, 132, 5, -168, -311, -341, -207, 73, 403, 632, 615, 281,
    -308, -946, -1338, -1191, -336, 1190, 3120, 5016, 6402, 6916,
    6402, 5016, 3120, 1190, -336, -1191, -1338, -946, -308, 281,
    615, 632, 403, 73, -207, -341, -311, -168, 5, 132, 176,
    140, 60, -21, -71, -79, -54, -16, 15, 30, 28, 16,
    2, -6, -9, -6, -3, 0, 1,
], dtype=np.float64) / 32768.0
LOG_FRAC = np.array([
    0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3,
    3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5,
    5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
], dtype=np.int32)


def read_mono(path: str) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), rate


def simulate_thingy_stream(audio: np.ndarray, source_rate: int) -> np.ndarray:
    """Approximate the firmware's 4:1 FIR, high-pass and 8-bit companding.

    This is for public training clips. Actual live audio is already decoded from
    the Thingy BLE packets and should bypass this function.
    """
    from fractions import Fraction

    ratio = Fraction(SOURCE_RATE, source_rate)
    x = signal.resample_poly(audio, ratio.numerator, ratio.denominator)
    # Firmware emits on the fourth input sample. lfilter has causal FIR delay.
    x = signal.lfilter(FIR_Q15, [1.0], x)[3::4]
    x = signal.lfilter([1.0, -1.0], [1.0, -(32113 / 32768.0)], x)
    pcm = np.clip(np.rint(x * 32768.0), -32767, 32767).astype(np.int32)
    m = np.clip(np.abs(pcm) + 1, 1, 32768)
    e = np.floor(np.log2(m)).astype(np.int32)
    right = m >> np.maximum(e - 6, 0)
    left = m << np.maximum(6 - e, 0)
    f = np.where(e >= 6, right, left) & 63
    q = np.minimum(8 * e + LOG_FRAC[f], 127)
    expanded = (np.exp2(q / 8.0) - 1.0) * np.where(pcm < 0, -1.0, 1.0)
    return (expanded / 32768.0).astype(np.float32)


def fixed_window(audio: np.ndarray, start: int = 0) -> np.ndarray:
    part = np.asarray(audio[start:start + WINDOW_SAMPLES], dtype=np.float32)
    if part.size < WINDOW_SAMPLES:
        part = np.pad(part, (0, WINDOW_SAMPLES - part.size))
    return part


def candidate_starts(audio: np.ndarray) -> list[int]:
    """Three loud windows help avoid assigning a brief clip label to silence."""
    if len(audio) <= WINDOW_SAMPLES:
        return [0]
    starts = list(range(0, len(audio) - WINDOW_SAMPLES + 1, HOP_SAMPLES))
    starts.append(len(audio) - WINDOW_SAMPLES)
    energies = [float(np.mean(audio[s:s + WINDOW_SAMPLES] ** 2)) for s in starts]
    ordered = np.argsort(energies)[::-1]
    return [starts[int(i)] for i in ordered[:3]]


def spectrogram(window: np.ndarray) -> np.ndarray:
    _, _, z = signal.stft(
        window, fs=STREAM_RATE, nperseg=N_FFT, noverlap=N_FFT - STFT_HOP,
        boundary=None, padded=False,
    )
    feature = np.log1p(1000.0 * np.abs(z)).astype(np.float32)
    if feature.shape[1] < FRAMES:
        feature = np.pad(feature, ((0, 0), (0, FRAMES - feature.shape[1])))
    return feature[:129, :FRAMES][None, :, :]


class SoundCNN(nn.Module):
    """Input [batch, 1, 129, 48], output five independent logits."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = 1
        for width in (16, 32, 64, 96):
            layers.extend((
                nn.Conv2d(channels, width, 3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ))
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(96, len(LABELS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
