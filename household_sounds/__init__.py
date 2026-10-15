"""Household Sounds Classifier package."""

from household_sounds.model import (
    FRAMES,
    HOP_SAMPLES,
    LABELS,
    PREPROCESSING,
    SILENCE_RMS,
    SOURCE_RATE,
    STREAM_RATE,
    WINDOW_SAMPLES,
    SoundCNN,
    candidate_starts,
    fixed_window,
    read_mono,
    simulate_thingy_stream,
    spectrogram,
)
from household_sounds.predict import Predictor

__all__ = [
    "FRAMES",
    "HOP_SAMPLES",
    "LABELS",
    "PREPROCESSING",
    "SILENCE_RMS",
    "SOURCE_RATE",
    "STREAM_RATE",
    "WINDOW_SAMPLES",
    "SoundCNN",
    "candidate_starts",
    "fixed_window",
    "read_mono",
    "simulate_thingy_stream",
    "spectrogram",
    "Predictor",
]
