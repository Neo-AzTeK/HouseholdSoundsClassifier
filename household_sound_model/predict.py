"""Use a trained checkpoint on a file or decoded live Thingy microphone chunks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from audio_model import (
    HOP_SAMPLES, LABELS, PREPROCESSING, SILENCE_RMS, STREAM_RATE, WINDOW_SAMPLES, SoundCNN,
    fixed_window, read_mono, simulate_thingy_stream, spectrogram,
)

ROOT = Path(__file__).resolve().parent


class Predictor:
    def __init__(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if tuple(checkpoint["labels"]) != LABELS:
            raise ValueError("Checkpoint labels do not match this code")
        if checkpoint.get("preprocessing") != PREPROCESSING:
            raise ValueError("Checkpoint preprocessing does not match this code")
        self.model = SoundCNN().eval()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.thresholds = dict(zip(LABELS, checkpoint["thresholds"]))
        self.silence_rms = float(checkpoint.get("silence_rms", SILENCE_RMS))

    @torch.no_grad()
    def scores(self, window: np.ndarray) -> dict[str, float]:
        window = fixed_window(window)
        if float(np.sqrt(np.mean(window * window))) < self.silence_rms:
            return {label: 0.0 for label in LABELS}
        features = spectrogram(window)
        x = torch.from_numpy(features)[None, :, :, :]
        values = torch.sigmoid(self.model(x)).numpy()[0]
        return {label: round(float(value), 4) for label, value in zip(LABELS, values)}


def decode_codes(codes: list[int]) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint8)
    magnitude = np.exp2((values & 127).astype(np.float32) / 8.0) - 1.0
    return (magnitude * np.where(values & 128, -1.0, 1.0) / 32768.0).astype(np.float32)


def file_mode(predictor: Predictor, path: Path, already_stream: bool, summary: bool) -> None:
    audio, rate = read_mono(str(path))
    if already_stream:
        if abs(rate - STREAM_RATE) > 15:
            raise SystemExit(f"Expected a roughly {STREAM_RATE:.0f} Hz decoded stream WAV, got {rate} Hz")
        stream = audio
    else:
        stream = simulate_thingy_stream(audio, rate)
    starts = list(range(0, max(len(stream) - WINDOW_SAMPLES + 1, 1), HOP_SAMPLES))
    if not starts:
        starts = [0]
    final_start = max(0, len(stream) - WINDOW_SAMPLES)
    if starts[-1] != final_start:
        starts.append(final_start)
    results = [predictor.scores(fixed_window(stream, start)) for start in starts]
    if summary:
        print(json.dumps({label: max(result[label] for result in results) for label in LABELS}))
    else:
        for result in results:
            print(json.dumps(result), flush=True)


def live_mode(predictor: Predictor, node_id: int | None) -> None:
    """Read NDJSON chunks from stdin; emit a scores object every ~0.5 s.

    Each input line must contain node_id and either decoded `samples` (float in
    [-1,1] or 16-bit PCM scale) or companded `codes` (uint8 values). This is the
    narrow bridge needed from the existing Thingy gateway's on_result handler.
    """
    audio = np.empty(0, dtype=np.float32)
    chosen = node_id
    for raw in sys.stdin:
        if not raw.strip():
            continue
        message = json.loads(raw)
        current = int(message["node_id"])
        if chosen is None:
            chosen = current
            print(f"Using microphone node {chosen}", file=sys.stderr)
        if current != chosen:
            continue
        if "codes" in message:
            chunk = decode_codes(message["codes"])
        else:
            chunk = np.asarray(message["samples"], dtype=np.float32)
            if chunk.size and float(np.max(np.abs(chunk))) > 2.0:
                chunk = chunk / 32768.0
        audio = np.concatenate((audio, chunk))
        while audio.size >= WINDOW_SAMPLES:
            print(json.dumps(predictor.scores(audio[:WINDOW_SAMPLES])), flush=True)
            audio = audio[HOP_SAMPLES:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "household_sound.pt")
    sub = parser.add_subparsers(dest="mode", required=True)
    file_parser = sub.add_parser("file", help="Classify a WAV/FLAC file")
    file_parser.add_argument("path", type=Path)
    file_parser.add_argument("--already-stream", action="store_true", help="Input is a decoded 4 kHz Thingy stream")
    file_parser.add_argument("--summary", action="store_true", help="One maximum score per label for the whole file")
    live_parser = sub.add_parser("live", help="Read decoded Thingy chunks as NDJSON from stdin")
    live_parser.add_argument("--node-id", type=int, default=None)
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint)
    if args.mode == "file":
        file_mode(predictor, args.path, args.already_stream, args.summary)
    else:
        live_mode(predictor, args.node_id)


if __name__ == "__main__":
    main()
