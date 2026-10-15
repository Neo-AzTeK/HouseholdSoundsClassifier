"""Inference engine supporting CPU (PyTorch) and Hailo NPU (HailoRT) backends."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from household_sounds.model import (
    HOP_SAMPLES,
    LABELS,
    PREPROCESSING,
    SILENCE_RMS,
    STREAM_RATE,
    WINDOW_SAMPLES,
    SoundCNN,
    fixed_window,
    read_mono,
    simulate_thingy_stream,
    spectrogram,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "household_sound.pt"
DEFAULT_HEF = REPO_ROOT / "artifacts" / "household_sound.hef"


class CpuBackend:
    """CPU inference backend powered by PyTorch."""

    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if tuple(checkpoint["labels"]) != LABELS:
            raise ValueError("Checkpoint labels do not match code")
        if checkpoint.get("preprocessing") != PREPROCESSING:
            raise ValueError("Checkpoint preprocessing does not match code")

        self.model = SoundCNN().eval()
        self.model.load_state_dict(checkpoint["state_dict"])
        self.thresholds = dict(zip(LABELS, checkpoint["thresholds"]))
        self.silence_rms = float(checkpoint.get("silence_rms", SILENCE_RMS))
        self.name = "CPU (PyTorch)"

    @torch.no_grad()
    def infer(self, features: np.ndarray) -> np.ndarray:
        """Run inference on spectrogram features shape (1, 129, 48)."""
        x = torch.from_numpy(features)[None, :, :, :]
        logits = self.model(x).numpy()[0]
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        return probs.astype(np.float32)


class NpuBackend:
    """Hailo-10H / Hailo-8 NPU inference backend using pyhailort."""

    def __init__(self, hef_path: Path) -> None:
        self.hef_path = hef_path
        self.name = "Hailo NPU (Hailo-10H)"
        self.is_ready = False
        self.error_message: str | None = None
        self._vdevice = None
        self._infer_model = None
        self._configured_model = None
        self._classic_network_group = None
        self._init_hailo()

    def _init_hailo(self) -> None:
        if not self.hef_path.is_file():
            self.error_message = f"HEF file not found: {self.hef_path.name}"
            return

        try:
            import hailo_platform as hp
            from hailo_platform import VDevice, HEF
        except ImportError:
            self.error_message = "pyhailort / hailo_platform is not installed"
            return

        try:
            self._vdevice = VDevice()
            if hasattr(self._vdevice, "create_infer_model"):
                self._infer_model = self._vdevice.create_infer_model(str(self.hef_path))
                self._infer_model.set_batch_size(1)
                self._configured_model = self._infer_model.configure()
                self.api_mode = "infer_model"
            else:
                from hailo_platform import ConfigureParams, HailoStreamInterface, InferVStreams
                hef = HEF(str(self.hef_path))
                config_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
                self._classic_network_group = self._vdevice.configure(hef, config_params)[0]
                self.api_mode = "vstreams"
            self.is_ready = True
        except Exception as exc:
            self.error_message = f"Failed to initialize Hailo device: {exc}"

    def infer(self, features: np.ndarray) -> np.ndarray:
        if not self.is_ready:
            raise RuntimeError(f"Hailo NPU not ready: {self.error_message}")

        batch_input = features[None, :, :, :].astype(np.float32)

        if self.api_mode == "infer_model":
            bindings = self._configured_model.create_bindings()
            input_names = list(bindings.input_names)
            output_names = list(bindings.output_names)
            bindings.input(input_names[0]).set_buffer(batch_input)
            output_buf = np.empty(bindings.output(output_names[0]).shape, dtype=np.float32)
            bindings.output(output_names[0]).set_buffer(output_buf)
            self._configured_model.wait_for_async_ready(timeout_ms=1000)
            self._configured_model.run([bindings], timeout_ms=1000)
            logits = output_buf.flatten()[:len(LABELS)]
        else:
            from hailo_platform import InferVStreams
            with InferVStreams(self._classic_network_group, self._classic_network_group.create_params()) as infer_pipeline:
                input_name = self._classic_network_group.get_input_vstream_infos()[0].name
                output_name = self._classic_network_group.get_output_vstream_infos()[0].name
                res = infer_pipeline.infer({input_name: batch_input})
                logits = res[output_name].flatten()[:len(LABELS)]

        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        return probs.astype(np.float32)


class Predictor:
    """Unified Predictor with dynamic runtime switching between CPU and Hailo NPU."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        hef_path: Path = DEFAULT_HEF,
        backend: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.hef_path = Path(hef_path)
        self.cpu = CpuBackend(self.checkpoint_path)
        self.npu = NpuBackend(self.hef_path)
        self.thresholds = dict(self.cpu.thresholds)
        self.silence_rms = self.cpu.silence_rms
        self.use_silence_gate = True

        self.active_backend_id = "cpu"
        if backend.lower() == "npu":
            if self.npu.is_ready:
                self.active_backend_id = "npu"
            else:
                print(f"[!] Warning: NPU requested but not ready ({self.npu.error_message}). Using CPU.")

    def set_backend(self, target: str) -> tuple[bool, str]:
        """Switch active backend to 'cpu' or 'npu'. Returns (success, message)."""
        target = target.lower().strip()
        if target == "cpu":
            self.active_backend_id = "cpu"
            return True, "Switched to CPU (PyTorch)"
        elif target == "npu":
            if not self.npu.is_ready:
                self.npu._init_hailo()
            if self.npu.is_ready:
                self.active_backend_id = "npu"
                return True, "Switched to Hailo NPU"
            else:
                return False, f"NPU unavailable: {self.npu.error_message}"
        else:
            return False, f"Unknown backend: {target}. Choose 'cpu' or 'npu'."

    def get_backends_info(self) -> dict[str, Any]:
        """Return diagnostic and availability status of all backends."""
        return {
            "active": self.active_backend_id,
            "silence_rms": self.silence_rms,
            "silence_gate": self.use_silence_gate,
            "thresholds": self.thresholds,
            "cpu": {
                "id": "cpu",
                "name": self.cpu.name,
                "available": True,
                "checkpoint": self.checkpoint_path.name,
            },
            "npu": {
                "id": "npu",
                "name": self.npu.name,
                "available": self.npu.is_ready,
                "hef_path": self.hef_path.name,
                "hef_exists": self.hef_path.is_file(),
                "error": self.npu.error_message,
            },
        }

    def predict(self, window: np.ndarray) -> dict[str, Any]:
        """Predict sound categories and return detailed metadata + latency."""
        window = fixed_window(window)
        rms = float(np.sqrt(np.mean(window * window)))

        if self.use_silence_gate and rms < self.silence_rms:
            return {
                "scores": {label: 0.0 for label in LABELS},
                "backend": self.cpu.name if self.active_backend_id == "cpu" else self.npu.name,
                "backend_id": self.active_backend_id,
                "latency_ms": 0.0,
                "rms": round(rms, 6),
                "silence": True,
            }

        features = spectrogram(window)

        t0 = time.perf_counter()
        if self.active_backend_id == "npu" and self.npu.is_ready:
            try:
                probs = self.npu.infer(features)
                backend_used = self.npu.name
                backend_id = "npu"
            except Exception as e:
                probs = self.cpu.infer(features)
                backend_used = f"{self.cpu.name} (fallback: {e})"
                backend_id = "cpu"
        else:
            probs = self.cpu.infer(features)
            backend_used = self.cpu.name
            backend_id = "cpu"
        latency_ms = (time.perf_counter() - t0) * 1000.0

        scores = {label: round(float(prob), 4) for label, prob in zip(LABELS, probs)}
        return {
            "scores": scores,
            "backend": backend_used,
            "backend_id": backend_id,
            "latency_ms": round(latency_ms, 2),
            "rms": round(rms, 6),
            "silence": False,
        }

    def scores(self, window: np.ndarray) -> dict[str, float]:
        """Backwards-compatible interface returning plain dictionary of scores."""
        return self.predict(window)["scores"]


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
    parser = argparse.ArgumentParser(description="Household sound classifier inference CLI")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hef", type=Path, default=DEFAULT_HEF)
    parser.add_argument("--backend", choices=["cpu", "npu"], default="cpu")
    sub = parser.add_subparsers(dest="mode", required=True)
    file_parser = sub.add_parser("file", help="Classify a WAV/FLAC file")
    file_parser.add_argument("path", type=Path)
    file_parser.add_argument("--already-stream", action="store_true", help="Input is a decoded 4 kHz Thingy stream")
    file_parser.add_argument("--summary", action="store_true", help="One maximum score per label for the whole file")
    live_parser = sub.add_parser("live", help="Read decoded Thingy chunks as NDJSON from stdin")
    live_parser.add_argument("--node-id", type=int, default=None)
    args = parser.parse_args()

    predictor = Predictor(checkpoint_path=args.checkpoint, hef_path=args.hef, backend=args.backend)
    if args.mode == "file":
        file_mode(predictor, args.path, args.already_stream, args.summary)
    else:
        live_mode(predictor, args.node_id)


if __name__ == "__main__":
    main()
