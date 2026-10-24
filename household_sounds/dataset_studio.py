"""Dataset Studio: Recording, interactive waveform annotation, and one-click training."""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

from household_sounds.model import LABELS, STREAM_RATE

REPO_ROOT = Path(__file__).resolve().parent.parent
CUSTOM_DATA_DIR = REPO_ROOT / "data" / "custom_dataset"
CUSTOM_AUDIO_DIR = CUSTOM_DATA_DIR / "audio"
CUSTOM_MANIFEST = CUSTOM_DATA_DIR / "manifest.jsonl"
CACHE_DIR = REPO_ROOT / "data" / "cache"


class DatasetStudioManager:
    """Manages audio takes, timing annotations, and dataset manifests."""

    def __init__(self, data_dir: Path = CUSTOM_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "audio"
        self.manifest_path = self.data_dir / "manifest.jsonl"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        self.is_recording: bool = False
        self.recording_id: str | None = None
        self.recording_source: str = "thingy"
        self.recording_sample_rate: int = 4031
        self._samples_buffer: list[np.ndarray] = []
        self._start_time: float = 0.0

        # Training task tracking
        self.training_task: asyncio.Task[None] | None = None
        self.training_status: dict[str, Any] = {
            "state": "idle",  # "idle", "training", "completed", "error"
            "message": "Ready",
            "progress_epoch": 0,
            "total_epochs": 10,
            "metrics": {},
        }

    def start_recording(self, source: str = "thingy", sample_rate: int = 4031) -> dict[str, Any]:
        """Start a new recording take."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_id = f"take_{source}_{timestamp}"
        self.recording_source = source
        self.recording_sample_rate = sample_rate
        self._samples_buffer = []
        self._start_time = time.monotonic()
        self.is_recording = True
        return {
            "ok": True,
            "recording_id": self.recording_id,
            "source": self.recording_source,
            "sample_rate": self.recording_sample_rate,
        }

    def push_audio(self, samples: np.ndarray, source: str = "thingy") -> None:
        """Push streaming audio samples into active recording take."""
        if not self.is_recording or source != self.recording_source:
            return
        if samples.ndim > 1:
            samples = samples.flatten()
        self._samples_buffer.append(samples.astype(np.float32))

    def stop_recording(self) -> dict[str, Any]:
        """Stop active recording take, resample to 16 kHz, save WAV, and generate waveform envelope."""
        if not self.is_recording or not self.recording_id:
            raise RuntimeError("No active recording take to stop")

        self.is_recording = False
        duration_s = time.monotonic() - self._start_time

        if not self._samples_buffer:
            raise ValueError("No audio samples collected during recording take")

        raw_audio = np.concatenate(self._samples_buffer)
        in_sr = self.recording_sample_rate

        # Resample to 16,000 Hz standard mono
        target_sr = 16000
        if in_sr != target_sr:
            # Resample cleanly using rational polyphase filter
            gcd = np.gcd(int(target_sr), int(round(in_sr)))
            up = int(target_sr // gcd)
            down = int(round(in_sr) // gcd)
            audio_16k = signal.resample_poly(raw_audio, up, down).astype(np.float32)
        else:
            audio_16k = raw_audio

        # Normalize clipping bounds
        max_val = float(np.max(np.abs(audio_16k)))
        if max_val > 0.01:
            audio_16k = audio_16k / max(1.0, max_val)

        # Write WAV file
        wav_path = self.audio_dir / f"{self.recording_id}.wav"
        int16_samples = np.clip(np.round(audio_16k * 32767.0), -32768, 32767).astype(np.int16)
        wavfile.write(str(wav_path), target_sr, int16_samples)

        actual_duration = float(len(audio_16k)) / target_sr

        # Compute 300-point downsampled waveform envelope for fast canvas display
        envelope = self._compute_waveform_envelope(audio_16k, points=300)

        return {
            "ok": True,
            "recording_id": self.recording_id,
            "filename": f"{self.recording_id}.wav",
            "duration_s": round(actual_duration, 2),
            "sample_rate": target_sr,
            "samples_count": len(audio_16k),
            "waveform": envelope,
        }

    def _compute_waveform_envelope(self, samples: np.ndarray, points: int = 300) -> list[float]:
        """Downsample audio waveform into min/max envelope points for UI rendering."""
        if len(samples) == 0:
            return [0.0] * points
        chunk_size = max(1, len(samples) // points)
        envelope: list[float] = []
        for i in range(points):
            chunk = samples[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk) > 0:
                peak = float(np.max(np.abs(chunk)))
                envelope.append(round(peak, 4))
            else:
                envelope.append(0.0)
        return envelope

    def list_recordings(self) -> list[dict[str, Any]]:
        """List all audio takes in the custom dataset library."""
        recordings: list[dict[str, Any]] = []
        if not self.audio_dir.exists():
            return []
        for p in sorted(self.audio_dir.glob("*.wav"), reverse=True):
            try:
                sr, data = wavfile.read(str(p))
                dur = float(len(data)) / float(sr)
                recordings.append({
                    "recording_id": p.stem,
                    "filename": p.name,
                    "duration_s": round(dur, 2),
                    "size_bytes": p.stat().st_size,
                    "created_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                })
            except Exception:
                continue
        return recordings

    def get_recording_waveform(self, filename: str) -> dict[str, Any]:
        """Retrieve audio waveform data for an existing recording."""
        wav_path = self.audio_dir / filename
        if not wav_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {filename}")

        sr, data = wavfile.read(str(wav_path))
        if data.dtype == np.int16:
            samples = data.astype(np.float32) / 32768.0
        else:
            samples = data.astype(np.float32)

        duration = float(len(samples)) / float(sr)
        envelope = self._compute_waveform_envelope(samples, points=300)
        return {
            "filename": filename,
            "duration_s": round(duration, 2),
            "sample_rate": sr,
            "waveform": envelope,
        }

    def annotate_segment(
        self,
        recording_id: str,
        start_s: float,
        end_s: float,
        labels: dict[str, int],
        split: str = "train",
    ) -> dict[str, Any]:
        """Add an annotated time segment with multi-label tags into the dataset manifest."""
        wav_path = self.audio_dir / f"{recording_id}.wav"
        if not wav_path.is_file():
            raise FileNotFoundError(f"Recording file not found: {wav_path.name}")

        sr, data = wavfile.read(str(wav_path))
        duration = float(len(data)) / float(sr)

        start_s = max(0.0, float(start_s))
        end_s = min(duration, float(end_s))
        if end_s <= start_s:
            raise ValueError("end_s must be strictly greater than start_s")

        seg_duration = end_s - start_s
        clean_labels = {l: int(labels.get(l, 0)) for l in LABELS}

        # Calculate candidate 1.0-second window starts
        # If segment is >= 1.0s, candidate start is start_s; if shorter, anchor start_s
        candidate_starts = [round(start_s, 2)]
        if seg_duration > 1.5:
            # Add midpoints for longer segments
            mid = round(start_s + (seg_duration - 1.0) / 2.0, 2)
            candidate_starts.append(mid)

        # Build segment entry
        seg_idx = int(time.time() * 1000) % 1000000
        segment_id = f"{recording_id}_seg{seg_idx}"

        manifest_entry = {
            "id": segment_id,
            "source_partition": "custom",
            "path": str(wav_path.resolve()),
            "labels": clean_labels,
            "candidate_starts": candidate_starts,
            "split": split,
            "meta": {
                "start_s": round(start_s, 2),
                "end_s": round(end_s, 2),
                "duration_s": round(seg_duration, 2),
                "annotated_at": datetime.now().isoformat(),
            },
        }

        # Append to manifest
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_entry) + "\n")

        # Precompute cache entry so future training runs need zero preprocessing
        self._precompute_cache(wav_path, segment_id, candidate_starts)

        return {
            "ok": True,
            "segment_id": segment_id,
            "recording_id": recording_id,
            "labels": clean_labels,
            "start_s": round(start_s, 2),
            "end_s": round(end_s, 2),
            "candidate_starts": candidate_starts,
        }

    def _precompute_cache(self, wav_path: Path, segment_id: str, candidate_starts: list[float]) -> None:
        """Precompute and save resampled Thingy-simulated mono audio to cache."""
        try:
            from household_sounds.model import read_mono, simulate_thingy_stream

            raw, sr = read_mono(str(wav_path))
            sim = simulate_thingy_stream(raw, sr)
            cache_file = CACHE_DIR / f"custom_{segment_id}.npy"
            np.save(cache_file, sim)
        except Exception as exc:
            print(f"[!] Warning: Failed to precompute cache for {segment_id}: {exc}", file=sys.stderr)

    def get_dataset_summary(self) -> dict[str, Any]:
        """Return dataset statistics, label distribution, and segments list."""
        if not self.manifest_path.is_file():
            return {
                "total_segments": 0,
                "label_counts": {l: 0 for l in LABELS},
                "split_counts": {"train": 0, "val": 0, "test": 0},
                "segments": [],
            }

        segments: list[dict[str, Any]] = []
        label_counts = {l: 0 for l in LABELS}
        split_counts = {"train": 0, "val": 0, "test": 0}

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    segments.append(entry)
                    split = entry.get("split", "train")
                    split_counts[split] = split_counts.get(split, 0) + 1
                    for l, val in entry.get("labels", {}).items():
                        if val == 1 and l in label_counts:
                            label_counts[l] += 1
                except Exception:
                    continue

        return {
            "total_segments": len(segments),
            "label_counts": label_counts,
            "split_counts": split_counts,
            "segments": segments[::-1],  # newest first
        }

    def delete_segment(self, segment_id: str) -> bool:
        """Remove an annotated segment from the manifest."""
        if not self.manifest_path.is_file():
            return False

        lines = self.manifest_path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        found = False
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("id") == segment_id:
                    found = True
                    # Remove cached file if present
                    cache_file = CACHE_DIR / f"custom_{segment_id}.npy"
                    if cache_file.is_file():
                        cache_file.unlink(missing_ok=True)
                    continue
                new_lines.append(line)
            except Exception:
                new_lines.append(line)

        if found:
            self.manifest_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        return found

    def build_combined_manifest(self, base_manifest: Path | None = None) -> Path:
        """Merge custom dataset manifest with base fusion manifest."""
        if base_manifest is None:
            base_manifest = REPO_ROOT / "data" / "manifest_fusion.jsonl"

        combined_manifest = REPO_ROOT / "data" / "manifest_combined.jsonl"
        lines = []

        if base_manifest.is_file():
            lines.extend(base_manifest.read_text(encoding="utf-8").splitlines())

        if self.manifest_path.is_file():
            custom_lines = self.manifest_path.read_text(encoding="utf-8").splitlines()
            # Ensure balanced split for custom lines (80% train, 20% val)
            for idx, c_line in enumerate(custom_lines):
                if not c_line.strip():
                    continue
                try:
                    obj = json.loads(c_line)
                    obj["split"] = "val" if (idx % 5 == 0) else "train"
                    lines.append(json.dumps(obj))
                except Exception:
                    lines.append(c_line)

        combined_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return combined_manifest

    async def train_model(self, epochs: int = 10, on_complete: Any = None) -> None:
        """Run online fine-tuning on the combined dataset."""
        if self.training_status["state"] == "training":
            raise RuntimeError("A training run is already in progress")

        combined_manifest = self.build_combined_manifest()
        output_ckpt = REPO_ROOT / "artifacts" / "household_sound.pt"

        self.training_status = {
            "state": "training",
            "message": f"Training on CPU for {epochs} epochs...",
            "progress_epoch": 0,
            "total_epochs": epochs,
            "metrics": {},
        }

        cmd = [
            sys.executable,
            "-m",
            "household_sounds.train",
            "--manifest",
            str(combined_manifest),
            "--loss",
            "asl",
            "--epochs",
            str(epochs),
            "--batch-size",
            "32",
            "--output",
            str(output_ckpt),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            while True:
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if "epoch " in line:
                    try:
                        # Parse epoch NN: loss=... val_macro_f1=...
                        parts = line.split(":")
                        ep_num = int(parts[0].replace("epoch", "").strip())
                        self.training_status["progress_epoch"] = ep_num
                        self.training_status["message"] = line
                    except Exception:
                        pass

            rc = await process.wait()
            if rc == 0:
                self.training_status["state"] = "completed"
                self.training_status["message"] = f"Training completed successfully ({epochs} epochs)!"
                if on_complete is not None:
                    on_complete()
            else:
                self.training_status["state"] = "error"
                self.training_status["message"] = f"Training exited with code {rc}"

        except Exception as exc:
            self.training_status["state"] = "error"
            self.training_status["message"] = f"Training failed: {exc}"
