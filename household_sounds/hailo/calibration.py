"""Prepare representative calibration data for Hailo INT8 Post-Training Quantization (PTQ)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from household_sounds.model import (
    candidate_starts,
    fixed_window,
    read_mono,
    simulate_thingy_stream,
    spectrogram,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "calib_set.npy"
DEFAULT_CACHE = REPO_ROOT / "data" / "cache"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.jsonl"


def generate_calibration_data(
    output_path: Path = DEFAULT_OUTPUT,
    num_samples: int = 128,
    cache_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> Path:
    """Generate a balanced calibration dataset of spectrograms."""
    output_path = Path(output_path)
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE
    if manifest_path is None:
        manifest_path = DEFAULT_MANIFEST

    windows: list[np.ndarray] = []

    # Priority 1: Use cached audio files if available
    cached_files = sorted(list(cache_dir.glob("*.npy")))
    if cached_files:
        print(f"[*] Found {len(cached_files)} cached audio clips in {cache_dir}")
        step = max(1, len(cached_files) // num_samples)
        selected_files = cached_files[::step][:num_samples]
        for f in selected_files:
            try:
                audio = np.load(f)
                starts = candidate_starts(audio)
                win = fixed_window(audio, starts[0])
                spec = spectrogram(win)
                windows.append(spec)
            except Exception as e:
                print(f"[!] Warning: failed to load {f.name}: {e}")

    # Priority 2: Load from manifest if cache is empty
    if len(windows) < num_samples and manifest_path.is_file():
        print(f"[*] Loading additional clips from {manifest_path}...")
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(windows) >= num_samples:
                    break
                row = json.loads(line)
                audio_file = REPO_ROOT / row["path"]
                if not audio_file.is_file():
                    continue
                try:
                    audio, sample_rate = read_mono(str(audio_file))
                    stream = simulate_thingy_stream(audio, sample_rate)
                    starts = candidate_starts(stream)
                    win = fixed_window(stream, starts[0])
                    spec = spectrogram(win)
                    windows.append(spec)
                except Exception as e:
                    print(f"[!] Warning: failed to process {audio_file}: {e}")

    # Fallback: Synthetic spectrograms
    if not windows:
        print("[*] Generating synthetic calibration spectrograms (fallback)...")
        np.random.seed(42)
        for _ in range(num_samples):
            t = np.linspace(0, 1.0, 4031, dtype=np.float32)
            sig = 0.1 * np.sin(2 * np.pi * np.random.uniform(100, 1500) * t)
            sig += 0.05 * np.random.normal(0, 0.02, size=len(t)).astype(np.float32)
            windows.append(spectrogram(sig))

    calib_array = np.stack(windows[:num_samples], axis=0).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, calib_array)

    nhwc_path = output_path.with_name(output_path.stem + "_nhwc.npy")
    calib_nhwc = np.transpose(calib_array, (0, 2, 3, 1))
    np.save(nhwc_path, calib_nhwc)

    print(f"[+] Calibration set saved:")
    print(f"    - NCHW: {output_path} (shape: {calib_array.shape}, dtype: {calib_array.dtype})")
    print(f"    - NHWC: {nhwc_path} (shape: {calib_nhwc.shape}, dtype: {calib_nhwc.dtype})")
    print(f"    - Min: {calib_array.min():.4f}, Max: {calib_array.max():.4f}, Mean: {calib_array.mean():.4f}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=128)
    args = parser.parse_args()

    generate_calibration_data(output_path=args.output, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
