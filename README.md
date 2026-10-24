# Household Sounds Classifier

Real-time, ultra-compact edge audio processing pipeline and multi-label convolutional neural network designed to continuously detect household sounds from a Nordic Thingy:53 microphone over Bluetooth Low Energy (BLE).

The model continuously classifies 3 orthogonal, physically grounded acoustic categories:
1. **`voices`** (speech, conversation, vocal formants, baby crying)
2. **`machines`** (appliances, fans, microwaves, vacuum cleaners, washing machines)
3. **`impacts`** (footsteps, walking, door knocks, dropped objects, glass breaking)

> [!TIP]
> **Detailed Model & Architectural Walkthrough:**
> For an in-depth explanation with diagrams of the neural network layers, DSP filters, STFT spectrograms, and training math, see [**`docs/Household Sound Model.md`**](docs/Household%20Sound%20Model.md).

---

## Key Highlights

- **Ultra-Compact Footprint:** Only **79,683 parameters** (323.6 KB PyTorch, 313.1 KB ONNX opset 15, $< 1.8\text{ MB}$ RAM).
- **Sub-1ms Latency:** $\approx 1.0\text{ ms}$ inference on Raspberry Pi 5 CPU; instantaneous on Hailo-8 NPU.
- **Dual Statistical Pooling:** Combines Adaptive Average Pooling (formants/drones) and Adaptive Max Pooling (transient attack peaks).
- **Embedded Audio Pipeline:** Nordic Thingy:53 PDM microphone (16.1 kHz) $\to$ 4:1 FIR decimation $\to$ 8-bit non-linear companding $\to$ BLE streaming at 28 packets/s.
- **Ambient Noise & Music Immunity:** Stationarity-aware AGC with crest-factor analysis and 80 Hz sub-bass filtering rejects quiet room hiss and music false alarms.
- **Multi-Sound Superposition:** Loudness-calibrated Mixup 2.0 training enables simultaneous co-occurring triggers (e.g., speech + footsteps).

---

## Repository Structure

```text
.
├── pyproject.toml              # Unified dependencies, build system & CLI console scripts
├── README.md                   # Project landing page & quickstart guide
├── artifacts/                  # Production models & compiled binaries
│   ├── household_sound.pt      # Active PyTorch checkpoint (79,683 params, 323.6 KB)
│   ├── household_sound.onnx    # Validated ONNX opset 15 graph (313.1 KB)
│   ├── household_sound.alls    # Hailo model compilation script
│   └── household_sound.metrics.json # Production validation & test evaluation metrics
├── data/                       # Dataset manifests & caches (gitignored)
│   ├── manifest_combined.jsonl # Complete training manifest (FSD50K + ESC-50 + Custom)
│   ├── manifest_fusion.jsonl   # Clean curated fusion manifest
│   └── manifest.jsonl          # Base public dataset manifest
├── docs/                       # Illustrated documentation & figures
│   ├── Household Sound Model.md # Comprehensive illustrated architectural deep dive
│   ├── generate_figures.py     # Script to reproduce architectural PNG diagrams
│   └── assets/                 # Generated spectrograms and architecture diagrams
│       ├── model_pipeline.png
│       ├── model_architecture.png
│       ├── first_convolution.png
│       └── spectrogram_examples.png
├── household_sounds/           # Core installable Python package
│   ├── __init__.py             # Exports SoundCNN, Predictor, and constants
│   ├── model.py                # SoundCNN architecture, STFT, and DSP FIR filters
│   ├── predict.py              # Stationarity-aware AGC, high-pass filter & inference engine
│   ├── train.py                # Superposition Mixup 2.0 training with Asymmetric Loss (ASL)
│   ├── dataset.py              # FSD50K data fetcher & dataset loader
│   ├── build_fusion_manifest.py# Multi-dataset fusion curator (ESC-50 + FSD50K)
│   ├── thingy_bridge.py        # Asynchronous Bleak BLE manager for Nordic Thingy:53
│   ├── dataset_studio.py       # In-situ recording take manager & fine-tuning engine
│   ├── hailo/                  # Hailo NPU compilation toolchain
│   │   ├── export_onnx.py      # PyTorch -> ONNX export & numerical verification
│   │   ├── calibration.py      # INT8 PTQ calibration dataset generator
│   │   └── compile.py          # Compile ONNX -> HEF
│   └── web/                    # Real-time WebSocket streaming server & Web UI
│       ├── app.py              # FastAPI application with REST & WebSocket endpoints
│       └── static/             # Responsive HTML5/Canvas UI with live threshold tuning
└── tests/                      # Automated regression test suite (20/20 passed)
```

---

## Quickstart with `uv`

All commands run from the project root using [`uv`](https://docs.astral.sh/uv/):

### 1. Launch the Live Web Dashboard & BLE Streamer

```bash
# Launch server with automatic Thingy:53 BLE connection on port 8000:
uv run sound-web --ble --port 8000

# Or expose across local Wi-Fi with auto-generated SSL (for mobile browser mic):
uv run sound-web --network --ble --port 8000
```

Navigate to **`http://localhost:8000`** in your browser to inspect live spectrogram waveforms, real-time prediction bars, and tune sensitivity thresholds interactively.

### 2. Run Automated Regression Tests

```bash
uv run pytest tests/ -v
```

### 3. Test File Inference (CPU or Hailo NPU)

```bash
# CPU mode:
uv run sound-predict file path/to/recording.wav --summary

# Hailo NPU mode:
uv run sound-predict --backend npu file path/to/recording.wav --summary
```

### 4. Export ONNX Graph & Validate

```bash
uv run python household_sounds/hailo/export_onnx.py
```

### 5. Retrain Model with Superposition & Noise Immunity

```bash
uv run sound-train --epochs 25 --output artifacts/household_sound.pt
```

---

## Production Test Metrics

| Sound Class | Test Precision | Test Recall | **Test $F_1$ Score** | Default Sensitivity Cutoff |
| :--- | :--- | :--- | :--- | :--- |
| **`voices`** | **91.7%** | 83.0% | **0.8713** | **62%** |
| **`machines`** | 53.8% | 29.8% | **0.5243** | **68%** |
| **`impacts`** | 61.0% | **89.7%** | **0.7262** | **51%** |
| **Macro Average**| **68.8%** | **67.5%** | **0.7073** | — |

- **Simultaneous Superposition:** Voice + Knock triggers both categories concurrently.
- **Ambient Noise & Music Immunity:** 88.2% rejection rate on diverse musical instruments; 1.2x AGC gain cap on quiet room hiss.

---

## Deep Dive Architecture & Documentation

For the full illustrated technical reference, including layer equations, parameter budgets, and receptive field proofs, see:

👉 [**`docs/Household Sound Model.md`**](docs/Household%20Sound%20Model.md)
