# Household Sounds Classifier

Real-time audio processing pipeline and compact multi-label neural network designed to continuously detect household sounds from a single microphone channel.

The model scores 5 independent categories simultaneously:
1. **Voices** (speech, conversation, singing)
2. **Machines** (appliances, fans, microwaves, power tools)
3. **Collisions** (impacts, dropped items, door slams)
4. **Steps** (footsteps, walking)
5. **Music** (instrumental or music playback)

---

## Repository Structure

```text
.
├── pyproject.toml              # Unified dependencies, build system & CLI console scripts
├── README.md                   # Project overview and quickstart guide
├── .gitignore                  # Single consolidated git ignore
├── artifacts/                  # Production models & compiled binaries
│   ├── household_sound.pt      # Active PyTorch checkpoint (324 KB)
│   ├── household_sound.onnx    # Validated ONNX graph (313 KB)
│   ├── household_sound.alls    # Hailo model script
│   ├── household_sound.hef     # Hailo compiled binary (when compiled)
│   ├── household_sound.metrics.json
│   └── archive/                # Historical experiment candidates (v1, v2, v3)
├── data/                       # Dataset manifests & caches (gitignored)
│   ├── manifest.jsonl          # Reproducible dataset manifest with licenses
│   └── cache/                  # Preprocessed Thingy-stream audio cache
├── docs/                       # Project documentation, figures, math notes
│   ├── assets/                 # Architecture figures and spectrogram plots
│   ├── TRAINING_EXPLAINED.md   # Mathematical derivations and DSP explanations
│   └── RUN_REPORT.md           # Benchmark and evaluation report
└── household_sounds/           # Single standard installable Python package
    ├── __init__.py             # Exports SoundCNN, Predictor, and constants
    ├── model.py                # SoundCNN architecture & DSP preprocessing
    ├── predict.py              # Dual CPU (PyTorch) / NPU (Hailo) inference engine
    ├── dataset.py              # FSD50K downloader & dataset loader
    ├── train.py                # Multi-label CNN training script
    ├── hailo/                  # Hailo-10H NPU compilation toolchain
    │   ├── export_onnx.py      # PyTorch -> ONNX export & verification
    │   ├── calibration.py      # INT8 PTQ calibration dataset generator
    │   ├── compile.py          # Compile ONNX -> HEF (SDK / CLI / Docker)
    │   └── pipeline.sh         # 1-step compilation pipeline
    └── web/                    # Browser microphone test & streaming server
        ├── app.py              # FastAPI server with WebSocket streaming
        ├── streaming.py        # Stream buffering & windowing
        └── static/             # Clean web frontend (HTML/CSS/JS/Worklet)
```

---

## Quickstart with `uv`

All commands are run from the project root using [`uv`](https://docs.astral.sh/uv/):

### 1. Run the Web Interface (Browser Mic or Thingy:53 Stream)

```bash
# Localhost:
uv run sound-web

# Or accessible across your local Wi-Fi / network (with auto-generated SSL):
uv run sound-web --network
```

The web UI provides an **Input Source** selector:
- **🎙️ Browser Mic**: Real-time microphone capture through your phone or laptop.
- **📡 Thingy:53 Stream**: Live classification of audio streamed from Nordic Thingy:53 hardware.

#### Linking with `thingy-ranging`

Run `gateway.py` with the forwarder flag pointing to `sound-web`:
```bash
# In the thingy-ranging repository (run on port 8080 to avoid collision):
uv run python tools/gateway.py --port 8080 --classifier-url ws://127.0.0.1:8000/ws/thingy
```

Or use the standalone bridge tool:
```bash
# Forward stdin chunks:
your_stream_command | uv run sound-bridge stdin

# Or replay a simulated stream for testing:
uv run sound-bridge replay --sound tone --duration 10
```

### 2. Test Model Inference (CPU or Hailo NPU)

```bash
# CPU mode:
uv run sound-predict file path/to/sound.wav --summary

# Hailo NPU mode:
uv run sound-predict --backend npu file path/to/sound.wav --summary
```

### 3. Hailo-10H HEF Compilation Pipeline

Compile the PyTorch checkpoint into a Hailo-10H `.hef` file:

```bash
# Run the complete pipeline (Export ONNX -> Generate Calibration Data -> Compile HEF)
./household_sounds/hailo/pipeline.sh hailo10h
```

Individual commands:
- **Export to ONNX**: `uv run sound-export`
- **Generate Calibration Dataset**: `uv run python -m household_sounds.hailo.calibration --num-samples 128`
- **Compile to HEF**: `uv run sound-compile --hw-arch hailo10h`

---

## Running on the Raspberry Pi 5 (with HTTPS for Browser Mic)

On the Raspberry Pi 5, simply run:

```bash
uv run sound-web --network
```

Then navigate to **`https://pi-team5.local:8000`** in your browser.

---

## Dataset & Training Commands

```bash
# Inspect dataset status:
uv run sound-dataset status

# Download required subset clips:
uv run sound-dataset download --max-gb 5

# Train a new candidate model:
uv run sound-train --epochs 25 --output artifacts/candidate.pt
```

For mathematical derivations of the DSP filters, STFT windowing, and companding, see [`docs/TRAINING_EXPLAINED.md`](docs/TRAINING_EXPLAINED.md).
