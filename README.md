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
├── pyproject.toml              # Project dependencies and uv configuration
├── README.md                   # This overview and quickstart guide
├── household_sound_model/      # Core ML pipeline, training, Hailo export & inference
│   ├── audio_model.py          # DSP preprocessing, Thingy stream simulation & CNN
│   ├── predict.py              # Unified inference engine (CPU + Hailo-10H NPU)
│   ├── export_onnx.py          # Exports PyTorch checkpoint to ONNX with validation
│   ├── prepare_calibration.py  # Generates calibration dataset for INT8 PTQ quantization
│   ├── compile_hef.py          # Hailo DFC compiler script (SDK & CLI modes)
│   ├── compile_pipeline.sh     # One-step pipeline script: PT -> ONNX -> CALIB -> HEF
│   ├── train.py                # Multi-label CNN training script
│   ├── dataset.py              # Reproducible FSD50K subset selector and downloader
│   ├── artifacts/              # Production models, configs, and metrics
│   │   ├── household_sound.pt  # Active PyTorch checkpoint (324 KB)
│   │   ├── household_sound.onnx# Exported static ONNX graph (313 KB)
│   │   ├── household_sound.alls# Hailo model script
│   │   ├── household_sound.metrics.json
│   │   └── archive/            # Historical experiment candidates (v1, v2, v3)
│   ├── data/
│   │   └── manifest.jsonl      # Reproducible dataset manifest with licenses
│   ├── docs/                   # Mathematical diagrams and obsidian documentation
│   └── RUN_REPORT.md           # Measured metrics, test splits, and run notes
└── browser_mic_test/           # Live browser microphone demonstration
    ├── app.py                  # FastAPI server with WebSocket audio streaming & backend controls
    ├── streaming.py            # Audio session buffering and sliding-window feeder
    └── static/                 # Clean, responsive Web UI with CPU/NPU switching & telemetry
```

---

## Quickstart with `uv`

This repository uses [`uv`](https://docs.astral.sh/uv/) for fast, deterministic dependency management.

### 1. Hailo-10H HEF Compilation Pipeline

To export the model and prepare it for the Hailo-10H NPU:

```bash
# Run the complete 3-step pipeline (Export ONNX -> Generate Calibration Data -> Compile HEF)
./household_sound_model/compile_pipeline.sh hailo10h
```

Individual steps:
1. **Export to ONNX**:
   ```bash
   uv run python household_sound_model/export_onnx.py
   ```
   Validates model graph structure and checks numerical equivalence against PyTorch using ONNX Runtime.
2. **Generate Calibration Dataset**:
   ```bash
   uv run python household_sound_model/prepare_calibration.py --num-samples 128
   ```
   Saves real spectrogram slices (`artifacts/calib_set.npy`) for INT8 Post-Training Quantization (PTQ).
3. **Compile HEF**:
   ```bash
   uv run python household_sound_model/compile_hef.py --hw-arch hailo10h
   ```
   Compiles via Hailo Dataflow Compiler (DFC) Python API or Hailo CLI tools (`hailo parser onnx`, `hailo optimize`, `hailo compiler`).

### 2. Test Model Inference (CPU or NPU)

Run prediction on a dummy signal or file from anywhere in the repository:

```bash
uv run --python household_sound_model/.venv/bin/python python household_sound_model/predict.py --help
```

To run inference on an audio file:

```bash
# CPU mode:
uv run --python household_sound_model/.venv/bin/python python household_sound_model/predict.py file path/to/sound.wav --summary

# Hailo NPU mode:
uv run --python household_sound_model/.venv/bin/python python household_sound_model/predict.py --backend npu file path/to/sound.wav --summary
```

### 3. Run the Browser Microphone Demo

Launch the local FastAPI streaming server:

```bash
uv run --python household_sound_model/.venv/bin/python python -m uvicorn browser_mic_test.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> in your browser, click **Start microphone**, and observe real-time classification bars and backend telemetry.

### 4. Run on the Raspberry Pi 5 (with HTTPS for Browser Mic)

On the Raspberry Pi 5, the model runs natively via `uv` (~3.4 ms latency). To allow browser microphone access over the local network, run with the generated SSL certificate:

```bash
cd ~/HouseholdSoundsClassifier/browser_mic_test
nohup ../.venv/bin/python -m uvicorn app:app \
  --host 0.0.0.0 --port 8000 \
  --ssl-keyfile ~/HouseholdSoundsClassifier/ssl/key.pem \
  --ssl-certfile ~/HouseholdSoundsClassifier/ssl/cert.pem &
```

Then visit **`https://pi-team5.local:8000`** in your browser.

---

## Model Retraining & Dataset Management

1. **Inspect dataset status**:
   ```bash
   uv run --python household_sound_model/.venv/bin/python python household_sound_model/dataset.py status
   ```
2. **Download required subset clips**:
   ```bash
   uv run --python household_sound_model/.venv/bin/python python household_sound_model/dataset.py download --max-gb 5
   ```
3. **Train a new candidate model**:
   ```bash
   uv run --python household_sound_model/.venv/bin/python python household_sound_model/train.py --epochs 25 --output household_sound_model/artifacts/candidate.pt
   ```

Detailed signal processing math and training explanations can be found in [`household_sound_model/TRAINING_EXPLAINED.md`](household_sound_model/TRAINING_EXPLAINED.md).
