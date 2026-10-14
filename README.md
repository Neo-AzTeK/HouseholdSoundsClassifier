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
├── household_sound_model/      # Core ML pipeline, training & inference
│   ├── audio_model.py          # DSP preprocessing, Thingy stream simulation & CNN
│   ├── predict.py              # File and live NDJSON prediction CLI
│   ├── train.py                # Multi-label CNN training script
│   ├── dataset.py              # Reproducible FSD50K subset selector and downloader
│   ├── artifacts/              # Production checkpoint and evaluation metrics
│   │   ├── household_sound.pt  # Active model checkpoint (324 KB)
│   │   ├── household_sound.metrics.json
│   │   └── archive/            # Historical experiment candidates (v1, v2, v3)
│   ├── data/
│   │   └── manifest.jsonl      # Reproducible dataset manifest with licenses
│   ├── docs/                   # Mathematical diagrams and obsidian documentation
│   └── RUN_REPORT.md           # Measured metrics, test splits, and run notes
└── browser_mic_test/           # Live browser microphone demonstration
    ├── app.py                  # FastAPI server with WebSocket audio streaming
    ├── streaming.py            # Audio session buffering and sliding-window feeder
    ├── compose.yaml            # Docker Compose service definition
    └── static/                 # Browser audio worklet & real-time visualizer
```

---

## Quickstart with `uv`

This repository uses [`uv`](https://docs.astral.sh/uv/) for fast, deterministic dependency management.

### 1. Test Model Inference

Run prediction on a dummy signal or file from anywhere in the repository:

```bash
uv run --python household_sound_model/.venv/bin/python python household_sound_model/predict.py --help
```

To run inference on an audio file:

```bash
uv run --python household_sound_model/.venv/bin/python python household_sound_model/predict.py file path/to/sound.wav --summary
```

### 2. Run the Browser Microphone Demo

Launch the local FastAPI streaming server:

```bash
uv run --python household_sound_model/.venv/bin/python python -m uvicorn browser_mic_test.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> in your browser, click **Start microphone**, and observe real-time classification bars.

### 3. Run with Docker Compose

If you prefer running the web interface in an isolated container:

```bash
docker compose -f browser_mic_test/compose.yaml up -d --build
```

Then visit <http://127.0.0.1:8000>.

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
