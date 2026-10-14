# Browser microphone test

This separate FastAPI app shows the five live scores from `../household_sound_model/artifacts/household_sound.pt` using the system microphone through your web browser.

## Docker Compose (recommended)

From this folder:

```bash
docker compose up --build -d
```

Open <http://127.0.0.1:8000> and click **Start microphone**. The browser captures the mic; the container receives audio through the WebSocket, so there is no sound-device passthrough to configure.

```bash
docker compose logs -f       # watch startup and errors; Ctrl+C leaves the app running
docker compose down          # stop and remove the app container
```

The first build downloads a Python base image and CPU PyTorch, so it may take several minutes. Later `docker compose up -d` starts the existing image. After app-code changes, use `docker compose up --build -d`; after replacing the checkpoint, use `docker compose up -d --force-recreate` to reload its read-only file mount. Compose binds port 8000 only to `127.0.0.1`.

The image build context contains only this app folder. The selected checkpoint and two model Python files are mounted read-only from `../household_sound_model/`. The 1.2 GB public training audio and local `.venv` are not copied into the image. Docker Compose serves the trained model; it does not train it. See [the training explanation](../household_sound_model/TRAINING_EXPLAINED.md).

## Run without Docker

From this folder:

```bash
../household_sound_model/.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, click **Start microphone**, and allow microphone access. Use **Stop** to release the mic. If the web packages are missing, install this folder's `requirements.txt` into the existing model environment:

```bash
uv pip install --python ../household_sound_model/.venv/bin/python -r requirements.txt
```

The browser sends mono Float32 PCM chunks through `/ws/audio`. The server builds overlapping one-second windows, shapes each window like the Thingy:53 stream, and returns one JSON object with `voices`, `machines`, `collisions`, `steps`, and `music` scores about every 0.5 seconds. The first score needs roughly one second of audio. The page shows bars and the raw JSON. Nothing in this app writes microphone audio to disk.

The app binds to `127.0.0.1` in both commands above. Browser microphone capture needs a secure context; browsers treat local `localhost`/`127.0.0.1` as eligible local origins. For another device to open the page over a network, use HTTPS and an explicit host setup. See [MDN's microphone access documentation](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia) and [AudioWorklet documentation](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet).

## What this tests

This is a quick interface and audio-path test. It uses the browser microphone and the current CPU PyTorch checkpoint. Browser audio processing, microphone response, and room acoustics differ from the Nordic Thingy:53 stream; the numbers are not calibrated room probabilities or a Hailo NPU result. The model's held-out results and limits are in `../household_sound_model/RUN_REPORT.md`.

The HTTP routes, JavaScript syntax, and a direct ASGI WebSocket exchange with real audio were checked locally. `docker compose config` passed. This workspace cannot access the Docker daemon, so a container build and browser microphone run still need to be checked from your terminal.
