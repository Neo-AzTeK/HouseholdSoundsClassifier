# Browser microphone test

This separate FastAPI app shows the five live scores from `../household_sound_model/artifacts/household_sound.pt` using the system microphone through your web browser.

## Run Locally

From this folder, using the Python environment:

```bash
../household_sound_model/.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>, click **Start microphone**, and allow microphone access. Use **Stop** to release the mic.

If the web packages are missing, install this folder's `requirements.txt` into the environment:

```bash
uv pip install --python ../household_sound_model/.venv/bin/python -r requirements.txt
```

---

## Run on Raspberry Pi 5 (over HTTPS)

Browsers require a **Secure Context** to grant microphone access (`navigator.mediaDevices.getUserMedia`) over a local network. On the Pi, run Uvicorn with the generated SSL certificate:

```bash
../.venv/bin/python -m uvicorn app:app \
  --host 0.0.0.0 --port 8000 \
  --ssl-keyfile ~/HouseholdSoundsClassifier/ssl/key.pem \
  --ssl-certfile ~/HouseholdSoundsClassifier/ssl/cert.pem
```

Then open **`https://pi-team5.local:8000`** in your browser, accept the self-signed certificate warning, and click **Start microphone**.

---

## How It Works

The browser sends mono Float32 PCM chunks through `/ws/audio`. The server builds overlapping one-second windows, shapes each window like the Thingy:53 stream, and returns one JSON object with `voices`, `machines`, `collisions`, `steps`, and `music` scores about every 0.5 seconds. The first score needs roughly one second of audio. The page shows bars and the raw JSON. Nothing in this app writes microphone audio to disk.
