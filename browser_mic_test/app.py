"""Local browser microphone demo for the household sound model."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

try:
    from streaming import AudioSession, CHECKPOINT, Predictor, classify_browser_window
except ModuleNotFoundError:
    from browser_mic_test.streaming import AudioSession, CHECKPOINT, Predictor, classify_browser_window

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Household sound microphone test")
predictor = Predictor(CHECKPOINT)
ASSET_TYPES = {"app.js": "text/javascript", "mic-worklet.js": "text/javascript", "style.css": "text/css"}


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((ROOT / "static" / "index.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


@app.get("/static/{name}")
async def static_asset(name: str) -> Response:
    if name not in ASSET_TYPES:
        raise HTTPException(status_code=404)
    return Response((ROOT / "static" / name).read_bytes(), media_type=ASSET_TYPES[name],
                    headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ready", "model": CHECKPOINT.name}


@app.websocket("/ws/audio")
async def audio_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    disconnected = False
    try:
        start = await websocket.receive_json()
        if start.get("type") != "start":
            raise ValueError("first message must be a start request")
        rate = int(start["sampleRate"])
        session = AudioSession(rate)
        await websocket.send_json({"type": "ready", "sampleRate": rate})
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                disconnected = True
                break
            if message.get("bytes") is not None:
                for window in session.push_bytes(message["bytes"]):
                    scores = classify_browser_window(predictor, window, rate)
                    await websocket.send_json(scores)
            elif message.get("text") is not None:
                if json.loads(message["text"]).get("type") == "stop":
                    break
                raise ValueError("expected binary Float32 PCM audio")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        disconnected = True
    finally:
        if not disconnected:
            await websocket.close()
