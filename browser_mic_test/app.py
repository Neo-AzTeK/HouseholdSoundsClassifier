"""Local browser microphone demo for the household sound model with CPU and NPU controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

try:
    from streaming import AudioSession, CHECKPOINT, HEF_FILE, Predictor, classify_browser_window
except ModuleNotFoundError:
    from browser_mic_test.streaming import AudioSession, CHECKPOINT, HEF_FILE, Predictor, classify_browser_window

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Household Sound Classifier Demo")
predictor = Predictor(checkpoint_path=CHECKPOINT, hef_path=HEF_FILE)

ASSET_TYPES = {
    "app.js": "text/javascript",
    "mic-worklet.js": "text/javascript",
    "style.css": "text/css",
}


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        (ROOT / "static" / "index.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/static/{name}")
async def static_asset(name: str) -> Response:
    if name not in ASSET_TYPES:
        raise HTTPException(status_code=404)
    return Response(
        (ROOT / "static" / name).read_bytes(),
        media_type=ASSET_TYPES[name],
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ready",
        "model": CHECKPOINT.name,
        "backends": predictor.get_backends_info(),
    }


@app.get("/api/backends")
async def get_backends() -> dict[str, Any]:
    return predictor.get_backends_info()


@app.post("/api/backend")
async def set_backend(request: Request) -> JSONResponse:
    data = await request.json()
    target = data.get("backend", "cpu")
    success, message = predictor.set_backend(target)
    return JSONResponse(
        {
            "success": success,
            "message": message,
            "backends": predictor.get_backends_info(),
        },
        status_code=200 if success else 400,
    )


@app.post("/api/config")
async def set_config(request: Request) -> dict[str, Any]:
    data = await request.json()
    if "silence_gate" in data:
        predictor.use_silence_gate = bool(data["silence_gate"])
    return predictor.get_backends_info()


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
        await websocket.send_json({
            "type": "ready",
            "sampleRate": rate,
            "backends": predictor.get_backends_info(),
        })

        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                disconnected = True
                break

            if message.get("bytes") is not None:
                for window in session.push_bytes(message["bytes"]):
                    res = classify_browser_window(predictor, window, rate)
                    await websocket.send_json(res)

            elif message.get("text") is not None:
                cmd = json.loads(message["text"])
                cmd_type = cmd.get("type")
                if cmd_type == "stop":
                    break
                elif cmd_type == "set_backend":
                    target = cmd.get("backend", "cpu")
                    ok, msg = predictor.set_backend(target)
                    await websocket.send_json({
                        "type": "backend_changed",
                        "success": ok,
                        "message": msg,
                        "backends": predictor.get_backends_info(),
                    })
                elif cmd_type == "set_silence_gate":
                    predictor.use_silence_gate = bool(cmd.get("enabled", True))
                    await websocket.send_json({
                        "type": "config_changed",
                        "silence_gate": predictor.use_silence_gate,
                    })

    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        disconnected = True
    finally:
        if not disconnected:
            await websocket.close()
