"""Local browser microphone demo for household sound classification with CPU and NPU controls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response

from household_sounds.predict import Predictor
from household_sounds.web.streaming import AudioSession, classify_browser_window

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
REPO_ROOT = PACKAGE_DIR.parent.parent

app = FastAPI(title="Household Sound Classifier Demo")
predictor = Predictor()

ASSET_TYPES = {
    "app.js": "text/javascript",
    "mic-worklet.js": "text/javascript",
    "style.css": "text/css",
}


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(
        (STATIC_DIR / "index.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/static/{name}")
async def static_asset(name: str) -> Response:
    if name not in ASSET_TYPES:
        raise HTTPException(status_code=404)
    return Response(
        (STATIC_DIR / name).read_bytes(),
        media_type=ASSET_TYPES[name],
        headers={"Cache-Control": "no-store"},
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ready",
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


def get_lan_ip() -> str:
    """Detect local LAN IP address."""
    import socket

    for target in ("192.168.178.1", "192.168.1.1", "10.153.58.1", "8.8.8.8"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((target, 80))
            ip = s.getsockname()[0]
            s.close()
            if not ip.startswith("127."):
                return ip
        except Exception:
            continue
    return "127.0.0.1"


def ensure_ssl_certificates(ssl_dir: Path, lan_ip: str) -> tuple[Path, Path]:
    """Ensure SSL certificates exist, automatically generating self-signed pair if needed."""
    import subprocess

    ssl_dir.mkdir(parents=True, exist_ok=True)
    cert_path = ssl_dir / "cert.pem"
    key_path = ssl_dir / "key.pem"

    if cert_path.is_file() and key_path.is_file():
        return cert_path, key_path

    print(f"[*] Generating self-signed SSL certificates for {lan_ip} & localhost...")
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", str(key_path),
        "-out", str(cert_path),
        "-days", "365",
        "-subj", "/CN=HouseholdSoundsClassifier",
        "-addext", f"subjectAltName=DNS:localhost,IP:127.0.0.1,IP:{lan_ip}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[+] SSL certificates created in {ssl_dir}/")
    except Exception as exc:
        print(f"[!] Warning: Failed to generate certificates with openssl ({exc}).")
        raise

    return cert_path, key_path


def main() -> None:
    """CLI launcher for the web demonstration."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Household Sound Classifier Web Server")
    parser.add_argument("--host", default=None, help="Host interface (e.g. 127.0.0.1 or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--network", "--lan",
        action="store_true",
        help="Make server accessible across local network (binds 0.0.0.0 and enables auto-SSL)",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        help="Enable HTTPS using automatically generated self-signed certificate",
    )
    parser.add_argument("--ssl-certfile", default=None, help="Path to custom SSL certificate")
    parser.add_argument("--ssl-keyfile", default=None, help="Path to custom SSL private key")
    parser.add_argument("--reload", action="store_true", help="Enable automatic code reloading")
    args = parser.parse_args()

    lan_ip = get_lan_ip()
    ssl_dir = REPO_ROOT / "ssl"

    # If --network is specified, default host is 0.0.0.0 and SSL is automatically enabled
    if args.network:
        host = args.host or "0.0.0.0"
        use_ssl = True
    else:
        host = args.host or "127.0.0.1"
        use_ssl = bool(args.ssl or (args.ssl_certfile and args.ssl_keyfile))

    certfile = args.ssl_certfile
    keyfile = args.ssl_keyfile

    if use_ssl and not (certfile and keyfile):
        cert_p, key_p = ensure_ssl_certificates(ssl_dir, lan_ip)
        certfile = str(cert_p)
        keyfile = str(key_p)

    protocol = "https" if certfile and keyfile else "http"

    print("=" * 68)
    print(f" Household Sound Classifier Web Server")
    if host == "0.0.0.0":
        print(f"   Local URL:    {protocol}://localhost:{args.port}")
        print(f"   Network URL:  {protocol}://{lan_ip}:{args.port}")
        if protocol == "https":
            print(f"   Note: Accept the self-signed certificate on your mobile browser.")
    else:
        print(f"   Server URL:   {protocol}://{host}:{args.port}")
    print("=" * 68)

    kwargs: dict[str, Any] = {
        "host": host,
        "port": args.port,
        "reload": args.reload,
    }
    if certfile and keyfile:
        kwargs["ssl_certfile"] = certfile
        kwargs["ssl_keyfile"] = keyfile

    uvicorn.run("household_sounds.web.app:app", **kwargs)


if __name__ == "__main__":
    main()
