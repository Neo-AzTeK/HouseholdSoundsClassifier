"""Bridge utility connecting Thingy:53 audio streams to the sound-web classifier.

Supports three modes:
1. stdin mode (default): Reads NDJSON stream chunks from stdin and forwards to sound-web over WebSocket.
2. replay mode: Streams simulated or recorded Thingy chunks at real-time 42 Hz cadence.
3. ble mode: Directly connects to a Thingy:53 gateway over Bluetooth Low Energy using Bleak.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

import numpy as np


async def forward_stdin(ws_url: str) -> None:
    """Read NDJSON lines from stdin and forward to sound-web WebSocket."""
    import websockets

    print(f"[*] Connecting to sound-web at {ws_url}...")
    try:
        async with websockets.connect(ws_url) as ws:
            print("[+] Connected! Forwarding stdin chunks to classifier (Ctrl+C to stop)...")
            loop = asyncio.get_running_loop()
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)

            pkts = 0
            t0 = time.time()
            while True:
                line = await reader.readline()
                if not line:
                    break
                text = line.decode("utf-8").strip()
                if not text:
                    continue
                await ws.send(text)
                pkts += 1
                if pkts % 50 == 0:
                    dt = time.time() - t0
                    print(f"[*] Forwarded {pkts} chunks ({pkts / dt:.1f} pkts/s)", end="\r")

    except Exception as exc:
        print(f"[!] Bridge error: {exc}", file=sys.stderr)


async def replay_stream(ws_url: str, node_id: int = 0, sound: str = "tone", duration_s: float = 10.0) -> None:
    """Stream synthetic 4,031 Hz companded Thingy chunks at real-time cadence."""
    import websockets

    print(f"[*] Connecting to sound-web at {ws_url}...")
    async with websockets.connect(ws_url) as ws:
        print(f"[+] Connected! Replaying {sound} on Node {node_id} for {duration_s}s at 42.5 pkts/s...")
        chunk_interval = 190.0 / 4031.25  # ~0.0471 sec per 190 samples
        chunks_total = int(duration_s / chunk_interval)

        t0 = 0.0
        dt = 1.0 / 4031.25

        for i in range(chunks_total):
            start_loop = time.monotonic()
            if sound == "tone":
                t = np.arange(190) * dt + t0
                # Alternate between 440 Hz tone and quiet
                freq = 440.0 if (i // 40) % 2 == 0 else 0.0
                samples = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
                t0 += 190 * dt
            elif sound == "noise":
                samples = np.random.uniform(-0.3, 0.3, 190).astype(np.float32)
            else:
                samples = np.zeros(190, dtype=np.float32)

            # Compress float samples into 8-bit companded codes (2^(q/8) - 1)
            int_samples = np.clip(np.round(samples * 32768.0), -32767, 32767).astype(np.int32)
            signs = np.where(int_samples < 0, 128, 0).astype(np.uint8)
            mags = np.abs(int_samples)
            # q = 8 * log2(mag + 1)
            q = np.clip(np.round(8.0 * np.log2(mags + 1.0)), 0, 127).astype(np.uint8)
            codes = (signs | q).tolist()

            msg = {"node_id": node_id, "codes": codes}
            await ws.send(json.dumps(msg))

            elapsed = time.monotonic() - start_loop
            sleep_time = max(0.0, chunk_interval - elapsed)
            await asyncio.sleep(sleep_time)

        print(f"\n[+] Replay complete ({chunks_total} chunks sent).")


async def ble_stream(ws_url: str, address: str | None = None) -> None:
    """Connect directly to Thingy:53 gateway over BLE and stream audio packets."""
    import websockets
    from bleak import BleakClient, BleakScanner

    SERVICE_UUID = "a88278e0-7009-4bee-a6f8-e1dc3ff02b92"
    RESULT_UUID = "a88278e1-7009-4bee-a6f8-e1dc3ff02b92"
    CONTROL_UUID = "a88278e2-7009-4bee-a6f8-e1dc3ff02b92"

    print(f"[*] Connecting to sound-web at {ws_url}...")
    async with websockets.connect(ws_url) as ws:
        if not address:
            print("[*] Scanning for Thingy ranging central device...")
            device = await BleakScanner.find_device_by_filter(
                lambda d, adv: d.name == "Thingy ranging" or SERVICE_UUID in adv.service_uuids,
                timeout=10.0,
            )
            if not device:
                print("[!] Could not find Thingy central device.")
                return
            address = device.address

        print(f"[*] Connecting to BLE device {address}...")
        async with BleakClient(address) as client:
            print("[+] Connected to Thingy central!")

            def on_notify(_handle: int, data: bytearray) -> None:
                if len(data) == 210 and data[5] == 0xB5:
                    asyncio.create_task(ws.send(bytes(data)))

            await client.start_notify(RESULT_UUID, on_notify)
            # Enable audio streaming: decim 4
            await client.write_gatt_char(CONTROL_UUID, b"stream 4\n")
            print("[+] Streaming audio to sound-web (Ctrl+C to stop)...")

            try:
                while True:
                    await asyncio.sleep(1.0)
            finally:
                await client.write_gatt_char(CONTROL_UUID, b"stream 0\n")
                await client.stop_notify(RESULT_UUID)


def main() -> None:
    parser = argparse.ArgumentParser(description="Thingy:53 Audio Classifier Bridge")
    parser.add_argument(
        "--ws-url",
        default="ws://127.0.0.1:8000/ws/thingy",
        help="sound-web ingestion WebSocket URL (default: ws://127.0.0.1:8000/ws/thingy)",
    )
    sub = parser.add_subparsers(dest="mode")
    parser.set_defaults(mode="stdin")

    # Stdin mode
    sub.add_parser("stdin", help="Read NDJSON from stdin and forward to sound-web")

    # Replay mode
    replay_parser = sub.add_parser("replay", help="Replay synthetic stream")
    replay_parser.add_argument("--sound", choices=["tone", "noise", "silence"], default="tone")
    replay_parser.add_argument("--node-id", type=int, default=0)
    replay_parser.add_argument("--duration", type=float, default=10.0)

    # BLE mode
    ble_parser = sub.add_parser("ble", help="Direct BLE connection to Thingy:53")
    ble_parser.add_argument("--address", help="BLE MAC address (optional)")

    args = parser.parse_args()

    if args.mode == "replay":
        asyncio.run(replay_stream(args.ws_url, args.node_id, args.sound, args.duration))
    elif args.mode == "ble":
        asyncio.run(ble_stream(args.ws_url, args.address))
    else:
        asyncio.run(forward_stdin(args.ws_url))


if __name__ == "__main__":
    main()
