"""Bridge utility connecting Thingy:53 audio streams to the sound-web classifier.

Supports three modes:
1. stdin mode (default): Reads NDJSON stream chunks from stdin and forwards to sound-web over WebSocket.
2. replay mode: Streams simulated or recorded Thingy chunks at real-time 42 Hz cadence.
3. ble mode: Directly connects to a Thingy:53 gateway over Bluetooth Low Energy using Bleak.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
import inspect
import json
import sys
import time
from typing import Any, Callable

import numpy as np


class ThingyBleManager:
    """Manages direct Bluetooth Low Energy connectivity to a Thingy:53 central device."""

    SERVICE_UUID = "a88278e0-7009-4bee-a6f8-e1dc3ff02b92"
    RESULT_UUID = "a88278e1-7009-4bee-a6f8-e1dc3ff02b92"
    CONTROL_UUID = "a88278e2-7009-4bee-a6f8-e1dc3ff02b92"

    def __init__(self, on_packet: Callable[[bytes], Any] | None = None) -> None:
        self.on_packet = on_packet
        self.state: str = "disconnected"  # disconnected, scanning, connecting, connected, error
        self.address: str | None = None
        self.name: str | None = None
        self.rssi: int | None = None
        self.error: str | None = None
        self.packet_count: int = 0
        self.last_packet_time: float | None = None
        self._task: asyncio.Task[None] | None = None
        self._client: Any = None
        self._stop_requested: bool = False
        self._packet_timestamps: deque[float] = deque(maxlen=100)

    def get_status(self) -> dict[str, Any]:
        """Return real-time BLE connection diagnostics."""
        now = time.monotonic()
        while self._packet_timestamps and now - self._packet_timestamps[0] > 2.0:
            self._packet_timestamps.popleft()
        rate = len(self._packet_timestamps) / 2.0 if self._packet_timestamps else 0.0
        return {
            "state": self.state,
            "address": self.address,
            "name": self.name,
            "rssi": self.rssi,
            "error": self.error,
            "packet_count": self.packet_count,
            "packet_rate": round(rate, 1),
            "connected": self.state == "connected",
        }

    def start(self, address: str | None = None) -> None:
        """Start background BLE scanning and streaming task."""
        if self._task and not self._task.done():
            return
        self._stop_requested = False
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run_loop(address))
        except RuntimeError:
            self.error = "No running asyncio event loop"
            print("[!] Warning: Cannot start BLE manager outside an active asyncio event loop", file=sys.stderr)

    async def stop(self) -> None:
        """Stop BLE streaming and disconnect device."""
        self._stop_requested = True
        self.state = "disconnected"
        if self._client is not None:
            try:
                if self._client.is_connected:
                    try:
                        await self._client.write_gatt_char(self.CONTROL_UUID, b"stream 0\n")
                    except Exception:
                        pass
                    try:
                        await self._client.stop_notify(self.RESULT_UUID)
                    except Exception:
                        pass
                    await self._client.disconnect()
            except Exception:
                pass
            finally:
                self._client = None

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self, target_address: str | None) -> None:
        from bleak import BleakClient, BleakScanner

        while not self._stop_requested:
            try:
                addr = target_address
                if not addr:
                    self.state = "scanning"
                    self.error = None

                    def _device_filter(d: Any, adv: Any) -> bool:
                        name = (d.name or getattr(adv, "local_name", None) or "").lower()
                        uuids = [u.lower() for u in getattr(adv, "service_uuids", [])]
                        return (
                            "thingy ranging" in name
                            or "conn time sync" in name
                            or "thingy" in name
                            or self.SERVICE_UUID.lower() in uuids
                            or "a88278d0-7009-4bee-a6f8-e1dc3ff02b92" in uuids
                        )

                    device = await BleakScanner.find_device_by_filter(
                        _device_filter,
                        timeout=8.0,
                    )
                    if not device:
                        self.error = "No Thingy:53 found advertising nearby"
                        self.state = "disconnected"
                        await asyncio.sleep(3.0)
                        continue
                    addr = device.address
                    self.name = device.name or "Thingy:53"
                    self.rssi = getattr(device, "rssi", None)

                self.state = "connecting"
                self.address = addr
                async with BleakClient(addr, timeout=20.0) as client:
                    self._client = client
                    service_uuids = [s.uuid.lower() for s in client.services]

                    if self.SERVICE_UUID.lower() not in service_uuids:
                        dev_name = self.name or "Thingy:53"
                        self.error = (
                            f"Detected '{dev_name}' (Milestone 1 firmware). "
                            "This firmware only tests clock sync and has no microphone driver. "
                            "Flash 'thingy_ranging' (Milestone 2) to enable live audio streaming."
                        )
                        self.state = "firmware_mismatch"
                        print(f"[!] {self.error}", file=sys.stderr)
                        await asyncio.sleep(6.0)
                        continue

                    self.state = "connected"
                    self.error = None

                    def _on_notify(_handle: int, data: bytearray) -> None:
                        if len(data) == 210 and data[5] == 0xB5:
                            now = time.monotonic()
                            self.packet_count += 1
                            self.last_packet_time = now
                            self._packet_timestamps.append(now)
                            if self.on_packet is not None:
                                raw = bytes(data)
                                if inspect.iscoroutinefunction(self.on_packet):
                                    asyncio.create_task(self.on_packet(raw))
                                else:
                                    try:
                                        self.on_packet(raw)
                                    except Exception as exc:
                                        print(f"[!] BLE callback error: {exc}", file=sys.stderr)

                    await client.start_notify(self.RESULT_UUID, _on_notify)
                    # roomsense uses stream 3 (5376 Hz); older prototype used stream 4 (4031 Hz)
                    try:
                        await client.write_gatt_char(self.CONTROL_UUID, b"stream 3\n")
                    except Exception:
                        await client.write_gatt_char(self.CONTROL_UUID, b"stream 4\n")

                    while client.is_connected and not self._stop_requested:
                        await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.error = str(exc)
                self.state = "error"
                if not self._stop_requested:
                    await asyncio.sleep(3.0)

        self.state = "disconnected"
        self._client = None


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
            # Enable audio streaming: decim 3 (roomsense 5376 Hz) or 4 (legacy 4031 Hz)
            try:
                await client.write_gatt_char(CONTROL_UUID, b"stream 3\n")
            except Exception:
                await client.write_gatt_char(CONTROL_UUID, b"stream 4\n")
            print("[+] Streaming audio to sound-web (Ctrl+C to stop)...")

            try:
                while True:
                    await asyncio.sleep(1.0)
            finally:
                await client.write_gatt_char(CONTROL_UUID, b"stream 0\n")
                await client.stop_notify(RESULT_UUID)


def main(argv: list[str] | None = None) -> None:
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

    args = parser.parse_args(argv)

    if args.mode == "replay":
        asyncio.run(replay_stream(args.ws_url, args.node_id, args.sound, args.duration))
    elif args.mode == "ble":
        asyncio.run(ble_stream(args.ws_url, args.address))
    else:
        asyncio.run(forward_stdin(args.ws_url))


if __name__ == "__main__":
    main()
