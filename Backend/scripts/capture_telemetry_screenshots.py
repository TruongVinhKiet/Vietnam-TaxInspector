"""
Capture authenticated telemetry screenshots with headless Edge via CDP.

The script avoids Playwright/Selenium dependencies. It logs in through the
local API, injects the session cookie into a temporary browser profile, then
captures desktop and mobile screenshots plus a small diagnostics JSON file.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import socket
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import requests


DEFAULT_EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


class CdpWebSocket:
    def __init__(self, ws_url: str):
        parsed = urllib.parse.urlparse(ws_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.path = parsed.path + (("?" + parsed.query) if parsed.query else "")
        self.sock = socket.create_connection((self.host, self.port), timeout=10)
        self.next_id = 1
        self.events: list[dict] = []
        self._handshake()

    def _handshake(self) -> None:
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode("ascii"))
        response = self.sock.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"CDP websocket handshake failed: {response[:200]!r}")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass

    def send(self, method: str, params: dict | None = None) -> int:
        msg_id = self.next_id
        self.next_id += 1
        payload = json.dumps({"id": msg_id, "method": method, "params": params or {}}).encode("utf-8")
        self._send_frame(payload)
        return msg_id

    def request(self, method: str, params: dict | None = None, timeout: float = 10) -> dict:
        msg_id = self.send(method, params)
        deadline = time.time() + timeout
        while time.time() < deadline:
            message = self.recv(deadline - time.time())
            if not message:
                continue
            if message.get("id") == msg_id:
                if "error" in message:
                    raise RuntimeError(f"CDP {method} failed: {message['error']}")
                return message.get("result", {})
            self.events.append(message)
        raise TimeoutError(f"Timed out waiting for CDP response: {method}")

    def drain(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            message = self.recv(max(0.05, deadline - time.time()))
            if message:
                self.events.append(message)

    def recv(self, timeout: float = 1) -> dict | None:
        self.sock.settimeout(max(0.05, timeout))
        try:
            header = self._recv_exact(2)
        except (TimeoutError, socket.timeout):
            return None
        except OSError:
            return None
        if not header:
            return None
        first, second = header
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else b""
        payload = self._recv_exact(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        if opcode == 8:
            return None
        if opcode == 9:
            return None
        if opcode != 1:
            return None
        return json.loads(payload.decode("utf-8"))

    def _recv_exact(self, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise TimeoutError("websocket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _send_frame(self, payload: bytes) -> None:
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask = secrets.token_bytes(4)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(bytes(header) + mask + masked)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture telemetry dashboard screenshots")
    parser.add_argument("--api-base", default="http://localhost:8000/api")
    parser.add_argument("--frontend-url", default="http://localhost:3000/pages/telemetry.html")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--out-dir", default="Backend/data/screenshots")
    parser.add_argument("--edge-path", default=DEFAULT_EDGE)
    parser.add_argument("--debug-port", type=int, default=9223)
    return parser.parse_args()


def wait_for_json(url: str, timeout: float = 10) -> dict:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - diagnostics script
            last_error = exc
            time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def create_target(port: int, url: str) -> dict:
    encoded = urllib.parse.quote(url, safe="")
    request = urllib.request.Request(f"http://127.0.0.1:{port}/json/new?{encoded}", method="PUT")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def login(api_base: str, username: str, password: str) -> str:
    response = requests.post(
        f"{api_base.rstrip('/')}/auth/login",
        json={"badge_id": username, "password": password, "expected_role": "inspector"},
        timeout=15,
    )
    response.raise_for_status()
    token = response.cookies.get("tax_session")
    if not token:
        raise RuntimeError("Login succeeded but tax_session cookie was not returned")
    return token


def capture_view(client: CdpWebSocket, url: str, token: str, width: int, height: int, scale: float, out_path: Path) -> dict:
    client.request("Network.enable")
    client.request("Page.enable")
    client.request("Runtime.enable")
    client.request("Log.enable")
    client.request(
        "Network.setCookie",
        {
            "name": "tax_session",
            "value": token,
            "url": "http://localhost:8000",
            "path": "/",
            "sameSite": "Lax",
        },
    )
    client.request(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": width,
            "height": height,
            "deviceScaleFactor": scale,
            "mobile": width < 700,
        },
    )
    client.request("Page.navigate", {"url": url})
    wait_for_dashboard(client, timeout=18)
    screenshot = client.request("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True}, timeout=20)
    out_path.write_bytes(base64.b64decode(screenshot["data"]))
    state = client.request(
        "Runtime.evaluate",
        {
            "returnByValue": True,
            "expression": """
            (() => ({
              href: location.href,
              title: document.title,
              bodyText: document.body.innerText.slice(0, 1200),
              total: document.getElementById('metric-total')?.innerText || null,
              rpm: document.getElementById('metric-rpm')?.innerText || null,
              latency: document.getElementById('metric-latency')?.innerText || null,
              satisfaction: document.getElementById('metric-satisfaction')?.innerText || null
            }))()
            """,
        },
    )
    return state.get("result", {}).get("value", {})


def wait_for_dashboard(client: CdpWebSocket, timeout: float = 15) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        client.drain(0.4)
        try:
            state = client.request(
                "Runtime.evaluate",
                {
                    "returnByValue": True,
                    "expression": """
                    (() => ({
                      total: document.getElementById('metric-total')?.innerText || '',
                      refresh: document.getElementById('refreshBtn')?.innerText || ''
                    }))()
                    """,
                },
                timeout=2,
            )
            value = state.get("result", {}).get("value", {})
            if value.get("total") not in {"", "-"} and "Loading" not in value.get("refresh", ""):
                return
        except Exception:  # noqa: BLE001 - keep waiting for page scripts
            pass
    client.drain(1)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token = login(args.api_base, args.username, args.password)
    profile = (out_dir / "edge_profile").resolve()
    profile.mkdir(parents=True, exist_ok=True)
    browser_log = out_dir / "browser_stderr.log"
    browser_err = browser_log.open("wb")
    edge = subprocess.Popen(
        [
            args.edge_path,
            "--headless=new",
            f"--remote-debugging-port={args.debug_port}",
            "--remote-debugging-address=127.0.0.1",
            f"--user-data-dir={profile}",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--disable-background-networking",
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=browser_err,
    )

    diagnostics: dict = {"screenshots": {}, "events": []}
    client: CdpWebSocket | None = None
    try:
        try:
            wait_for_json(f"http://127.0.0.1:{args.debug_port}/json/version")
        except Exception as exc:
            if edge.poll() is not None:
                raise RuntimeError(
                    f"Browser exited before CDP was ready with code {edge.returncode}. "
                    f"See {browser_log}"
                ) from exc
            raise
        target = create_target(args.debug_port, "about:blank")
        client = CdpWebSocket(target["webSocketDebuggerUrl"])
        diagnostics["screenshots"]["desktop"] = capture_view(
            client,
            args.frontend_url,
            token,
            width=1440,
            height=1000,
            scale=1,
            out_path=out_dir / "telemetry_desktop.png",
        )
        diagnostics["screenshots"]["mobile"] = capture_view(
            client,
            args.frontend_url,
            token,
            width=390,
            height=844,
            scale=2,
            out_path=out_dir / "telemetry_mobile.png",
        )
        diagnostics["events"] = [
            event for event in client.events
            if event.get("method") in {"Runtime.exceptionThrown", "Runtime.consoleAPICalled", "Log.entryAdded"}
        ]
        (out_dir / "telemetry_capture_diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"status": "success", "out_dir": str(out_dir), **diagnostics["screenshots"]}, ensure_ascii=True, indent=2))
        return 0
    finally:
        if client:
            client.close()
        edge.terminate()
        try:
            edge.wait(timeout=5)
        except subprocess.TimeoutExpired:
            edge.kill()
        browser_err.close()


if __name__ == "__main__":
    raise SystemExit(main())
