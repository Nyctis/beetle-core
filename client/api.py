"""A small client of the local daemon, shared by the CLI, the SDK and the MCP server.
It starts the daemon if nothing answers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from client.daemon import owner_token
from node import wire

START_WAIT = 60.0


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


class ApiError(Exception):
    def __init__(self, status: int, error: dict):
        super().__init__(error.get("message", str(error)))
        self.status, self.error, self.code = status, error, error.get("code")


class Api:
    def __init__(self, token: str | None = None):
        self.home = wire.home()
        self.token = token or owner_token(self.home)
        self.http = self._connect() or self._start()

    def _client(self, port: int) -> httpx.Client:
        return httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=None,
                            headers={"Authorization": f"Bearer {self.token}"})

    def _connect(self) -> httpx.Client | None:
        state = wire.read_json(self.home / "serve.json")
        if not state:
            return None
        client = self._client(state["port"])
        try:
            client.get("/beetle/health").raise_for_status()
            return client
        except httpx.HTTPStatusError as e:
            # Our daemon said no, and a second one would be worse. If it's gone, whoever
            # answered only took over the port.
            if e.response.status_code == 401 and alive(state["pid"]):
                raise SystemExit("the running daemon refused this token") from e
            return None
        except httpx.HTTPError:
            return None

    def _start(self) -> httpx.Client:
        log = open(self.home / "serve.log", "ab")
        subprocess.Popen([sys.executable, "-m", "client.cli", "serve"], stdout=log, stderr=log,
                         stdin=subprocess.DEVNULL, start_new_session=True,
                         cwd=Path(__file__).resolve().parents[1])
        deadline = time.time() + START_WAIT
        while time.time() < deadline:
            time.sleep(0.5)
            if client := self._connect():
                return client
        raise SystemExit(f"the daemon did not start; see {self.home / 'serve.log'}")

    def call(self, method: str, path: str, body: dict | None = None, **params):
        r = self.http.request(method, path, json=body, params=params or None)
        if r.status_code >= 400:
            raise ApiError(r.status_code, r.json().get("error", {"message": r.text}))
        return r.json()

    def get(self, path: str, **params):
        return self.call("GET", path, **{k: v for k, v in params.items() if v is not None})

    def post(self, path: str, body: dict | None = None):
        return self.call("POST", path, body or {})

    def chat(self, model: str, messages: list[dict], **options):
        """Yields (headers, None) once, then (None, piece) for each piece, then
        (None, final chunk) with the usage and cost."""
        params = {k: options.pop(k) for k in ("max_tokens", "temperature", "seed") if k in options}
        body = {"model": model, "messages": messages, "stream": True, **params,
                "beetle": {k: v for k, v in options.items() if v is not None}}
        with self.http.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code >= 400:
                r.read()
                raise ApiError(r.status_code, r.json().get("error", {"message": r.text}))
            yield dict(r.headers), None
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    event = json.loads(line[6:])
                    choice = event["choices"][0]
                    if choice["finish_reason"]:
                        yield None, event
                    elif choice["delta"].get("content"):
                        yield None, choice["delta"]["content"]
