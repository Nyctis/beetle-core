"""The seller's inference gateway.

It takes P2P links, opens the sealed channel and forwards each request to any
OpenAI-compatible server (vLLM, llama.cpp server, mlx_lm.server), streaming the
answer back in fixed-length frames. It has two halves with a seam between them:

    Gateway      the host side: iroh links and a spool of sealed files.
                 It handles ciphertext only.
    Environment  the sealed side: keys, plaintext, the backend, traffic shaping.
                 It talks to the outside through link-shaped objects alone.

On an ordinary machine the host can read everything in memory, which is why the
declared tier is T3-C. The seam is what lets the Environment move into a sealed
container or a CVM later; it is not a protection today. So this protects a
session against the network and against anyone who can only watch the host's
traffic. It does not protect it against whoever runs the machine, against the
backend, or against a bug in either. Only the buyer's own session reaches its
adapter; base models and adapters loaded at startup are shared. A backend may
keep its own logs, so turn them off.

A decrypted adapter goes in `scratch`, which must be a tmpfs: with require_tmpfs
an Environment on any other disk refuses adapters rather than write them there.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from contextlib import aclosing
from pathlib import Path

import httpx

import net  # noqa: F401  (puts common/ on the import path)

import beetle_archive
import beetle_crypto as bc
from net import channel, echo, transfer
from net.peer import Peer
from net.rendezvous import FileRendezvous, Rendezvous
from serve import (ADAPTER_FILES, DEFAULT_TIER, MAX_RESPONSE_TOKENS, PASSED_PARAMS, UNPROVEN_TIERS,
                   adapter_aad, estimate_tokens, next_transcript, receipt_digest)

RESUME_WINDOW = 120.0    # how long a session waits for its buyer to come back
UPLOAD_TIMEOUT = 600.0
ANNOUNCE_WAIT = 10.0     # how long an upload may wait for its sealed announcement
ADAPTER_NAME = re.compile(r"a[0-9a-f]{16}")
DIAL_WINDOW = 30.0


class BackendError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"backend answered {status}: {detail}")
        self.status = status


class Backend:
    """An OpenAI-compatible server. Adapter calls use vLLM's endpoints; a server
    without them answers 404, which reaches the buyer as a plain error."""

    def __init__(self, url: str, api_key: str | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Prompts cross this hop in the clear, so no proxy from the environment may join it.
        self.http = httpx.AsyncClient(base_url=url.rstrip("/").removesuffix("/v1"),
                                      headers=headers, trust_env=False,
                                      timeout=httpx.Timeout(None, connect=10.0))

    async def models(self) -> list[str]:
        r = await self.http.get("/v1/models")
        r.raise_for_status()
        return [m["id"] for m in r.json()["data"]]

    async def chat(self, body: dict):
        """Yield the parsed events of one streaming chat completion."""
        async with self.http.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise BackendError(r.status_code, "chat refused")
            async for line in r.aiter_lines():
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    yield json.loads(line[5:])

    async def _post(self, path: str, body: dict) -> None:
        r = await self.http.post(path, json=body)
        if r.status_code != 200:
            raise BackendError(r.status_code, r.text[:200])

    async def load_lora(self, name: str, path: Path) -> None:
        await self._post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})

    async def unload_lora(self, name: str) -> None:
        await self._post("/v1/unload_lora_adapter", {"lora_name": name})


class Spool:
    """Host side: sealed uploads on disk, until the environment asks for them."""

    def __init__(self, root: Path, max_size: int):
        self.root, self.max_size = Path(root), max_size
        self.root.mkdir(parents=True, exist_ok=True)
        self._ready: dict[bytes, asyncio.Future] = {}

    def _future(self, root: bytes) -> asyncio.Future:
        return self._ready.setdefault(root, asyncio.get_running_loop().create_future())

    async def take(self, stream) -> None:
        root = await stream.read_exact(32)
        deadline = time.monotonic() + ANNOUNCE_WAIT
        while root not in self._ready:  # only what a session announced may fill the disk
            if time.monotonic() > deadline:
                raise transfer.TransferError("no session announced this file")
            await asyncio.sleep(0.05)
        path = await transfer.receive(stream, root, self.root / root.hex(), self.max_size)
        if not self._future(root).done():
            self._future(root).set_result(path)

    async def wait(self, root: bytes, timeout: float) -> Path:
        return await asyncio.wait_for(asyncio.shield(self._future(root)), timeout)

    def discard(self, root: bytes) -> None:
        self._ready.pop(root, None)
        for path in (self.root / root.hex(), self.root / f"{root.hex()}.part"):
            path.unlink(missing_ok=True)


class Session:
    """One buyer's session inside the environment. It outlives any single link."""

    def __init__(self, env: "Environment", ch: channel.Channel, hello: dict):
        self.env, self.ch = env, ch
        self.pipe = channel.Pipe(ch, mode=hello.get("mode", "burst"),
                                 tick=hello.get("tick", channel.TICK),
                                 bucket=hello.get("bucket", channel.DURATION_BUCKET),
                                 on_lost=self._lost)
        self.adapters: dict[str, Path] = {}
        self.uploads: list[bytes] = []
        self.cache_salt = os.urandom(32).hex()  # keeps prefix-cache timing inside the session
        self.seq = self.in_tokens = self.out_tokens = 0
        self.transcript = bytes(32)
        self.task: asyncio.Task | None = None
        self._expiry: asyncio.TimerHandle | None = None

    def _lost(self, exc) -> None:
        self._expiry = asyncio.get_running_loop().call_later(self.env.resume_window,
                                                             self.task.cancel)

    def reattached(self) -> None:
        if self._expiry:
            self._expiry.cancel()
            self._expiry = None

    def _receipt(self) -> dict:
        receipt = {"channel_id": self.ch.session_id.hex(), "seq": self.seq,
                   "in_tokens": self.in_tokens, "out_tokens": self.out_tokens,
                   "transcript_hash": self.transcript.hex()}
        return {**receipt, "sig": self.env.sign_key.sign(receipt_digest(receipt)).hex()}

    def _refuse(self, code: str, detail: str) -> None:
        self.pipe.send({"t": "error", "code": code, "detail": detail})

    async def run(self) -> None:
        env = self.env
        self.pipe.send({"t": "ready", "tier": env.tier, "protected": False,
                        "notice": UNPROVEN_TIERS[env.tier], "models": sorted(env.models),
                        "sign_pk": env.sign_pk.hex()})
        try:
            while True:
                msg = await self.pipe.recv()
                kind = msg.get("t")
                if kind == "close":
                    break
                try:
                    if kind == "chat":
                        await self._chat(msg)
                    elif kind == "adapter":
                        await self._adapter(msg)
                    else:
                        self._refuse("unknown_message", str(kind))
                except Exception as e:  # never tell a buyer's content back through a log
                    self.pipe.release()
                    self._refuse("internal", type(e).__name__)
        finally:
            await self._release_all()
            env.sessions.pop(self.ch.enc, None)
        await self.pipe.settled()
        self.pipe.send({"t": "closed", "receipt": self._receipt()})

    async def keep_going(self, pieces: int) -> bool:
        """Asked after every piece of an answer. False ends it where it stands."""
        return True

    async def _chat(self, msg: dict) -> None:
        model = msg.get("model")
        if model not in self.env.models and model not in self.adapters:
            self._refuse("unknown_model", str(model))
            return
        request = bc.canonical_json(msg)
        params = {k: msg[k] for k in PASSED_PARAMS if k in msg}
        params["max_tokens"] = min(int(params.get("max_tokens") or MAX_RESPONSE_TOKENS), MAX_RESPONSE_TOKENS)
        body = {**params, "model": model, "messages": msg.get("messages", []), "stream": True,
                "stream_options": {"include_usage": True}, "cache_salt": self.cache_salt}
        self.pipe.hold()
        text, usage, pieces, stopped = [], None, 0, False
        try:
            async with aclosing(self.env.backend.chat(body)) as events:
                async for event in events:
                    usage = event.get("usage") or usage
                    for choice in event.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            pieces += 1
                            text.append(piece)
                            self.pipe.send({"t": "delta", "text": piece})
                            stopped = not await self.keep_going(pieces)
                    if stopped:
                        break  # closing the stream is what stops the backend
        except (BackendError, httpx.HTTPError) as e:
            self._refuse("backend", str(e))
            self.pipe.release()
            return
        self.seq += 1
        self.in_tokens += (usage or {}).get("prompt_tokens", estimate_tokens(request))
        self.out_tokens += (usage or {}).get("completion_tokens", pieces)
        self.transcript = next_transcript(self.transcript, request, "".join(text))
        self.pipe.send({"t": "done", "receipt": self._receipt(), "stopped": stopped})
        self.pipe.release()

    async def _adapter(self, msg: dict) -> None:
        env, root = self.env, bytes.fromhex(msg["root"])
        if problem := env.scratch_problem():
            self._refuse("adapter_refused", problem)
            return
        self.uploads.append(root)
        try:
            path = await env.spool.wait(root, UPLOAD_TIMEOUT)  # this also announces the upload
            blob = await asyncio.to_thread(path.read_bytes)
            plain = await asyncio.to_thread(bc.open_blob, bytes.fromhex(msg["key"]), blob,
                                            adapter_aad(self.ch.session_id))
            files = await asyncio.to_thread(beetle_archive.read_tar, plain)
            if not files or not all(name in ADAPTER_FILES for name in files):
                raise ValueError("not an adapter")
        except Exception as e:  # a bad upload ends the upload, not the session
            env.spool.discard(root)
            self._refuse("adapter_invalid", type(e).__name__)
            return
        name = f"a{os.urandom(8).hex()}"  # a random name says nothing about the buyer
        where = env.scratch / name
        where.mkdir(parents=True, mode=0o700)
        self.adapters[name] = where
        for filename, data in files.items():
            (where / filename).write_bytes(data)
        env.spool.discard(root)
        try:
            await env.backend.load_lora(name, where)
        except (BackendError, httpx.HTTPError) as e:
            await self._release(name, loaded=False)
            self._refuse("adapter_rejected", str(e))
            return
        self.pipe.send({"t": "adapter_ok", "name": name})

    async def _release(self, name: str, loaded: bool = True) -> None:
        where = self.adapters.pop(name)
        try:
            if loaded:
                await self.env.backend.unload_lora(name)
        finally:  # the files go even if the backend has gone away
            shutil.rmtree(where, ignore_errors=True)

    async def _release_all(self) -> None:
        for name in list(self.adapters):
            try:
                await self._release(name)
            except (BackendError, httpx.HTTPError) as e:
                print(f"unload failed: {e}", file=sys.stderr, flush=True)
        for root in self.uploads:
            self.env.spool.discard(root)


class Environment:
    """The sealed side. In a real deployment enc_pk and sign_pk are what the
    attestation binds; here they are only as trustworthy as the machine."""

    def __init__(self, backend: Backend, models: set[str], spool: Spool, scratch: Path,
                 *, tier: str = DEFAULT_TIER, frame: int = channel.FRAME,
                 resume_window: float = RESUME_WINDOW, require_tmpfs: bool = False):
        if tier not in UNPROVEN_TIERS:
            raise ValueError(f"tier {tier} needs evidence this gateway cannot produce; "
                             f"it can only declare {sorted(UNPROVEN_TIERS)}")
        self.backend, self.models, self.spool, self.tier = backend, models, spool, tier
        self.scratch, self.frame, self.resume_window = Path(scratch), frame, resume_window
        self.require_tmpfs = require_tmpfs
        self.keypair = bc.HpkeKeyPair()
        self.sign_key = bc.new_signing_key()
        self.sessions: dict[bytes, Session] = {}
        self.seen: set[bytes] = set()

    @property
    def enc_pk(self) -> bytes:
        return self.keypair.public_bytes

    @property
    def sign_pk(self) -> bytes:
        return bc.public_bytes(self.sign_key)

    def scratch_is_memory(self) -> bool:
        return not self.require_tmpfs or is_tmpfs(self.scratch)

    def scratch_problem(self) -> str | None:
        """Why adapters can't be taken here. After a reboot Docker may have recreated a
        mounted directory as root, and a buyer should hear that, not PermissionError."""
        if not self.scratch_is_memory():
            return "this seller's scratch directory is on disk, so adapters would be too"
        if not os.access(self.scratch if self.scratch.exists() else self.scratch.parent, os.W_OK):
            return f"this seller can't write to {self.scratch}: its owner must fix that"
        return None

    async def sweep(self) -> None:
        """Clear adapters left by a crash, so they cannot be offered as public models."""
        self.scratch.mkdir(parents=True, exist_ok=True)
        for where in self.scratch.iterdir():
            if ADAPTER_NAME.fullmatch(where.name):
                self.models.discard(where.name)
                try:
                    await self.backend.unload_lora(where.name)
                except (BackendError, httpx.HTTPError):
                    pass  # the backend restarted too, so nothing is loaded
                shutil.rmtree(where, ignore_errors=True)

    async def attach(self, link, first: bytes) -> None:
        """Start or resume the session named by a link's first frame."""
        kind, _, enc = channel.peek(first)
        if kind == channel.OPEN:
            ch, hello = channel.accept(self.keypair, first, self.seen)
            session = Session(self, ch, json.loads(hello))
            self.sessions[enc] = session
            session.task = asyncio.create_task(session.run())
            session.pipe.attach(link)
            return
        session = self.sessions.get(enc)
        if session is None:
            raise channel.ChannelError("no such session")
        peer_rx = session.ch.read_resume(first)
        session.ch.frames_from(peer_rx)  # refuse now if they are gone, before saying yes
        session.pipe.detach()
        await link.write(session.ch.resume_frame())
        session.pipe.attach(link, peer_rx)
        session.reattached()


class Tap:
    """Record everything the host handles, so its contents can be audited.

    The record is what any host could keep for itself. It holds ciphertext only.
    """

    def __init__(self, link, log):
        self.link, self.log = link, log

    def _note(self, direction: bytes, data: bytes) -> None:
        self.log.write(direction + struct.pack(">I", len(data)) + data)
        self.log.flush()

    async def write(self, data: bytes) -> None:
        self._note(b">", data)
        await self.link.write(data)

    async def read_exact(self, n: int) -> bytes:
        data = await self.link.read_exact(n)
        self._note(b"<", data)
        return data

    def __getattr__(self, name):
        return getattr(self.link, name)


class Gateway:
    """The host side: links in, sealed frames and files through."""

    def __init__(self, peer: Peer, env: Environment, addrs: list[str], tap=None):
        self.peer, self.env, self.addrs, self.tap = peer, env, addrs, tap
        self._tasks: set[asyncio.Task] = set()

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _view(self, link):
        return Tap(link, self.tap) if self.tap else link

    async def serve(self, rendezvous: Rendezvous) -> None:
        """Link with each contact that arrives. A buyer coming back is just another one."""
        while True:
            them = await rendezvous.wait()
            await rendezvous.publish(self.contact())
            self._spawn(self._connect(them))

    def contact(self) -> dict:
        """What a buyer needs. `enc_pk` lives as long as this process, so a restart
        also ends what `seen` remembers. It is only as good as the rendezvous and
        the machine; a tier above T3-C would bind it with evidence instead."""
        return {"id": self.peer.id, "addrs": self.addrs, "enc_pk": self.env.enc_pk.hex()}

    async def _connect(self, them: dict) -> None:
        try:
            link = await self.peer.link(them["id"], them["addrs"], initiator=False,
                                        window=DIAL_WINDOW)
        except Exception as e:
            print(f"no link to {them.get('id', '?')[:8]}: {e}", file=sys.stderr, flush=True)
            return
        try:
            view = self._view(link)
            await self.env.attach(view, await view.read_exact(self.env.frame))
        except Exception as e:  # one bad buyer must not stop the others
            print(f"refused {them['id'][:8]}: {e}", file=sys.stderr, flush=True)
            await link.close()
            return
        try:
            while True:
                stream = await link.accept_stream()
                self._spawn(self._upload(self._view(stream)))
        except Exception:
            pass  # the connection ended; the session waits for its buyer

    async def _upload(self, stream) -> None:
        try:
            await self.env.spool.take(stream)
        except Exception as e:  # the part file stays, so the sender can resume
            print(f"upload stopped: {e}", file=sys.stderr, flush=True)


def default_scratch() -> Path:
    shm = Path("/dev/shm")
    return (shm if shm.is_dir() else Path(tempfile.gettempdir())) / "beetle"


def is_tmpfs(path: Path) -> bool:
    """On Linux, whether `path` is on a memory file system. Other systems have none
    by default, and macOS's temporary directory is an ordinary disk."""
    try:
        mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    except OSError:
        return False
    target, best, kind = Path(path).resolve(), -1, ""
    for _, point, fstype, *_ in mounts:
        if target.is_relative_to(point) and len(point) > best:
            best, kind = len(point), fstype
    return kind in ("tmpfs", "ramfs")


async def run(args) -> None:
    backend = Backend(args.backend, args.api_key)
    models = set(args.models.split(",")) if args.models else set(await backend.models())
    env = Environment(backend, models, Spool(args.spool, args.max_upload), args.scratch,
                      tier=args.tier, frame=args.frame, require_tmpfs=not args.allow_disk_scratch)
    await env.sweep()
    peer = await Peer.bind(args.port, services={echo.ALPN: echo.answer})
    addrs = args.addr or peer.addrs
    tap = open(args.tap, "ab") if args.tap else None
    print(json.dumps({"enc_pk": env.enc_pk.hex(), "sign_pk": env.sign_pk.hex(),
                      "endpoint_id": peer.id, "addrs": addrs, "tier": env.tier,
                      "models": sorted(models), "scratch_is_memory": env.scratch_is_memory()}), flush=True)
    rendezvous = FileRendezvous(args.rendezvous / "seller.json", args.rendezvous / "buyer.json")
    await Gateway(peer, env, addrs, tap).serve(rendezvous)


def main() -> None:
    e = os.environ.get
    ap = argparse.ArgumentParser(description="Private inference gateway for any OpenAI-compatible server")
    ap.add_argument("--backend", default=e("BEETLE_BACKEND", "http://127.0.0.1:8000"),
                    help="base URL of the OpenAI-compatible server")
    ap.add_argument("--api-key", default=e("BEETLE_BACKEND_KEY"))
    ap.add_argument("--models", default=e("BEETLE_MODELS"),
                    help="comma-separated models to offer; default: all the backend lists")
    ap.add_argument("--port", type=int, default=int(e("BEETLE_PORT", "41710")))
    ap.add_argument("--addr", action="append", help="ip:port to advertise; repeat for several")
    ap.add_argument("--rendezvous", type=Path, default=Path(e("BEETLE_RENDEZVOUS", "rendezvous")),
                    help="directory shared with the buyer")
    ap.add_argument("--tier", default=e("BEETLE_TIER", DEFAULT_TIER))
    ap.add_argument("--scratch", type=Path, default=Path(e("BEETLE_SCRATCH", default_scratch())),
                    help="where decrypted adapters go: one per gateway, on tmpfs, readable by the backend")
    ap.add_argument("--allow-disk-scratch", action="store_true",
                    help="for development: let decrypted adapters reach a disk")
    ap.add_argument("--spool", type=Path,
                    default=Path(e("BEETLE_SPOOL", Path(tempfile.gettempdir()) / "beetle-spool")),
                    help="where sealed uploads go")
    ap.add_argument("--max-upload", type=int, default=1 << 30)
    ap.add_argument("--frame", type=int, default=channel.FRAME)
    ap.add_argument("--tap", type=Path, help="record all bytes the host handles here")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
