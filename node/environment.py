"""The seller's environment as a process of its own: `python -m node.environment`.

This is where prompts are plaintext, so this process has no wallet: it never
opens that file, and the daemon that starts it passes only the seller's public
key. It takes one kind of command on stdin, a buyer's contact, and writes
vouchers and session ends on stdout, one JSON object per line. Logs go to stderr.

This separates memory, not users. Run as the same Unix user it can still read
the wallet file, so a machine that wants more runs it as another one.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from solders.pubkey import Pubkey

import node  # noqa: F401  (import paths)

from net import echo
from node import wire
from node.metered import MIN_GRACE_S, MeteredEnvironment
from serve.gateway import Backend, Gateway, Spool, default_scratch

MAX_UPLOAD = 1 << 30


def emit(msg: dict) -> None:
    print(json.dumps(msg), flush=True)


async def commands(gateway: Gateway) -> None:
    """Runs until the daemon goes away, and this process goes with it."""
    while line := await asyncio.to_thread(sys.stdin.readline):
        msg = json.loads(line)
        if msg["t"] == "contact":
            gateway._spawn(gateway._connect(msg["them"]))


async def serve(backend: Backend | None = None) -> None:
    where = wire.home()
    cfg = wire.load_config(where)
    host = cfg["host"]
    backend = backend or Backend(host["backend"], host.get("backend_key"))
    served = set(await backend.models())
    names = {host.get("serve_as", {}).get(m, m) for m in host["models"]}
    env = MeteredEnvironment(backend, served & names, Spool(where / "spool", MAX_UPLOAD),
                             host.get("scratch") or default_scratch(),
                             chain=wire.load_chain(cfg, where), report=emit,
                             payee=Pubkey.from_string(os.environ["BEETLE_PAYEE"]),
                             min_grace=host.get("min_grace_s", MIN_GRACE_S),
                             require_tmpfs=not host.get("allow_disk_scratch", False))
    await env.sweep()
    port = host.get("port", 0)
    peer = await wire.bound_peer(cfg, port + 1 if port else 0, services={echo.ALPN: echo.answer})
    public = host.get("public_addr")
    given = [f"{public.rsplit(':', 1)[0]}:{peer.port}"] if public else []
    addrs = list(dict.fromkeys(given + await wire.candidates(peer, cfg)))[:wire.MAX_CANDIDATES]
    gateway = Gateway(peer, env, addrs, open(host["tap"], "ab") if host.get("tap") else None)
    emit({"t": "ready", **gateway.contact(), "models": sorted(env.models),
          "adapters": env.scratch_problem() or True})
    await commands(gateway)


if __name__ == "__main__":
    asyncio.run(serve())
