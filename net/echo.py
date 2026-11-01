"""Address echo: learn this endpoint's public address and what kind of NAT it is behind.

A public peer tells each caller where its packets came from. iroh keeps its UDP
socket to itself, so the probe has to run on the same endpoint that will later
dial; a separate STUN client would see a different NAT mapping.

This answers "what address should I publish". It proves nothing: a lying echo
node can hand out a wrong address, which at worst makes the dial fail, and one
node alone is a central STUN server under another name. Echo nodes also learn
that this IP uses the protocol, so use a throwaway endpoint key and rotate
nodes. The mapping is only fresh for as long as the NAT keeps it, so dial soon
after probing.
"""

from __future__ import annotations

import asyncio

import iroh

ALPN = b"beetle/echo/1"
PROBE_TIMEOUT = 3.0
MAX_ADDR_LEN = 128


async def answer(conn) -> None:
    """Server side: write back the address this connection arrived from."""
    path = next((p for p in conn.paths() if p.is_selected), None)
    if path is None:
        conn.close(1, b"no path")
        return
    send = await conn.open_uni()
    await send.write_all(path.remote_addr.encode())
    await send.finish()
    try:  # the caller closes once it has read; closing first could lose the reply
        await asyncio.wait_for(conn.closed(), PROBE_TIMEOUT)
    except TimeoutError:
        conn.close(0, b"done")


async def observe(peer, node: dict, timeout: float = PROBE_TIMEOUT) -> str:
    """Ask one echo node {id, addrs} how it sees us."""
    addr = iroh.EndpointAddr(iroh.EndpointId.from_string(node["id"]), None, node["addrs"])

    async def ask() -> str:
        conn = await peer.endpoint.connect(addr, ALPN)
        recv = await conn.accept_uni()
        seen = await recv.read_to_end(MAX_ADDR_LEN)
        conn.close(0, b"thanks")
        return seen.decode()

    return await asyncio.wait_for(ask(), timeout)


def classify(observed: list[str], local: list[str]) -> str:
    """open, eim, edm, unknown (one answer behind a NAT) or blocked (no answer)."""
    if not observed:
        return "blocked"
    if set(observed) <= set(local):
        return "open"
    if len(observed) < 2:
        return "unknown"
    return "eim" if len(set(observed)) == 1 else "edm"


async def probe(peer, nodes: list[dict], timeout: float = PROBE_TIMEOUT) -> dict:
    """Ask every node at once. Returns {"addrs": candidates, "nat": hint, "observed": [...]}."""
    answers = await asyncio.gather(*(observe(peer, n, timeout) for n in nodes),
                                   return_exceptions=True)
    observed = [a for a in answers if isinstance(a, str)]
    local = peer.addrs
    return {"addrs": list(dict.fromkeys(observed + local)),
            "nat": classify(observed, local), "observed": observed}
