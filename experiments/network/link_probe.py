"""One end of a connectivity probe: the product's rendezvous, then a direct iroh link.

The orchestrator (items.py) creates a throwaway seller, opens a job or a payment
channel, and starts one of these on each machine, with its settings on stdin. The
two ends share nothing but the chain. Relays are off in Peer, and the path is read
back from iroh, so a run that passes was direct.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

import node  # noqa: E402,F401

from beetle_chain import Chain, net_keypair  # noqa: E402,F401
from net.rendezvous import ChainRendezvous, ChannelRendezvous  # noqa: E402
from node import wire  # noqa: E402
from solders.keypair import Keypair  # noqa: E402
from solders.pubkey import Pubkey  # noqa: E402

MIB = 1 << 20
PINGS = 20


def rendezvous(s: dict, chain: Chain, kp: Keypair):
    about = bytes.fromhex(s["job_id"]) if s["via"] == "job" else Pubkey.from_string(s["channel"])
    secret = bytes.fromhex(s.get("net_secret", ""))
    if s["role"] == "seller":
        if s["via"] == "job":
            return ChainRendezvous.seller(chain, about, kp, secret)
        return ChannelRendezvous.seller(chain, about, kp, Pubkey.from_string(s["other"]), secret)
    if s["via"] == "job":
        return ChainRendezvous.buyer(chain, about, kp)
    return ChannelRendezvous.buyer(chain, about, kp, Pubkey.from_string(s["other"]), bytes.fromhex(s["net_pk"]))


async def pump(link, size: int, sending: bool) -> float:
    block, left, t0 = bytes(MIB), size, time.monotonic()
    while left > 0:
        n = min(left, MIB)
        await (link.write(block[:n]) if sending else link.read_exact(n))
        left -= n
    return time.monotonic() - t0


async def main() -> None:
    s = json.loads(sys.stdin.read())
    chain = Chain.from_file(Path(s["chain"]).expanduser())
    kp = Keypair.from_bytes(bytes.fromhex(s["key"]))
    cfg = {"stun": s["stun"], "echo_nodes": s.get("echo_nodes", [])}
    peer = await wire.bound_peer(cfg, s.get("port", 0))
    addrs = [s["public"]] if s.get("public") else await wire.candidates(peer, cfg)
    rdv = rendezvous(s, chain, kp)
    size = s["mib"] * MIB
    if s["role"] == "seller":
        them = await rdv.wait(s["window"])
        await rdv.publish({"id": peer.id, "addrs": addrs})
        link = await peer.link(them["id"], them["addrs"], initiator=False, window=s["window"])
        for _ in range(PINGS):
            await link.write(await link.read_exact(16))
        await pump(link, size, True)
        await pump(link, size, False)
        await link.read_exact(1)  # the buyer has its numbers
        print(json.dumps({"role": "seller", "saw_buyer_at": them["addrs"]}), flush=True)
    else:
        t0 = time.monotonic()
        await rdv.publish({"id": peer.id, "addrs": addrs})
        t1 = time.monotonic()
        them = await rdv.wait(s["window"])
        t2 = time.monotonic()
        link = await peer.link(them["id"], them["addrs"], initiator=True, window=s["window"])
        t3 = time.monotonic()
        rtts = []
        for _ in range(PINGS):
            t = time.monotonic()
            await link.write(bytes(16))
            await link.read_exact(16)
            rtts.append((time.monotonic() - t) * 1000)
        down = await pump(link, size, False)
        up = await pump(link, size, True)
        path = wire.path_of(link)
        await link.write(b"x")
        print(json.dumps({
            "role": "buyer", "publish_s": round(t1 - t0, 2), "answer_s": round(t2 - t1, 2),
            "rendezvous_s": round(t2 - t0, 2), "link_s": round(t3 - t2, 2),
            "rtt_ms_median": round(sorted(rtts)[PINGS // 2], 1), "rtt_ms_path": path["rtt_ms"],
            "direct": path["direct"], "remote_addr_known": bool(path["remote_addr"]),
            "down_mbps": round(size * 8 / down / 1e6, 1), "up_mbps": round(size * 8 / up / 1e6, 1),
            "candidates": len(addrs)}), flush=True)
    await asyncio.sleep(0.5)
    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
