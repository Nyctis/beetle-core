"""Measure an iroh link between two peers: time to connect, RTT and throughput.

Same harness as udp_punch.py, and the file swap again stands in for the on-chain
rendezvous. Relays are off, so a success means the path is direct.

Usage (one process per peer):
    python scripts/iroh_pair.py --out me.json --peer-file peer.json --role buyer|seller
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from net.peer import Peer  # noqa: E402
from udp_punch import STUN_SERVERS, stun_mapping  # noqa: E402

BLOCK = 1024 * 1024


def public_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return stun_mapping(sock, STUN_SERVERS[0])[0]


async def run(args) -> dict:
    peer = await Peer.bind(args.port)
    me = {"id": peer.id, "addrs": [f"{public_ip()}:{args.port}"]}
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(me))
    os.replace(tmp, args.out)

    deadline = time.time() + args.wait
    while True:
        try:
            other = json.loads(args.peer_file.read_text())
            break
        except (FileNotFoundError, ValueError):  # scp may still be writing it
            if time.time() > deadline:
                return {"error": "peer address never arrived", "me": me}
            await asyncio.sleep(0.2)

    t0 = time.time()
    link = await peer.link(other["id"], other["addrs"], initiator=args.role == "buyer",
                           window=args.window)
    connected_s = round(time.time() - t0, 3)

    rtts = []
    if args.role == "buyer":
        for _ in range(args.pings):
            t = time.time()
            await link.write(b"p")
            await link.read_exact(1)
            rtts.append((time.time() - t) * 1000)
        t = time.time()
        for _ in range(args.megabytes):
            await link.write(b"\0" * BLOCK)
        await link.read_exact(1)
        seconds = time.time() - t
        await link.close()
    else:
        for _ in range(args.pings):
            await link.write(await link.read_exact(1))
        for _ in range(args.megabytes):
            await link.read_exact(BLOCK)
        await link.write(b"k")
        await asyncio.sleep(1)
    rtts.sort()
    result = {"role": args.role, "connected_s": connected_s, "peer_addrs": other["addrs"]}
    if rtts:
        result.update(rtt_ms_median=round(rtts[len(rtts) // 2], 1),
                      mbps=round(args.megabytes * 8 / seconds, 1))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--peer-file", type=Path, required=True)
    ap.add_argument("--role", choices=["buyer", "seller"], required=True)
    ap.add_argument("--port", type=int, default=41710)
    ap.add_argument("--window", type=float, default=30)
    ap.add_argument("--wait", type=float, default=60)
    ap.add_argument("--pings", type=int, default=20)
    ap.add_argument("--megabytes", type=int, default=32)
    args = ap.parse_args()
    try:
        result = asyncio.run(run(args))
    except TimeoutError:
        result = {"role": args.role, "error": "no direct path within the window"}
    print(json.dumps(result))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
