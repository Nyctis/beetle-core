"""Measure whether two peers can reach each other directly over UDP.

Each side learns its public mapping from two public STUN servers, publishes it
to a file, waits for the peer's mapping, then both send at once. The file
exchange stands in for the on-chain rendezvous; no relay carries any packet.
Standard library only, so it runs on a bare node.

Usage (one process per peer, started within a few seconds of each other):
    python3 scripts/udp_punch.py --out me.json --peer-file peer.json [--port 41641]
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import struct
import sys
import time
from pathlib import Path

STUN_SERVERS = [("stun.l.google.com", 19302), ("stun.cloudflare.com", 3478)]
MAGIC = 0x2112A442


def stun_mapping(sock: socket.socket, server: tuple[str, int]) -> tuple[str, int] | None:
    txid = secrets.token_bytes(12)
    request = struct.pack("!HHI", 0x0001, 0, MAGIC) + txid
    addr = socket.gethostbyname(server[0])
    for _ in range(4):
        sock.sendto(request, (addr, server[1]))
        sock.settimeout(1.0)
        try:
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            continue
        if data[8:20] != txid:
            continue
        pos = 20
        while pos + 4 <= len(data):
            kind, length = struct.unpack("!HH", data[pos:pos + 4])
            value = data[pos + 4:pos + 4 + length]
            if kind == 0x0020 and value[1] == 1:  # XOR-MAPPED-ADDRESS, IPv4
                port = struct.unpack("!H", value[2:4])[0] ^ (MAGIC >> 16)
                ip = struct.unpack("!I", value[4:8])[0] ^ MAGIC
                return socket.inet_ntoa(struct.pack("!I", ip)), port
            pos += 4 + length + (-length % 4)
    return None


def punch(sock: socket.socket, peer: tuple[str, int], seconds: float, pings: int) -> dict:
    t0 = time.time()
    sock.settimeout(0.1)
    first_rx, rtts, sent = None, [], 0
    last_send = 0.0
    while time.time() - t0 < seconds:
        now = time.time()
        if now - last_send >= 0.1:
            sock.sendto(b"PUNCH " + struct.pack("!d", now), peer)
            sent += 1
            last_send = now
        try:
            data, src = sock.recvfrom(256)
        except socket.timeout:
            continue
        if src != peer:
            continue
        if first_rx is None:
            first_rx = time.time() - t0
        if data.startswith(b"PUNCH "):
            sock.sendto(b"ECHO  " + data[6:14], peer)
        elif data.startswith(b"ECHO  "):
            rtts.append((time.time() - struct.unpack("!d", data[6:14])[0]) * 1000)
            if len(rtts) >= pings:
                break
    rtts.sort()
    return {"connected": bool(rtts), "first_packet_s": round(first_rx, 3) if first_rx else None,
            "sent": sent, "echoes": len(rtts),
            "rtt_ms_median": round(rtts[len(rtts) // 2], 1) if rtts else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--peer-file", type=Path, required=True)
    ap.add_argument("--port", type=int, default=41641)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--wait", type=float, default=60)
    ap.add_argument("--pings", type=int, default=30)
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    maps = [stun_mapping(sock, s) for s in STUN_SERVERS]
    good = [m for m in maps if m]
    if not good:
        print(json.dumps({"error": "no STUN reply; UDP may be blocked"}))
        return 2
    nat = ("endpoint-independent" if len(good) == 2 and good[0] == good[1]
           else "endpoint-dependent" if len(good) == 2 else "unknown")
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"ip": good[0][0], "port": good[0][1], "nat": nat}))
    os.replace(tmp, args.out)

    deadline = time.time() + args.wait
    while True:
        try:
            peer = json.loads(args.peer_file.read_text())
            break
        except (FileNotFoundError, ValueError):  # scp may still be writing it
            if time.time() > deadline:
                print(json.dumps({"error": "peer address never arrived", "mapped": good[0],
                                  "nat": nat}))
                return 3
            # keep the NAT mapping alive while waiting for the rendezvous
            sock.sendto(b"\0", (socket.gethostbyname(STUN_SERVERS[0][0]), STUN_SERVERS[0][1]))
            time.sleep(0.5)
    result = punch(sock, (peer["ip"], peer["port"]), args.seconds, args.pings)
    print(json.dumps({"mapped": list(good[0]), "nat": nat, "peer": peer, **result}))
    return 0 if result["connected"] else 1


if __name__ == "__main__":
    sys.exit(main())
