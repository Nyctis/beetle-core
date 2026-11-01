"""Two separate processes meet through the chain, then link directly with iroh.

The buyer and the seller share nothing but chain.json and a job id: all they
learn about each other is read from the Rendezvous account. Measures how long
each leg takes. After scripts/setup_chain.sh, from the beetle directory:

    uv run --extra net python tests/chain_rendezvous_pair.py --rounds 10

Each round is a fresh job, and both ends use 127.0.0.1. To cross a real network,
run the seller on another machine that can reach the same chain:

    ... --seller-ssh HOST --seller-dir DIR --seller-python PY --addr PUBLIC_IP:PORT --port PORT \\
        --stun STUN_HOST:PORT --local-port PORT

The seller then advertises that address, and the buyer its interface addresses
plus what --stun reports. STUN uses its own socket, so that assumes the buyer's
NAT keeps the local port. Throwaway keys throughout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "common")]

import beetle_crypto as bc  # noqa: E402
from beetle_chain import (KIND_TRAINING, Chain, OfferTerms, load_keypair, net_keypair,  # noqa: E402
                          new_job_id)
from job_defaults import TIERS  # noqa: E402
from net.peer import Peer  # noqa: E402
from net.rendezvous import ChainRendezvous  # noqa: E402
from solders.keypair import Keypair  # noqa: E402

USDC = 1_000_000
UNITS = 1_000_000


def kp_hex(kp: Keypair) -> str:
    return bytes(kp).hex()


def kp_from(s: str) -> Keypair:
    return Keypair.from_bytes(bytes.fromhex(s))


def lan_addr(peer: Peer, host: str) -> str:
    return f"{host}:{peer.port}"


def stun_address(args) -> list[str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from udp_punch import stun_mapping

    host, port = args.stun.rsplit(":", 1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("0.0.0.0", args.local_port))
        ip, mapped = stun_mapping(sock, (host, int(port)))
    return [f"{ip}:{mapped}"]


async def seller(args) -> None:
    chain = Chain.from_file(args.chain)
    op = kp_from(args.operator)
    rdv = ChainRendezvous.seller(chain, bytes.fromhex(args.job), op, bytes.fromhex(args.net_secret),
                                 poll=args.poll)
    peer = await Peer.bind(port=args.port)
    print("READY", flush=True)
    them = await rdv.wait(args.timeout)
    t_offer = time.time()
    await rdv.publish({"id": peer.id, "addrs": [args.addr or lan_addr(peer, "127.0.0.1")]})
    link = await peer.link(them["id"], them["addrs"], initiator=False, window=args.window)
    t_link = time.time()
    await link.write(await link.read_exact(16))  # echo, so the buyer sees real traffic
    print("RESULT " + json.dumps({"offer_seen": t_offer, "linked": t_link, "their_id": them["id"]}), flush=True)
    await asyncio.sleep(0.5)
    await peer.close()


async def buyer(args, chain: Chain, requester: Keypair, job_id: bytes) -> dict:
    rdv = ChainRendezvous.buyer(chain, job_id, requester, poll=args.poll)
    public = stun_address(args) if args.stun else []
    peer = await Peer.bind(port=args.local_port)
    t0 = time.time()
    addrs = public + peer.addrs if args.seller_ssh else [lan_addr(peer, "127.0.0.1")]
    await rdv.publish({"id": peer.id, "addrs": addrs})
    t_offer = time.time()
    them = await rdv.wait(args.timeout)
    t_answer = time.time()
    link = await peer.link(them["id"], them["addrs"], initiator=True, window=args.window)
    t_link = time.time()
    ping = os.urandom(16)
    await link.write(ping)
    assert await link.read_exact(16) == ping, "the echo came back different"
    t_echo = time.time()
    rtt = link.rtt_ms()
    await peer.close()
    await rdv.close()
    return {"t0": t0, "offer_landed": t_offer, "answer_seen": t_answer, "linked": t_link,
            "echoed": t_echo, "rtt_ms": rtt, "their_id": them["id"], "my_id": peer.id}


def setup(args, chain: Chain) -> tuple:
    mint_authority = load_keypair(ROOT / args.mint_authority)
    buyer_kp, seller_kp, op = Keypair(), Keypair(), Keypair()
    for kp in (buyer_kp, seller_kp, op):
        chain.airdrop(kp.pubkey())
        chain.create_ata(kp, kp.pubkey())
    chain.mint_test_usdc(mint_authority, buyer_kp.pubkey(), 100 * USDC)
    chain.mint_test_usdc(mint_authority, seller_kp.pubkey(), 100 * USDC)
    secret = os.urandom(32)
    chain.register_seller(seller_kp, op.pubkey(), net_keypair(secret).public_bytes)
    chain.stake(seller_kp, 50 * USDC)
    chain.post_offer(op, seller_kp.pubkey(), 0, OfferTerms(
        KIND_TRAINING, TIERS["mock"], bytes(32), price_a=USDC // 100, slots_total=64,
        valid_until=chain.chain_time() + 86_400))
    return buyer_kp, op, secret, chain.get_offer(chain.offer_pda(seller_kp.pubkey(), 0))


def open_job(chain: Chain, buyer_kp: Keypair, offer) -> bytes:
    nonce, slot = os.urandom(32), chain.current_slot()
    job_id = new_job_id(buyer_kp.pubkey(), nonce, slot)
    chain.open_job(buyer_kp, buyer_kp, offer, job_id, nonce, slot, os.urandom(32), UNITS)
    return job_id


def pct(values: list[float], q: float) -> float:
    return sorted(values)[min(len(values) - 1, int(q * len(values)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default=str(ROOT / "chain.json"))
    ap.add_argument("--mint-authority", default=".keys/deployer.json")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--poll", type=float, default=0.25)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--window", type=float, default=15.0)
    ap.add_argument("--addr", help="address the seller advertises (default: loopback)")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--local-port", type=int, default=0, help="the buyer's UDP port")
    ap.add_argument("--stun", help="STUN server for the buyer's public address")
    ap.add_argument("--seller-ssh", help="run the seller on this ssh host")
    ap.add_argument("--seller-dir", default="beetle")
    ap.add_argument("--seller-python", default="python")
    ap.add_argument("--role-seller", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--job"), ap.add_argument("--operator"), ap.add_argument("--net-secret")
    args = ap.parse_args()
    if args.role_seller:
        asyncio.run(seller(args))
        return 0

    chain = Chain.from_file(args.chain)
    bc.configure(bytes(chain.program_id), chain.chain_ref.hex().encode())
    buyer_kp, op, secret, offer = setup(args, chain)
    rows = []
    for i in range(args.rounds):
        job_id = open_job(chain, buyer_kp, offer)
        flags = ["--role-seller", "--job", job_id.hex(), "--operator", kp_hex(op),
                 "--net-secret", secret.hex(), "--poll", str(args.poll), "--timeout", str(args.timeout),
                 "--window", str(args.window), "--port", str(args.port)]
        flags += ["--addr", args.addr] if args.addr else []
        if args.seller_ssh:
            remote = shlex.join([args.seller_python, "tests/chain_rendezvous_pair.py", "--chain",
                                 "chain.json", *flags])
            cmd = ["ssh", "-o", "BatchMode=yes", args.seller_ssh, f"cd {args.seller_dir} && {remote}"]
        else:
            cmd = [sys.executable, __file__, "--chain", args.chain, *flags]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
        assert proc.stdout.readline().strip() == "READY"
        result = asyncio.run(buyer(args, chain, buyer_kp, job_id))
        side = next(line for line in proc.stdout if line.startswith("RESULT"))
        proc.wait(timeout=10)
        s = json.loads(side.split(" ", 1)[1])
        assert s["their_id"] == result["my_id"], "the seller met someone else"
        row = {"offer_tx": result["offer_landed"] - result["t0"],
               "buyer_sees_answer": result["answer_seen"] - result["t0"],
               "link_up": result["linked"] - result["answer_seen"],
               "first_echo": result["echoed"] - result["t0"]}
        rows.append(row)
        print(f"round {i + 1}: " + "  ".join(f"{k} {v:.3f}" for k, v in row.items())
              + f"  rtt {result['rtt_ms']} ms", flush=True)
    print(f"\n{len(rows)} rounds, seconds (median / p95 / max):")
    for k in rows[0]:
        vals = [r[k] for r in rows]
        print(f"  {k:18} {statistics.median(vals):.3f} / {pct(vals, 0.95):.3f} / {max(vals):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
