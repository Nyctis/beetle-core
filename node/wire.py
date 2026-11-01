"""What a buyer and a seller share: their state directory, how they find each
other, and the messages on a link.

"rendezvous" in config.json picks how two sides exchange addresses, for jobs and
sessions alike: {"kind": "chain"}, or {"kind": "file", "dir": ...} for a directory
both can reach. On chain a job uses its Rendezvous account. Where there is no such
account, a session on a payment channel or a job whose account has closed, the
two sides use memos on that address instead.

Addresses come from what this machine is told: "candidates" for known ones,
"echo_nodes" and "stun" for servers to ask, none of them by default, because
each of these learns this machine's IP. "lan": true adds the local interfaces,
which are otherwise kept from the other side.

Messages are length-prefixed JSON. Big files follow on the same stream through
net/transfer. Nothing here is secret: what crosses a link is ciphertext and the
public job header, so a link only needs to be reliable.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import struct
from pathlib import Path

import node  # noqa: F401  (import paths)

import beetle_crypto as bc
from beetle_chain import Chain, load_keypair
from net import echo
from net.peer import Peer
from net.rendezvous import ChainRendezvous, ChannelRendezvous, FileRendezvous, Rendezvous
from udp_punch import MAGIC as STUN_MAGIC
from udp_punch import stun_mapping

MAX_MESSAGE = 1 << 20
MAX_CANDIDATES = 5  # what fits in an on-chain rendezvous blob
_LENGTH = struct.Struct(">I")

# Offer.hw_class: the low byte is the kind of hardware, the high byte a GPU from this
# list. A seller reports both, as it does mem_gb, and nothing checks them at any tier.
HW_CPU, HW_NVIDIA, HW_APPLE = 0, 1, 2
GPU_MODELS = ("", "T4", "L4", "A2", "A10", "A100", "L40S", "H100", "H200")


def hardware_class(gpu: dict | None) -> int:
    """`gpu` is what the driver reports: {"name": "NVIDIA A100-SXM4-80GB", ...}."""
    if gpu is None:
        return HW_CPU
    model = next((i for i, name in enumerate(GPU_MODELS) if name and re.search(rf"\b{name}\b", gpu["name"])), 0)
    return HW_NVIDIA | model << 8


def gpu_name(hw_class: int) -> str | None:
    kind, model = hw_class & 0xff, hw_class >> 8
    if kind == HW_APPLE:
        return "Apple"
    if kind != HW_NVIDIA:
        return None
    return f"NVIDIA {GPU_MODELS[model]}" if 0 < model < len(GPU_MODELS) else "NVIDIA"


def home() -> Path:
    return Path(os.environ.get("BEETLE_HOME", "~/.beetle")).expanduser()


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.chmod(0o600)
    os.replace(tmp, path)


def load_config(where: Path | None = None) -> dict:
    cfg = read_json((where or home()) / "config.json")
    if cfg is None:
        raise SystemExit(f"no config in {where or home()}: run `beetle init` or `beetle host init`")
    return cfg


def load_chain(cfg: dict, where: Path | None = None) -> Chain:
    """The network named by config.json. Every digest binds its program and cluster."""
    chain = Chain(read_json((where or home()) / cfg["chain"]))
    bc.configure(bytes(chain.program_id), chain.chain_ref.hex().encode())
    return chain


def load_wallet(where: Path | None = None):
    return load_keypair((where or home()) / "wallet.json")


def new_wallet(where: Path | None = None):
    """Creates the wallet once. It is a plain file: hot, and only as safe as this machine."""
    from solders.keypair import Keypair

    path = (where or home()) / "wallet.json"
    if not path.exists():
        write_json(path, list(bytes(Keypair())))
    return load_keypair(path)


def secret(where: Path, name: str) -> bytes:
    """A 32-byte secret kept in a file; made on first use."""
    path = where / name
    if not path.exists():
        path.write_bytes(os.urandom(32))
        path.chmod(0o600)
    return path.read_bytes()


def file_rendezvous(directory: str, session: str, role: str) -> FileRendezvous:
    d = Path(directory).expanduser() / session
    mine, theirs = ("buyer.json", "seller.json") if role == "buyer" else ("seller.json", "buyer.json")
    return FileRendezvous(d / mine, d / theirs)


def job_rendezvous(cfg: dict, job_id: bytes, role: str, chain, signer, where: Path | None = None) -> Rendezvous:
    section = cfg["rendezvous"]
    if section["kind"] == "file":
        return file_rendezvous(section["dir"], job_id.hex(), role)
    if section["kind"] != "chain":
        raise ValueError(f"unknown rendezvous {section['kind']!r}")
    if role == "buyer":
        return ChainRendezvous.buyer(chain, job_id, signer)
    return ChainRendezvous.seller(chain, job_id, signer, secret(where, "net.key"))


def buyer_session(cfg: dict, channel, chain, wallet, payee, net_pk: bytes) -> Rendezvous:
    section = cfg["rendezvous"]
    if section["kind"] == "file":
        return file_rendezvous(section["dir"], str(channel), "buyer")
    return ChannelRendezvous.buyer(chain, channel, wallet, payee, net_pk)


def seller_session(cfg: dict, channel, chain, wallet, where: Path) -> Rendezvous:
    section = cfg["rendezvous"]
    if section["kind"] == "file":
        return file_rendezvous(section["dir"], str(channel.address), "seller")
    return ChannelRendezvous.seller(chain, channel.address, wallet, channel.payer, secret(where, "net.key"))


def closed_job_rendezvous(cfg: dict, job_id: bytes, role: str, chain, wallet, other,
                          net_pk: bytes | None = None, where: Path | None = None) -> Rendezvous:
    """To reach a result after the job's account is gone. `other` is the far wallet."""
    section = cfg["rendezvous"]
    if section["kind"] == "file":
        return file_rendezvous(section["dir"], job_id.hex(), role)
    about = chain.job_pda(job_id)
    if role == "buyer":
        return ChannelRendezvous.buyer(chain, about, wallet, other, net_pk)
    return ChannelRendezvous.seller(chain, about, wallet, other, secret(where, "net.key"))


def stun_servers(cfg: dict) -> list[tuple[str, int]]:
    return [(s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in cfg.get("stun", [])]


def map_port(cfg: dict, port: int = 0) -> tuple[int, str | None]:
    """Ask STUN what this NAT maps `port` to (0: any free port), then free the port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
        port = sock.getsockname()[1]
        for server in stun_servers(cfg):
            try:
                mapped = stun_mapping(sock, server)
            except OSError:
                continue
            if mapped:
                return port, f"{mapped[0]}:{mapped[1]}"
        return port, None
    except OSError:  # taken: binding the peer will say so properly
        return port, None
    finally:
        sock.close()


async def bound_peer(cfg: dict, port: int = 0, **options) -> Peer:
    """A peer whose public port is known. iroh keeps its socket to itself, so STUN runs
    on that port just before iroh takes it: a NAT that maps by source port (nearly all
    do) gives iroh the same mapping. A symmetric NAT doesn't, and nothing here helps."""
    mapped = None
    if stun_servers(cfg):
        port, mapped = await asyncio.to_thread(map_port, cfg, port)
    peer = await Peer.bind(port, **options)
    peer.mapped = mapped
    return peer


def public_ip(cfg: dict) -> str | None:
    """STUN on its own socket, so this gives the IP but not iroh's own mapping; for a
    peer that didn't come from bound_peer. net/echo is the exact way, when an echo
    node's inbound UDP is open."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for server in stun_servers(cfg):
            try:
                mapped = stun_mapping(sock, server)
            except OSError:
                continue
            if mapped:
                return mapped[0]
    finally:
        sock.close()
    return None


def nat_type(cfg: dict) -> dict:
    """What two STUN servers say about this network: open, eim, edm, unknown or blocked."""
    if not stun_servers(cfg):
        return {"nat": "not probed: no STUN server configured", "observed": []}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 0))
    port = sock.getsockname()[1]
    seen = []
    try:
        for server in stun_servers(cfg)[:2]:
            try:
                mapped = stun_mapping(sock, server)
            except OSError:
                mapped = None
            if mapped:
                seen.append(f"{mapped[0]}:{mapped[1]}")
        local = [f"{socket.gethostbyname(socket.gethostname())}:{sock.getsockname()[1]}"]
    except OSError:
        local = []
    finally:
        sock.close()
    return {"nat": echo.classify(seen, local), "observed": seen,
            "port_changed": any(int(a.rsplit(":", 1)[1]) != port for a in seen)}


STUN_WAIT = 1.0
CHANGE_REQUEST = 0x0003
CHANGE_IP_AND_PORT = 0x06
CAN_CHANGE = (0x0005, 0x802C)  # CHANGED-ADDRESS, OTHER-ADDRESS: this server has a second address


def _stun_ask(sock: socket.socket, server: tuple[str, int], body: bytes = b"") -> tuple[bytes, tuple] | None:
    """One request, three tries. The answer and where it came from, or None."""
    txid = os.urandom(12)
    for _ in range(3):
        sock.sendto(struct.pack("!HHI", 1, len(body), STUN_MAGIC) + txid + body, server)
        try:
            data, source = sock.recvfrom(2048)
        except TimeoutError:
            continue
        if data[8:20] == txid:
            return data, source
    return None


def _stun_attrs(data: bytes) -> set[int]:
    kinds, pos = set(), 20
    while pos + 4 <= len(data):
        kind, length = struct.unpack("!HH", data[pos:pos + 4])
        kinds.add(kind)
        pos += 4 + length + (-length % 4)
    return kinds


def inbound_open(cfg: dict, port: int) -> bool | None:
    """Whether a packet from an address this port never wrote to gets in, tested by a
    STUN server that has a second address (RFC 5780) answering from it. True or False
    only when one such server took part. None when none did, which is most of them, or
    when the daemon holds the port. Nobody can tell from inside: a packet sent to one's
    own public address may never leave the building."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
        sock.settimeout(STUN_WAIT)
        for host, server_port in stun_servers(cfg):
            try:
                server = (socket.gethostbyname(host), server_port)
                plain = _stun_ask(sock, server)
                if plain is None or not _stun_attrs(plain[0]) & set(CAN_CHANGE):
                    continue
                changed = _stun_ask(sock, server, struct.pack("!HHI", CHANGE_REQUEST, 4, CHANGE_IP_AND_PORT))
            except OSError:
                continue
            return changed is not None and changed[1] != server
        return None
    except OSError:
        return None
    finally:
        sock.close()


NAT_ADVICE = {
    "open": "no NAT in the way",
    "eim": "a NAT that keeps one mapping per port: sellers with or without a public port are reachable",
    "edm": "a symmetric NAT: only a seller with a public port is reachable, and no relay helps",
    "unknown": "one STUN answer, so the NAT type is unknown; two servers would tell",
    "blocked": "no STUN answer: UDP may be blocked, and no direct link will come up",
}


def nat_advice(nat: dict, echo_nodes: int = 0) -> str:
    """What this network means for reaching sellers, said plainly."""
    kind = nat.get("nat", "")
    text = NAT_ADVICE.get(kind, kind)
    if nat.get("port_changed"):
        text += ("; this NAT changes ports, and only the port asked about just before a link is known, "
                 f"so a long-lived seller here goes stale without an echo node ({echo_nodes} on chain)")
    return text


async def candidates(peer, cfg: dict, echo_nodes: list[dict] = ()) -> list[str]:
    """This endpoint's addresses, best first: given ones, then what echo nodes saw
    on this very socket, then what STUN saw on this port, then the machine's own
    interfaces."""
    addrs = list(cfg.get("candidates", []))
    nodes = [*cfg.get("echo_nodes", []), *echo_nodes]
    if nodes:
        addrs += (await echo.probe(peer, nodes))["observed"]
    if getattr(peer, "mapped", None):
        addrs.append(peer.mapped)
    elif ip := await asyncio.to_thread(public_ip, cfg):
        addrs.append(f"{ip}:{peer.port}")
    if cfg.get("lan") or not addrs:
        addrs += peer.addrs
    return list(dict.fromkeys(addrs))[:MAX_CANDIDATES]


def path_of(link) -> dict:
    """How this link runs right now. What is observed here, not proven."""
    for p in link.conn.paths():
        if p.is_selected:
            return {"direct": not p.is_relay, "remote_addr": p.remote_addr, "rtt_ms": p.rtt_ms}
    return {"direct": None, "remote_addr": None, "rtt_ms": None}


async def send_msg(link, msg: dict) -> None:
    body = bc.canonical_json(msg)
    await link.write(_LENGTH.pack(len(body)) + body)


async def recv_msg(link) -> dict:
    n = _LENGTH.unpack(await link.read_exact(_LENGTH.size))[0]
    if n > MAX_MESSAGE:
        raise ValueError("message too large")
    return json.loads(await link.read_exact(n))


def ack_digest(job_id: bytes, adapter_ct_hash: str) -> bytes:
    """What the buyer signs to say it holds the ciphertext."""
    return bc.digest("delivered", job_id + bytes.fromhex(adapter_ct_hash))


def taken_digest(job_id: bytes) -> bytes:
    """What the buyer signs to say the seller can delete the result."""
    return bc.digest("taken", job_id)
