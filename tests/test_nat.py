"""What a machine learns about its NAT, and what it publishes."""

from __future__ import annotations

import asyncio
import socket
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import node  # noqa: E402,F401
from node import cli, wire  # noqa: E402

MAGIC = 0x2112A442
PUBLIC = ("203.0.113.5", 40000)


def answer(request: bytes, second: bool = False) -> bytes:
    ip = struct.unpack(">I", socket.inet_aton(PUBLIC[0]))[0] ^ MAGIC
    value = struct.pack(">BBHI", 0, 1, PUBLIC[1] ^ (MAGIC >> 16), ip)
    attr = struct.pack(">HH", 0x0020, len(value)) + value
    if second:  # OTHER-ADDRESS
        attr += struct.pack(">HH", 0x802C, 8) + struct.pack(">BBH", 0, 1, 3479) + socket.inet_aton("192.0.2.9")
    return struct.pack(">HHI", 0x0101, len(attr), MAGIC) + request[8:20] + attr


class Stun:
    """A STUN server on loopback. `second`: it says it has another address. `gets_through`:
    an answer from there reaches the caller, or the caller's firewall stops it."""

    def __init__(self, second=False, gets_through=True):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.spare = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.spare.bind(("127.0.0.1", 0))
        self.second, self.gets_through = second, gets_through
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            try:
                data, source = self.sock.recvfrom(2048)
            except OSError:
                return
            changing = len(data) > 20  # the only attribute these requests carry
            if not self.second or not changing:
                self.sock.sendto(answer(data, self.second), source)
            elif self.gets_through:
                self.spare.sendto(answer(data, True), source)

    @property
    def cfg(self) -> dict:
        return {"stun": [f"127.0.0.1:{self.sock.getsockname()[1]}"]}


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("0.0.0.0", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_the_mapping_of_the_port_iroh_will_take_is_what_gets_published():
    server = Stun()
    port, mapped = wire.map_port(server.cfg)
    assert mapped == f"{PUBLIC[0]}:{PUBLIC[1]}" and port != PUBLIC[1]
    again = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    again.bind(("0.0.0.0", port))  # it was freed for iroh
    again.close()


def test_a_taken_port_is_left_for_the_peer_to_report():
    held = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    held.bind(("0.0.0.0", 0))
    port = held.getsockname()[1]
    assert wire.map_port(Stun().cfg, port) == (port, None)
    held.close()


def test_candidates_trust_the_mapping_over_the_local_port(monkeypatch):
    monkeypatch.setattr(wire, "public_ip", lambda cfg: pytest.fail("no second STUN query"))
    peer = SimpleNamespace(mapped="203.0.113.5:40000", port=41000, addrs=["10.0.0.2:41000"])
    assert asyncio.run(wire.candidates(peer, {})) == ["203.0.113.5:40000"]


def test_a_peer_nobody_asked_about_still_gets_the_old_guess(monkeypatch):
    monkeypatch.setattr(wire, "public_ip", lambda cfg: "203.0.113.5")
    peer = SimpleNamespace(mapped=None, port=41000, addrs=[])
    assert asyncio.run(wire.candidates(peer, {})) == ["203.0.113.5:41000"]


def test_inbound_takes_a_server_with_a_second_address(monkeypatch):
    monkeypatch.setattr(wire, "STUN_WAIT", 0.1)
    assert wire.inbound_open(Stun(second=True).cfg, free_port()) is True
    assert wire.inbound_open(Stun(second=True, gets_through=False).cfg, free_port()) is False
    assert wire.inbound_open(Stun().cfg, free_port()) is None  # most servers: can't tell


def seller(public_udp, proved=None):
    host = {"public_addr": "203.0.113.5:41710", "port": 41710}
    return SimpleNamespace(public_udp=public_udp), host


@pytest.mark.parametrize("flag, proved, listed", [
    (None, True, True),     # proved: on by itself
    (None, None, False),    # unknown: stays off
    (None, False, False),
    (False, True, False),   # the operator's no is final
    (True, True, True),
    (True, None, True),     # unknown: on the operator's word
])
def test_public_udp_follows_the_proof(monkeypatch, flag, proved, listed):
    monkeypatch.setattr(wire, "inbound_open", lambda cfg, port: proved)
    a, host = seller(flag)
    assert cli.public_udp(a, {}, host) is listed


def test_a_claim_the_proof_contradicts_is_refused(monkeypatch):
    monkeypatch.setattr(wire, "inbound_open", lambda cfg, port: False)
    a, host = seller(True)
    with pytest.raises(SystemExit, match="closed"):
        cli.public_udp(a, {}, host)


def test_public_udp_needs_an_address():
    assert cli.public_udp(SimpleNamespace(public_udp=None), {}, {}) is False
    with pytest.raises(SystemExit, match="public-addr"):
        cli.public_udp(SimpleNamespace(public_udp=True), {}, {})


def test_doctor_says_what_a_port_changing_nat_means():
    said = wire.nat_advice({"nat": "eim", "port_changed": True}, 0)
    assert "changes ports" in said and "0 on chain" in said
    assert "changes ports" not in wire.nat_advice({"nat": "eim", "port_changed": False})
    assert "symmetric" in wire.nat_advice({"nat": "edm"})


def test_nat_type_notices_a_changed_port():
    assert wire.nat_type(Stun().cfg)["port_changed"] is True
