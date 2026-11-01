"""ChainRendezvous against an in-memory stand-in for the program, and the blob
format. What the program itself enforces is in scripts/chain_attack_test.py; a
real pair over a real chain is tests/chain_rendezvous_pair.py.

Run from the beetle directory:  uv run --with pytest pytest -q tests/test_rendezvous.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "common")]

import beetle_crypto as bc  # noqa: E402
import beetle_chain as bch  # noqa: E402
from beetle_chain import ROLE_BUYER, ROLE_SELLER, Chain, RendezvousAccount  # noqa: E402
from net.rendezvous import ChainRendezvous, RendezvousFailed  # noqa: E402
from solders.keypair import Keypair  # noqa: E402

CONTACT = {"id": "ab" * 32, "addrs": ["203.0.113.1:4000", "[2001:db8::1]:4001"]}
THEIRS = {"id": "cd" * 32, "addrs": ["198.51.100.2:5000"]}


class FakeChain(Chain):
    """The real sealing code; the chain's state is a dict and rules are the program's."""

    def __init__(self, net_pk: bytes):
        super().__init__({"rpc_url": "http://127.0.0.1:1", "chain_ref": os.urandom(32).hex(),
                          "program_id": str(Keypair().pubkey()), "mint": str(Keypair().pubkey())})
        self.net_pk, self.rdv, self.calls, self.clock = net_pk, None, [], 1000
        self.host = Keypair().pubkey()
        self.job_exists = True

    def get_job(self, job_id):
        return SimpleNamespace(host=self.host) if self.job_exists else None

    def get_seller(self, authority):
        return SimpleNamespace(net_pk=self.net_pk)

    def get_rendezvous(self, job_id):
        return self.rdv

    def rdv_open(self, requester, job_id, offer, ttl_secs=600, policy=0, payer=None):
        self.calls.append(("open", 1))
        self.clock += 1
        self.rdv = RendezvousAccount(self.rdv_pda(job_id), job_id, requester.pubkey(), self.host,
                                     requester.pubkey(), self.clock, self.clock + ttl_secs, policy, 1,
                                     "Offered", offer, bytes(224))

    def rdv_answer(self, operator, rdv, round_, answer):
        self.calls.append(("answer", round_))
        self.rdv = dataclasses.replace(self.rdv, answer=answer, state="Answered")

    def rdv_reoffer(self, requester, rdv, round_, offer):
        self.calls.append(("reoffer", round_))
        self.clock += 1
        self.rdv = dataclasses.replace(self.rdv, opened_at=self.clock, round=round_, state="Offered",
                                       offer=offer, answer=bytes(224))

    def rdv_fail(self, caller, rdv, reason=0):
        self.calls.append(("fail", reason))
        self.rdv = dataclasses.replace(self.rdv, state="Failed")

    def rdv_close(self, caller, rdv):
        self.calls.append(("close", 0))
        self.rdv = None


@pytest.fixture
def pair():
    secret = os.urandom(32)
    chain = FakeChain(bch.net_keypair(secret).public_bytes)
    job_id = os.urandom(32)
    fast = {"poll": 0.001}
    return (chain, ChainRendezvous.buyer(chain, job_id, Keypair(), **fast),
            ChainRendezvous.seller(chain, job_id, Keypair(), secret, **fast))


async def exchange(buyer, seller) -> tuple[dict, dict]:
    """Both sides in the order the client and the gateway call them."""
    async def buy():
        await buyer.publish(CONTACT)
        return await buyer.wait(2)

    async def sell():
        them = await seller.wait(2)
        await seller.publish(THEIRS)
        return them

    return await asyncio.gather(buy(), sell())


# --- The blob -----------------------------------------------------------------------


def plain(**kw) -> bytes:
    fields = {"role": ROLE_BUYER, "round_": 1, "endpoint_id": b"\x11" * 32, "not_after": 77,
              "addrs": ["203.0.113.1:4000"], "reply_pk": b"\x22" * 32, "nat_hint": 2, "flags": 1, **kw}
    return bch.rdv_plaintext(**fields)


def test_plaintext_layout_is_the_published_one():
    p = plain()
    assert len(p) == bch.RDV_PLAIN_LEN == 176
    assert list(p[:6]) == [1, ROLE_BUYER, 1, 2, 1, 1] and p[6:8] == bytes(2)
    assert p[8:40] == b"\x11" * 32 and p[40:72] == b"\x22" * 32
    assert struct.unpack("<q", p[72:80])[0] == 77
    assert p[80] == 4 and p[81:85] == bytes([203, 0, 113, 1]) and p[85:97] == bytes(12)
    assert struct.unpack("<H", p[97:99])[0] == 4000
    assert p[99:] == bytes(77)


def test_parse_reverses_it():
    f = bch.rdv_parse(plain(addrs=CONTACT["addrs"]))
    assert f["id"] == "11" * 32 and f["reply_pk"] == b"\x22" * 32
    assert (f["role"], f["round"], f["nat_hint"], f["flags"], f["not_after"]) == (ROLE_BUYER, 1, 2, 1, 77)
    assert f["addrs"] == CONTACT["addrs"]


def test_size_hides_the_number_of_candidates():
    assert len(plain(addrs=[])) == len(plain(addrs=["203.0.113.1:1"] * 5)) == 176


def test_only_five_candidates_fit():
    addrs = [f"203.0.113.{i}:1" for i in range(1, 9)]
    assert bch.rdv_parse(plain(addrs=addrs))["addrs"] == addrs[:5]


def test_hostnames_are_refused():
    with pytest.raises(ValueError):
        plain(addrs=["relay.example.com:443"])


@pytest.mark.parametrize("bad", [b"", bytes(176), b"\x02" + bytes(175), bytes([1, 0, 1, 0, 0, 6]) + bytes(170)])
def test_garbage_does_not_parse(bad):
    with pytest.raises(ValueError):
        bch.rdv_parse(bad)


def test_public_contact_of_an_offer():
    endpoint = bch.public_endpoint(b"\x33" * 32, "203.0.113.5:4433").ljust(64, b"\0")
    o = SimpleNamespace(endpoint=endpoint)
    assert bch.Offer.public_contact.fget(o) == {"id": "33" * 32, "addrs": ["203.0.113.5:4433"]}
    assert bch.Offer.public_contact.fget(SimpleNamespace(endpoint=bytes(64))) is None


def test_net_key_comes_from_its_secret():
    secret = os.urandom(32)
    assert bch.net_keypair(secret).public_bytes == bch.net_keypair(secret).public_bytes


def test_sealed_blob_size_and_binding(pair):
    chain, _, _ = pair
    key = bc.HpkeKeyPair()
    job = os.urandom(32)
    blob = chain.rdv_seal(key.public_bytes, job, ROLE_BUYER, 1, plain())
    assert len(blob) == bch.RDV_BLOB_LEN == 224
    assert chain.rdv_unseal(key, job, ROLE_BUYER, 1, blob)["id"] == "11" * 32
    for j, role, round_ in ((os.urandom(32), ROLE_BUYER, 1), (job, ROLE_SELLER, 1), (job, ROLE_BUYER, 2)):
        with pytest.raises(Exception):
            chain.rdv_unseal(key, j, role, round_, blob)


# --- The exchange --------------------------------------------------------------------


def test_contacts_cross(pair):
    chain, buyer, seller = pair

    async def main():
        got_buyer, got_seller = await exchange(buyer, seller)
        await asyncio.sleep(0.05)  # the answer goes out in the background
        return got_buyer, got_seller

    got_buyer, got_seller = asyncio.run(main())
    assert got_seller["id"] == CONTACT["id"] and got_seller["addrs"] == CONTACT["addrs"]
    assert got_buyer["id"] == THEIRS["id"] and got_buyer["addrs"] == THEIRS["addrs"]
    assert chain.calls == [("open", 1), ("answer", 1)]


def test_seller_may_publish_first(pair):
    chain, buyer, seller = pair

    async def main():
        await seller.publish(THEIRS)
        waiting = asyncio.create_task(seller.wait(2))
        await buyer.publish(CONTACT)
        return await asyncio.gather(buyer.wait(2), waiting)

    got_buyer, got_seller = asyncio.run(main())
    assert got_buyer["id"] == THEIRS["id"] and got_seller["id"] == CONTACT["id"]


def test_the_seller_does_not_wait_for_its_answer_to_land(pair):
    chain, buyer, seller = pair
    slow = chain.rdv_answer

    def rdv_answer(*args):
        time.sleep(0.3)
        slow(*args)

    chain.rdv_answer = rdv_answer

    async def main():
        await buyer.publish(CONTACT)
        await seller.wait(2)
        t0 = time.perf_counter()
        await seller.publish(THEIRS)
        took = time.perf_counter() - t0
        await buyer.wait(2)
        return took

    assert asyncio.run(main()) < 0.1  # it can start dialing straight away


def test_each_publication_is_returned_once(pair):
    chain, buyer, seller = pair

    async def main():
        await exchange(buyer, seller)
        with pytest.raises(asyncio.TimeoutError):
            await buyer.wait(0.05)
        with pytest.raises(asyncio.TimeoutError):
            await seller.wait(0.05)

    asyncio.run(main())


def test_reconnects_take_a_new_round_then_start_over(pair):
    chain, buyer, seller = pair

    async def main():
        for _ in range(3):
            await exchange(buyer, seller)
            await asyncio.sleep(0.01)

    asyncio.run(main())
    assert [c for c in chain.calls if c[0] != "answer"] == [("open", 1), ("reoffer", 2), ("open", 1)]


def test_silence_is_a_timeout(pair):
    _, buyer, seller = pair

    async def main():
        await buyer.publish(CONTACT)
        with pytest.raises(asyncio.TimeoutError):
            await buyer.wait(0.05)

    asyncio.run(main())


def test_an_expired_offer_is_ignored(pair):
    chain, buyer, seller = pair
    buyer.ttl = -1

    async def main():
        await buyer.publish(CONTACT)
        with pytest.raises(asyncio.TimeoutError):
            await seller.wait(0.05)

    asyncio.run(main())


def test_the_buyer_hears_when_the_seller_gives_up(pair):
    chain, buyer, seller = pair

    async def main():
        await buyer.publish(CONTACT)
        await seller.fail()
        with pytest.raises(RendezvousFailed):
            await buyer.wait(1)

    asyncio.run(main())


def test_an_offer_that_does_not_open_fails_the_rendezvous(pair):
    chain, buyer, seller = pair

    async def main():
        await buyer.publish(CONTACT)
        chain.rdv = dataclasses.replace(chain.rdv, offer=os.urandom(224))
        with pytest.raises(RendezvousFailed):
            await seller.wait(1)

    asyncio.run(main())
    assert chain.rdv.state == "Failed"


def test_a_seller_without_a_net_key_is_refused(pair):
    chain, buyer, _ = pair
    chain.net_pk = bytes(32)
    with pytest.raises(RendezvousFailed, match="net_pk"):
        asyncio.run(buyer.publish(CONTACT))


def test_a_missing_job_is_refused(pair):
    chain, buyer, _ = pair
    chain.job_exists = False
    with pytest.raises(RendezvousFailed, match="no such job"):
        asyncio.run(buyer.publish(CONTACT))


def test_a_transaction_that_lands_late_is_not_sent_twice(pair):
    chain, buyer, _ = pair
    real = chain.rdv_open

    def lands_but_times_out(*args, **kw):
        real(*args, **kw)
        raise TimeoutError

    chain.rdv_open = lands_but_times_out
    asyncio.run(buyer.publish(CONTACT))
    assert chain.calls == [("open", 1)]


def test_close_returns_the_deposit(pair):
    chain, buyer, seller = pair

    async def main():
        await buyer.publish(CONTACT)
        await buyer.close()

    asyncio.run(main())
    assert chain.calls[-1] == ("close", 0) and chain.rdv is None
