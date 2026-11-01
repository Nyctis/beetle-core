"""An inference session finds its seller through memos: only the two parties
count, nothing is readable by anyone else, and a reconnect is one more round."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import node  # noqa: E402,F401

import beetle_chain as bch  # noqa: E402
import pytest  # noqa: E402
from net.rendezvous import ChannelRendezvous  # noqa: E402
from solders.keypair import Keypair  # noqa: E402


class Ledger(bch.Chain):
    """The chain's sealing, with an in-memory ledger for memos."""

    def __init__(self):
        super().__init__({"rpc_url": "http://127.0.0.1:1", "program_id": str(Keypair().pubkey()),
                          "mint": str(Keypair().pubkey()), "chain_ref": os.urandom(32).hex()})
        self.posted = []

    def post_memo(self, signer, about, data):
        self.posted.append((signer.pubkey(), about, data))
        return "sig"

    def memos(self, about, signer, seen, limit=20):
        fresh = [(i, d) for i, (s, a, d) in enumerate(self.posted)
                 if s == signer and a == about and i not in seen]
        seen.update(i for i, _ in fresh)
        return [d for _, d in fresh]


def contact(n: int) -> dict:
    return {"id": bytes([n]).ljust(32, b"\1").hex(), "addrs": [f"203.0.113.{n}:41710"]}


def parties():
    chain, channel, buyer, seller, secret = Ledger(), Keypair().pubkey(), Keypair(), Keypair(), os.urandom(32)
    pk = bch.net_keypair(secret).public_bytes
    return (chain, channel, buyer, seller, secret,
            ChannelRendezvous.buyer(chain, channel, buyer, seller.pubkey(), pk, poll=0.01),
            ChannelRendezvous.seller(chain, channel, seller, buyer.pubkey(), secret, poll=0.01))


async def one_round(buyer, seller, b: dict, s: dict) -> tuple[dict, dict]:
    await buyer.publish(b)
    offer = await seller.wait(2)
    await seller.publish(s)
    return offer, await buyer.wait(2)


def test_both_sides_learn_each_other():
    async def run():
        *_, buyer, seller = parties()
        offer, answer = await one_round(buyer, seller, contact(1), contact(2))
        assert (offer["id"], offer["addrs"]) == (contact(1)["id"], contact(1)["addrs"])
        assert (answer["id"], answer["addrs"]) == (contact(2)["id"], contact(2)["addrs"])

    asyncio.run(run())


def test_a_reconnect_is_one_more_round():
    async def run():
        *_, buyer, seller = parties()
        await one_round(buyer, seller, contact(1), contact(2))
        offer, answer = await one_round(buyer, seller, contact(3), contact(4))
        assert offer["id"] == contact(3)["id"] and answer["id"] == contact(4)["id"]

    asyncio.run(run())


def test_nothing_on_the_ledger_is_readable():
    async def run():
        chain, *_, buyer, seller = parties()
        await one_round(buyer, seller, contact(1), contact(2))
        for _, _, data in chain.posted:
            assert b"203.0.113" not in data and bytes.fromhex(contact(1)["id"]) not in data
            assert len(data) == 2 + bch.RDV_BLOB_LEN

    asyncio.run(run())


def test_a_stranger_cannot_stand_in_for_the_buyer():
    async def run():
        chain, channel, _, _, secret, _, seller = parties()
        stranger = ChannelRendezvous.buyer(chain, channel, Keypair(), Keypair().pubkey(),
                                           bch.net_keypair(secret).public_bytes, poll=0.01)
        await stranger.publish(contact(9))
        with pytest.raises(TimeoutError):
            await seller.wait(0.1)

    asyncio.run(run())


def test_garbage_and_expired_offers_are_skipped():
    async def run():
        chain, channel, buyer_kp, _, _, buyer, seller = parties()
        chain.post_memo(buyer_kp, channel, b"\0\1not a blob")
        buyer.ttl = -1
        await buyer.publish(contact(1))
        with pytest.raises(TimeoutError):
            await seller.wait(0.1)
        buyer.ttl = 600
        await buyer.publish(contact(2))
        assert (await seller.wait(2))["id"] == contact(2)["id"]

    asyncio.run(run())


def test_a_seller_does_not_answer_its_own_memo():
    async def run():
        *_, buyer, seller = parties()
        await buyer.publish(contact(1))
        await seller.wait(2)
        await seller.publish(contact(2))
        with pytest.raises(TimeoutError):
            await seller.wait(0.1)

    asyncio.run(run())


def test_public_endpoint_round_trips():
    data = bch.public_endpoint(b"\7" * 32, "203.0.113.5:41710").ljust(64, b"\0")
    assert data[32:51] and bch.unpack_addr(data[32:51]) == "203.0.113.5:41710"
