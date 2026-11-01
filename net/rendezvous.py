"""How two peers exchange {id, addrs} before they dial.

Each side publishes its own contact and waits for the other's. A reconnect is
simply one more publish and wait, so wait() must return each publication once.

Nothing secret passes through here. For a job, a rendezvous that lies can only
send a peer to the wrong machine, and that machine cannot open what is sealed to
the chain's enc_pk. An inference session may also take its enc_pk from here: on
chain that is a memo the seller signed, and the tier is T3-C anyway. A contact
reveals addresses to anyone who can read it: MemoryRendezvous and FileRendezvous
don't encrypt them. ChainRendezvous and ChannelRendezvous do, as the chain is public.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

log = logging.getLogger(__name__)

TX_TRIES = 3    # a blockhash lives about 32 s; send() gives up after 30 s
MEMO_SEARCH = 1000  # transactions per look; what an RPC node gives in one page
BAD_BLOB = 1    # rdv_fail reason: the other side's blob wouldn't open


class Rendezvous(ABC):
    @abstractmethod
    async def publish(self, contact: dict) -> None:
        """Make {id, addrs} visible to the other side."""

    @abstractmethod
    async def wait(self, timeout: float | None = None) -> dict:
        """The other side's next contact, or TimeoutError."""


class MemoryRendezvous(Rendezvous):
    """One end of an in-process pair."""

    def __init__(self, inbox: asyncio.Queue, outbox: asyncio.Queue):
        self.inbox, self.outbox = inbox, outbox

    @classmethod
    def pair(cls) -> tuple["MemoryRendezvous", "MemoryRendezvous"]:
        a, b = asyncio.Queue(), asyncio.Queue()
        return cls(a, b), cls(b, a)

    async def publish(self, contact: dict) -> None:
        self.outbox.put_nowait(contact)

    async def wait(self, timeout: float | None = None) -> dict:
        return await asyncio.wait_for(self.inbox.get(), timeout)


class FileRendezvous(Rendezvous):
    """Two files, each written by one side. Any way of sharing them will do."""

    def __init__(self, mine: Path, theirs: Path, poll: float = 0.2):
        self.mine, self.theirs, self.poll = Path(mine), Path(theirs), poll
        self._seen = None

    def _write(self, blob: bytes) -> None:
        self.mine.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.mine.with_suffix(".tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, self.mine)

    def _read(self) -> bytes | None:
        try:
            return self.theirs.read_bytes()
        except FileNotFoundError:
            return None

    async def publish(self, contact: dict) -> None:
        # The nonce makes a republished, otherwise identical contact a new event.
        blob = json.dumps({**contact, "nonce": os.urandom(8).hex()}).encode()
        await asyncio.to_thread(self._write, blob)

    async def wait(self, timeout: float | None = None) -> dict:
        async def next_contact():
            while True:
                blob = await asyncio.to_thread(self._read)
                if blob and blob != self._seen:
                    try:
                        contact = json.loads(blob)
                    except ValueError:  # a copy tool may still be writing it
                        contact = None
                    if contact:
                        self._seen = blob
                        return contact
                await asyncio.sleep(self.poll)

        return await asyncio.wait_for(next_contact(), timeout)


class RendezvousFailed(Exception):
    """The other side gave up, or what it wrote can't be opened."""


class ChannelRendezvous(Rendezvous):
    """The same exchange where there is no Rendezvous account to use: an inference
    session, which has a payment channel and no job, and a job whose account has
    closed. The sealed blobs of ChainRendezvous travel as memos on transactions
    that mention the address, and only memos signed by the two parties are read.

    The buyer publishes and then waits, once per connection attempt. The seller
    loops on wait and publish, as Gateway.serve does. A seller's contact can carry
    an `enc_pk`, in the blob's reply_pk. Memos are permanent and can't be closed,
    unlike the account: only sealed bytes go there. Anyone can mention an address,
    so the search looks further back than one screenful of noise.
    """

    def __init__(self, chain, channel, signer, other, role: int, *, seal_to: bytes | None = None,
                 net_key=None, ttl: int = 600, poll: float = 0.5):
        import beetle_chain  # needs solders, which the file rendezvous doesn't

        self.bch, self.chain, self.channel, self.signer = beetle_chain, chain, channel, signer
        self.other, self.role, self.seal_to, self.net_key = other, role, seal_to, net_key
        self.ttl, self.poll = ttl, poll
        self._round = 0
        self._seen: set[str] = set()
        self._reply = None     # buyer: the one-time key the answer is sealed to

    @classmethod
    def buyer(cls, chain, channel, payer, payee, net_pk: bytes, **kw) -> "ChannelRendezvous":
        return cls(chain, channel, payer, payee, 0, seal_to=net_pk, **kw)

    @classmethod
    def seller(cls, chain, channel, payee, payer, net_secret: bytes, **kw) -> "ChannelRendezvous":
        import beetle_chain

        return cls(chain, channel, payee, payer, 1, net_key=beetle_chain.net_keypair(net_secret), **kw)

    @property
    def _buying(self) -> bool:
        return self.role == self.bch.ROLE_BUYER

    async def publish(self, contact: dict) -> None:
        import beetle_crypto as bc

        reply_pk = bytes.fromhex(contact["enc_pk"]) if contact.get("enc_pk") else bytes(32)
        if self._buying:
            self._round += 1
            self._reply = bc.HpkeKeyPair()
            reply_pk = self._reply.public_bytes
        plain = self.bch.rdv_plaintext(self.role, self._round, bytes.fromhex(contact["id"]),
                                       int(time.time()) + self.ttl, contact["addrs"], reply_pk)
        blob = self.chain.rdv_seal(self.seal_to, bytes(self.channel), self.role, self._round, plain)
        await asyncio.to_thread(self.chain.post_memo, self.signer, self.channel,
                                bytes([self.role, self._round]) + blob)

    def _open(self, memo: bytes) -> dict | None:
        """A memo from the other side for the round in play, opened, or None."""
        sender, round_, blob = memo[0], memo[1], memo[2:]
        if sender == self.role or len(blob) != self.bch.RDV_BLOB_LEN \
                or (self._buying and round_ != self._round) or (not self._buying and round_ <= self._round):
            return None
        key = self._reply if self._buying else self.net_key
        try:
            fields = self.chain.rdv_unseal(key, bytes(self.channel), sender, round_, blob)
        except Exception:  # a party can post nonsense; it costs it a fee and nothing else
            return None
        return fields if fields["not_after"] >= time.time() else None

    async def wait(self, timeout: float | None = None) -> dict:
        async def next_contact() -> dict:
            while True:
                memos = await asyncio.to_thread(self.chain.memos, self.channel, self.other, self._seen,
                                                MEMO_SEARCH)
                for memo in memos:
                    fields = self._open(memo)
                    if fields:
                        if self._buying:
                            return {**fields, "enc_pk": fields["reply_pk"].hex()}
                        self._round, self.seal_to = memo[1], fields["reply_pk"]  # the answer goes to it
                        return fields
                await asyncio.sleep(self.poll)

        return await asyncio.wait_for(next_contact(), timeout)


class ChainRendezvous(Rendezvous):
    """One side of a job's Rendezvous account (net.md 5.5 to 5.11).

    The buyer calls publish and then wait, once per connection attempt: publish
    writes an offer sealed to the seller's net_pk, wait returns the answer. The
    seller loops on wait and then publish, or publishes first: wait returns the
    next offer, and the answer goes out in the background, so the seller can
    start dialing before it lands. Contacts are {id, addrs}; the id is a 64-char
    hex iroh id, and at most five "ip:port" addresses fit. Only the two sides of
    the job can write here, so a blob's sender is known from the chain. The
    default poll keeps within a public RPC's limit of 40 calls of one method per
    10 s; lower it on your own node.
    """

    def __init__(self, chain, job_id: bytes, signer, role: int, *, net_key=None, payer=None,
                 policy: int = 0, ttl: int = 600, poll: float = 0.4):
        import beetle_chain  # needs solders, which the other rendezvous kinds don't

        self.bch, self.chain, self.job_id, self.signer = beetle_chain, chain, job_id, signer
        self.role, self.net_key, self.payer = role, net_key, payer
        self.policy, self.ttl, self.poll = policy, ttl, poll
        self._reply = None     # buyer: the one-time key the current answer is sealed to
        self._round = 0
        self._seen = None      # the last publication wait() returned
        self._mine = None      # seller: a contact waiting for an offer
        self._offer = None     # seller: an offer waiting for a contact
        self._sending = None

    @classmethod
    def buyer(cls, chain, job_id: bytes, requester, **kw) -> "ChainRendezvous":
        return cls(chain, job_id, requester, 0, **kw)

    @classmethod
    def seller(cls, chain, job_id: bytes, operator, net_secret: bytes, **kw) -> "ChainRendezvous":
        """`net_secret` is the 32 bytes behind the net_pk this seller put on chain."""
        import beetle_chain

        return cls(chain, job_id, operator, 1, net_key=beetle_chain.net_keypair(net_secret), **kw)

    @property
    def _buying(self) -> bool:
        return self.role == self.bch.ROLE_BUYER

    async def _read(self):
        return await asyncio.to_thread(self.chain.get_rendezvous, self.job_id)

    async def _send(self, landed, fn, *args) -> None:
        """Retries with a fresh blockhash; a copy that landed late still counts."""
        for attempt in range(TX_TRIES):
            try:
                await asyncio.to_thread(fn, *args)
                return
            except TimeoutError:
                rdv = await self._read()
                if rdv is not None and landed(rdv):
                    return
                if attempt == TX_TRIES - 1:
                    raise

    def _plain(self, contact: dict, round_: int, reply_pk: bytes = bytes(32)) -> bytes:
        return self.bch.rdv_plaintext(
            self.role, round_, bytes.fromhex(contact["id"]), int(time.time()) + self.ttl,
            contact["addrs"], reply_pk, contact.get("nat_hint", 0), contact.get("flags", 0))

    async def publish(self, contact: dict) -> None:
        if self._buying:
            await self._publish_offer(contact)
        elif self._offer is None:
            self._mine = contact
        else:
            self._answer(self._offer, contact)
            self._offer = None

    async def _publish_offer(self, contact: dict) -> None:
        import beetle_crypto as bc

        rdv = await self._read()
        job = None if rdv else await asyncio.to_thread(self.chain.get_job, self.job_id)
        if rdv is None and job is None:
            raise RendezvousFailed("no such job on chain")
        seller = await asyncio.to_thread(self.chain.get_seller, rdv.host if rdv else job.host)
        if seller.net_pk == bytes(32):
            raise RendezvousFailed("this seller has no net_pk on chain")
        # A new connection starts over; a retry of this one takes the next round.
        fresh = rdv is None or rdv.state == "Failed" or rdv.round >= self.bch.RDV_MAX_ROUNDS
        round_ = 1 if fresh else rdv.round + 1
        reply = bc.HpkeKeyPair()
        blob = self.chain.rdv_seal(seller.net_pk, self.job_id, self.role, round_,
                                   self._plain(contact, round_, reply.public_bytes))
        landed = lambda r: r.offer == blob
        if fresh:
            await self._send(landed, self.chain.rdv_open, self.signer, self.job_id, blob, self.ttl,
                             self.policy, self.payer)
        else:
            await self._send(landed, self.chain.rdv_reoffer, self.signer, rdv, round_, blob)
        self._reply, self._round = reply, round_

    def _answer(self, offer: tuple, contact: dict) -> None:
        rdv, fields = offer

        async def send():
            try:
                blob = self.chain.rdv_seal(fields["reply_pk"], self.job_id, self.role, rdv.round,
                                           self._plain(contact, rdv.round))
                await self._send(lambda r: r.answer == blob, self.chain.rdv_answer, self.signer, rdv,
                                 rdv.round, blob)
            except Exception as e:  # the buyer times out and offers again
                log.warning("rendezvous answer failed: %s", e)

        self._sending = asyncio.create_task(send())

    async def wait(self, timeout: float | None = None) -> dict:
        return await asyncio.wait_for(
            self._next_answer() if self._buying else self._next_offer(), timeout)

    async def _next_answer(self) -> dict:
        while True:
            rdv = await self._read()
            if rdv is not None and rdv.round == self._round:
                if rdv.state == "Failed":
                    raise RendezvousFailed("the seller gave up on this rendezvous")
                if rdv.state == "Answered" and self._seen != (rdv.opened_at, rdv.round):
                    self._seen = (rdv.opened_at, rdv.round)
                    return await self._open(rdv, rdv.answer, self.bch.ROLE_SELLER, self._reply)
            await asyncio.sleep(self.poll)

    async def _next_offer(self) -> dict:
        if self._sending is not None:
            await self._sending
            self._sending = None
        while True:
            rdv = await self._read()
            if rdv is not None and rdv.state == "Offered" and self._seen != (rdv.opened_at, rdv.round):
                self._seen = (rdv.opened_at, rdv.round)
                fields = await self._open(rdv, rdv.offer, self.bch.ROLE_BUYER, self.net_key)
                if fields["not_after"] >= time.time():
                    if self._mine is not None:
                        self._answer((rdv, fields), self._mine)
                        self._mine = None
                    else:
                        self._offer = (rdv, fields)
                    return fields
            await asyncio.sleep(self.poll)

    async def _open(self, rdv, blob: bytes, sender: int, key) -> dict:
        try:
            return self.chain.rdv_unseal(key, self.job_id, sender, rdv.round, blob)
        except Exception:
            await asyncio.to_thread(self.chain.rdv_fail, self.signer, rdv, BAD_BLOB)
            raise RendezvousFailed("the other side's blob does not open") from None

    async def fail(self, reason: int = 0) -> None:
        """Give up on this connection. Nothing here moves money: see abort_unstarted."""
        rdv = await self._read()
        if rdv is not None and rdv.state != "Failed":
            await asyncio.to_thread(self.chain.rdv_fail, self.signer, rdv, reason)

    async def close(self) -> None:
        """Buyer, once connected or done: returns the deposit. A reconnect reopens."""
        rdv = await self._read()
        if rdv is not None:
            await asyncio.to_thread(self.chain.rdv_close, self.signer, rdv)
