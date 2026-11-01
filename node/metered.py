"""Inference that gets paid: the gateway from serve/, plus a payment channel.

The gateway's sessions know nothing about money. This adds one message on the
sealed channel, a voucher: the buyer's one-time session key saying the seller
may take this much in total from the channel. Two halves, in two processes,
because the one that reads prompts must not hold the wallet:

    MeteredEnvironment  node/environment.py. Plaintext, no wallet. Serves only
                        while what was used stays below the latest voucher, and
                        hands every voucher to the daemon.
    Settler             node/daemon.py. The wallet, no plaintext. Puts vouchers
                        on chain before they can go stale.

Neither side should have to trust the other for more than a step:

- A response is metered as it runs. Past the voucher it pauses for the next one,
  and stops if none comes, so a large `max_tokens` buys nothing unpaid.
- The channel's grace period is how long the seller has to settle once a buyer
  asks to close. A channel that leaves too little is refused, because after it
  runs out unsettled vouchers are worth nothing.
- Vouchers go on chain during a session and as soon as a channel starts closing.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

from solders.pubkey import Pubkey

import node  # noqa: F401  (import paths)

import beetle_crypto as bc
from beetle_chain import KIND_INFERENCE, Channel, Offer, session_salt, voucher_message
from net import channel
from serve import TOKEN_SLACK, TOKEN_TOLERANCE, estimate_tokens
from serve.gateway import Environment, Session

MIN_GRACE_S = 3600
VOUCHER_WAIT = 10.0   # how long a paused response waits for the next voucher
SETTLE_SHARE = 4      # settle once this fraction of the deposit is unsettled
SETTLE_EVERY = 300.0  # and at least this often


class PaymentRefused(Exception):
    pass


@dataclass
class Payment:
    channel: Channel
    offer: Offer
    covered: int = 0

    def take(self, cumulative: int, sig: bytes, used: int, final: bool) -> None:
        """A voucher only moves up. The closing one names what the buyer counted,
        which may sit below what this side counted, within an honest difference."""
        if not bc.verify(bytes(self.channel.authorized_signer), sig,
                         voucher_message(self.channel.address, cumulative)):
            raise PaymentRefused("bad signature")
        if cumulative > self.channel.deposit:
            raise PaymentRefused("more than the channel holds")
        slack = self.offer.usage_cost(TOKEN_SLACK, TOKEN_SLACK)
        floor = (used - slack) / TOKEN_TOLERANCE if final else self.covered
        if cumulative < floor:
            raise PaymentRefused("voucher went backwards")
        self.covered = cumulative


class MeteredSession(Session):
    def __init__(self, env: "MeteredEnvironment", ch, hello: dict):
        super().__init__(env, ch, hello)
        self.payment: Payment | None = None
        self.running_in = self.running_out = 0  # the response being generated
        self.credit = asyncio.Event()
        self.waiting: asyncio.Queue = asyncio.Queue()
        self._from_pipe = self.pipe.recv
        self.pipe.recv = self._next_request

    def used(self) -> int:
        if self.payment is None:
            return 0
        return self.payment.offer.usage_cost(self.in_tokens + self.running_in,
                                             self.out_tokens + self.running_out)

    async def _next_request(self) -> dict:
        item = await self.waiting.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def _pump(self) -> None:
        """Vouchers are read the moment they land: a response in flight waits for them."""
        try:
            while True:
                msg = await self._from_pipe()
                if msg.get("t") != "voucher":
                    self.waiting.put_nowait(msg)
                    continue
                try:
                    await self._voucher(msg)
                except PaymentRefused as e:  # the buyer sees it on its next call
                    self._refuse("payment_refused", str(e))
        except Exception as e:
            self.waiting.put_nowait(e)

    async def _voucher(self, msg: dict) -> None:
        if self.payment is None:
            self.payment = await asyncio.to_thread(self.env.open_payment, msg)
        self.payment.take(msg["cumulative"], bytes.fromhex(msg["sig"]), self.used(), msg.get("final", False))
        self.env.report({"t": "voucher", "channel": msg["channel"], "cumulative": msg["cumulative"],
                         "sig": msg["sig"], "used": self.used()})
        self.credit.set()

    async def _chat(self, msg: dict) -> None:
        self.running_in = estimate_tokens(bc.canonical_json(msg))
        if self.payment is None or self.used() >= self.payment.covered:
            self.running_in = 0
            self._refuse("payment_required", "send a voucher that covers this request")
            return
        try:
            await super()._chat(msg)
        finally:
            self.running_in = self.running_out = 0

    async def keep_going(self, pieces: int) -> bool:
        self.running_out = pieces
        while self.used() >= self.payment.covered:
            self.credit.clear()
            try:
                await asyncio.wait_for(self.credit.wait(), VOUCHER_WAIT)
            except TimeoutError:
                return False
        return True

    async def _adapter(self, msg: dict) -> None:
        if self.payment is None or self.payment.covered == 0:
            self._refuse("payment_required", "an adapter takes a paid session")
            return
        await super()._adapter(msg)

    async def run_metered(self) -> None:
        pump = asyncio.create_task(self._pump())
        try:
            await self.run()
        finally:
            pump.cancel()
            if self.payment:
                self.env.report({"t": "ended", "channel": str(self.payment.channel.address)})


class MeteredEnvironment(Environment):
    def __init__(self, *args, chain, payee: Pubkey, report, min_grace: int = MIN_GRACE_S, **kwargs):
        super().__init__(*args, **kwargs)
        self.chain, self.payee, self.report, self.min_grace = chain, payee, report, min_grace
        self.paying: set[Pubkey] = set()

    async def attach(self, link, first: bytes) -> None:
        kind, _, enc = channel.peek(first)
        if kind != channel.OPEN:
            await super().attach(link, first)
            return
        ch, hello = channel.accept(self.keypair, first, self.seen)
        session = MeteredSession(self, ch, json.loads(hello))
        self.sessions[enc] = session
        session.task = asyncio.create_task(session.run_metered())
        session.pipe.attach(link)

    def open_payment(self, msg: dict) -> Payment:
        """The first voucher names a channel and the terms its address commits to."""
        chain, address = self.chain, Pubkey.from_string(msg["channel"])
        ch = chain.get_channel(address)
        if ch is None or ch.status != "Open":
            raise PaymentRefused("no such open channel")
        if ch.payee != self.payee or ch.mint != chain.mint:
            raise PaymentRefused("this channel does not pay this seller")
        if ch.grace_period < self.min_grace:
            raise PaymentRefused(f"grace period under {self.min_grace} s: vouchers could go stale")
        if session_salt(msg["terms"]) != ch.salt:
            raise PaymentRefused("terms do not match the channel")
        offer = chain.get_offer(Pubkey.from_string(msg["terms"]["offer"]))
        if offer is None or offer.kind != KIND_INFERENCE or offer.seller != chain.seller_pda(self.payee):
            raise PaymentRefused("not an offer of this seller")
        if address in self.paying:
            raise PaymentRefused("channel already in use")
        self.paying.add(address)
        return Payment(ch, offer)


class Settler:
    """The vouchers of every channel, kept on disk so a restart loses no money.

    A voucher runs a step ahead of what was used, and what is on chain can't come
    back down. So during a session only a voucher already earned goes on chain.
    The latest one goes when the session ends, or when the channel starts closing:
    by then it's that or nothing, and a buyer gone quiet risks at most that step.
    """

    def __init__(self, chain, wallet, root: Path, log):
        self.chain, self.wallet, self.root, self.log = chain, wallet, Path(root), log
        self.root.mkdir(parents=True, exist_ok=True)
        self.settled_at: dict[str, float] = {}

    def _path(self, channel_id: str) -> Path:
        return self.root / f"{channel_id}.json"

    def _held(self, channel_id: str) -> dict:
        path = self._path(channel_id)
        return json.loads(path.read_text()) if path.exists() else {}

    def record(self, msg: dict) -> None:
        held = self._held(msg["channel"])
        latest = {"cumulative": msg["cumulative"], "sig": msg["sig"]}
        earned = held.get("earned")
        if held.get("latest") and held["latest"]["cumulative"] <= msg.get("used", 0):
            earned = held["latest"]
        path = self._path(msg["channel"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"latest": latest, "earned": earned}))
        tmp.replace(path)

    def _voucher(self, ch: Channel, which: str = "latest") -> tuple[int, tuple[bytes, bytes]] | None:
        """That voucher, if it would move the watermark. The closing one can be lower
        than one already on chain, and that one then stands."""
        held = self._held(str(ch.address)).get(which)
        if not held or held["cumulative"] <= ch.settled:
            return None
        return held["cumulative"], (voucher_message(ch.address, held["cumulative"]), bytes.fromhex(held["sig"]))

    def watch(self, channels: list[Channel]) -> None:
        """Every tick. A channel starting to close can't wait for the session to end."""
        for ch in channels:
            name = str(ch.address)
            try:
                if ch.status in ("Closing", "Sealed"):
                    self.close(ch)
                elif ch.status == "Open" and (earned := self._voucher(ch, "earned")):
                    due = earned[0] - ch.settled >= max(ch.deposit // SETTLE_SHARE, 1)
                    if due or time.time() - self.settled_at.get(name, time.time()) > SETTLE_EVERY:
                        self.chain.settle_channel(self.wallet, ch, earned[1])
                        self.settled_at[name] = time.time()
                    self.settled_at.setdefault(name, time.time())
            except RuntimeError as e:
                self.log(f"channel {name[:12]}: not settled yet: {str(e)[:100]}")

    def close(self, ch: Channel) -> None:
        name = str(ch.address)
        if ch.status in ("Open", "Closing"):
            held = self._voucher(ch)
            self.chain.settle_and_seal(self.wallet, ch, held[1] if held else None)
            ch = self.chain.get_channel(ch.address)
        settled = ch.settled if ch else 0
        if ch is not None and ch.status == "Sealed":
            self.chain.distribute(self.wallet, ch)
        self.log(f"channel {name[:12]} settled at {settled / 1e6:.6f} USDC")
        self._path(name).unlink(missing_ok=True)
        self.settled_at.pop(name, None)

    def finish(self, channel_id: str) -> None:
        """A session ended, for any reason."""
        try:
            ch = self.chain.get_channel(Pubkey.from_string(channel_id))
            if ch is not None:
                self.close(ch)
        except RuntimeError as e:
            self.log(f"channel {channel_id[:12]} not settled: {str(e)[:100]}")
