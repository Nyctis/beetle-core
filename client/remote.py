"""One paid inference session with a seller.

The buyer opens a payment channel with a one-time session key, finds the seller
through the rendezvous, opens the sealed channel and pays as it goes. Before a
request it signs a voucher for what was used, the request's input and one step;
during the answer it signs another step each time it nears the limit. So the
seller is never owed more than a step, and never holds more than a step it
hasn't earned. Closing sends a voucher for exactly what was used.

What was used is the smaller of two counts, this buyer's and the seller's receipt,
and a receipt far above this buyer's count stops the session.

The floor comes before everything private. A session shows no evidence, whatever
the offer claims, so it counts as T3-C, and serve/client.py accepts nothing better.
An adapter or a hidden question leaves only after that passed.
"""

from __future__ import annotations

import asyncio
import time

from solders.keypair import Keypair
from solders.pubkey import Pubkey

import model_profiles as mp
from beetle_chain import Offer, Seller, session_salt, sign_voucher
from client.errors import BeetleError, budget_exceeded, no_route
from client.local import vllm_files
from client.policy import accepts
from client.proof import card
from client.tokens import TokenCounter
from client.verify import PARAMS, compare, questions, require_effect
from net.peer import Peer
from node import wire
from serve import UNPROVEN_TIERS, counts_agree
from serve.client import GatewayError, ReceiptError, Session, SessionLost

GRACE_S = 6 * 3600  # how long a seller has to settle after this buyer asks to close
STEP_SHARE = 50     # a step is this fraction of the deposit


def seller_of(chain, offer: Offer) -> Seller:
    return Seller.decode(offer.seller, chain._account(offer.seller))


class RemoteSession:
    def __init__(self, core, offer: Offer, profile: mp.Profile):
        self.core, self.offer, self.profile = core, offer, profile
        self.session: Session | None = None
        self.peer: Peer | None = None
        self.key = Keypair()
        self.covered = self.used = 0
        self.usage = {"in_tokens": 0, "out_tokens": 0}
        self.last_cost = 0
        self.broken = False
        self.adapter: str | None = None  # the name it was loaded under
        self.effect: dict | None = None
        self.proof: dict | None = None
        self.channel: Pubkey | None = None
        self.counter = TokenCounter()
        self.lock = asyncio.Lock()
        self.last_used = 0.0

    @property
    def tier(self) -> str:
        return self.session.tier

    def pipe_alive(self) -> bool:
        return self.session is not None and not self.session.closing and not self.broken

    async def open(self, deposit: int, floor: str) -> None:
        chain, wallet, offer = self.core.chain, self.core.wallet, self.offer
        self.deposit, self.step = deposit, max(deposit // STEP_SHARE, 1)
        self.counter = await asyncio.to_thread(TokenCounter, self.profile)
        seller = await asyncio.to_thread(seller_of, chain, offer)
        payee = seller.authority
        self.terms = {"offer": str(offer.address), "rev": offer.rev, "step": self.step}
        self.channel = await asyncio.to_thread(
            chain.open_channel, wallet, payee, self.key.pubkey(), deposit,
            self.core.cfg.get("grace_s", GRACE_S), session_salt(self.terms))
        self.core.ledger.entries[self.ledger_key]["channel"] = str(self.channel)
        self.core.ledger.update(self.ledger_key, "escrowed", deposit / 1e6)
        self.peer = await wire.bound_peer(self.core.cfg)
        nodes = await asyncio.to_thread(self.core.echo_nodes)
        addrs = await wire.candidates(self.peer, self.core.cfg, nodes)
        rendezvous = wire.buyer_session(self.core.cfg, self.channel, chain, wallet, payee, seller.net_pk)
        self.session = await Session.open(rendezvous, peer=self.peer, addrs=addrs,
                                          mode=self.core.cfg.get("traffic", "constant"))
        if not accepts(floor, self.session.tier):  # still nothing private has left this machine
            raise no_route(f"the session is {self.session.tier}, below the floor {floor}", floor=floor)
        self.proof = card(self.session.tier, peer={"seller": str(payee), "offer": str(offer.address),
                                                   "transport": self.session.contact["id"],
                                                   "environment": self.session.contact["enc_pk"]},
                          evidence={"type": "none", "verified": False,
                                    "note": UNPROVEN_TIERS[self.session.tier]},
                          path=wire.path_of(self.session.pipe.link))
        self.proof["receipt"] = {"gpu": {"name": wire.gpu_name(offer.hw_class), "memory_gb": offer.mem_gb},
                                 "model": self.profile.name, "model_revision": self.profile.revision,
                                 "model_hash": self.profile.manifest_sha256, "channel": str(self.channel),
                                 "settlement": "payment channel, settled when the session ends"}

    def _pay(self, cumulative: int, final: bool = False) -> None:
        _, sig = sign_voucher(self.key, self.channel, cumulative)
        self.session.pipe.send({"t": "voucher", "channel": str(self.channel), "terms": self.terms,
                                "cumulative": cumulative, "sig": sig.hex(), "final": final}, tiered=True)
        self.covered = cumulative
        self.core.ledger.note(self.ledger_key, signed=cumulative / 1e6)  # the most a crash can leave behind

    def _bill(self, mine_in: int, mine_out: int, before: dict) -> None:
        receipt = self.session.receipt
        claimed_in = receipt["in_tokens"] - before["in_tokens"]
        claimed_out = receipt["out_tokens"] - before["out_tokens"]
        if not counts_agree(claimed_in, mine_in) or not counts_agree(claimed_out, mine_out):
            self.broken = True
            raise BeetleError(502, "usage_mismatch", f"the seller claims {claimed_in}+{claimed_out} tokens, "
                              f"this machine counted {mine_in}+{mine_out}; session stopped")
        billed_in, billed_out = min(claimed_in, mine_in), min(claimed_out, mine_out)
        self.usage = {"in_tokens": billed_in, "out_tokens": billed_out}
        self.last_cost = self.offer.usage_cost(billed_in, billed_out)
        self.used += self.last_cost
        self.core.ledger.entries[self.ledger_key]["used"] = self.used / 1e6

    async def chat(self, messages: list[dict], params: dict, name: str | None = None):
        async with self.lock:
            mine_in = self.counter.prompt(messages)
            target = min(self.used + self.offer.usage_cost(mine_in, 0) + self.step, self.deposit)
            if target <= self.used:
                raise budget_exceeded("this session's deposit is used up")
            if target > self.covered:
                self._pay(target)
            before = self.session.receipt or {"in_tokens": 0, "out_tokens": 0}
            pieces: list[str] = []
            running = 0
            try:
                async for piece in self.session.chat(messages, model=name or self.served(), **params):
                    pieces.append(piece)
                    running += self.counter.count(piece)
                    spent = self.used + self.offer.usage_cost(mine_in, running)
                    if spent + self.step // 2 >= self.covered and self.covered < self.deposit:
                        self._pay(min(spent + self.step, self.deposit))
                    yield piece
            except (GatewayError, ReceiptError, SessionLost):
                self.broken = True
                raise
            self._bill(mine_in, self.counter.count("".join(pieces)), before)

    def served(self) -> str:
        names = self.session.info["models"]
        return next((n for n in names if n in (self.profile.name, self.profile.repo)), names[0])

    async def load_adapter(self, files: dict, hidden: list[dict] | None) -> None:
        """Hand the adapter to the seller's environment, then prove it took."""
        if self.covered == 0:
            self._pay(min(self.step, self.deposit))
        name = await self.session.upload_adapter_files(vllm_files(files, self.profile))
        items = questions(hidden)
        ask = [[{"role": "user", "content": it["question"]}] for it in items]
        base = [await self._text(m, None) for m in ask]
        tuned = [await self._text(m, name) for m in ask]
        self.effect = require_effect(compare(items, base, tuned), "the seller's server")
        self.adapter = name

    async def _text(self, messages, name) -> str:
        return "".join([p async for p in self.chat(messages, dict(PARAMS), name)])

    async def abandon(self) -> None:
        if self.peer is not None:
            await self.peer.close()
        if self.channel is None:
            self.core.ledger.release(self.ledger_key)

    async def close(self) -> None:
        if self.session is None:
            return
        try:
            async with self.lock:
                self._pay(self.used, final=True)
                await self.session.close()
        finally:  # the money counts as settled once the chain says so: see sweep_channels
            self.core.ledger.entries[self.ledger_key]["closed_at"] = time.time()
            await self.peer.close()
