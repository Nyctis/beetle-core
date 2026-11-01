"""Inference money, over a real loopback link against a fake server: a response is
bound by vouchers, a channel can't strand them, and the buyer pays what it counted."""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import client  # noqa: E402,F401

import beetle_crypto as bc  # noqa: E402
import node.metered as metered  # noqa: E402
import serve.gateway as gateway  # noqa: E402
from beetle_chain import Channel, session_salt, sign_voucher, voucher_message  # noqa: E402
from client.errors import BeetleError  # noqa: E402
from client.policy import Ledger  # noqa: E402
from client.remote import RemoteSession  # noqa: E402
from client.tokens import TokenCounter  # noqa: E402
from net.peer import Peer  # noqa: E402
from net.rendezvous import MemoryRendezvous  # noqa: E402
from node.metered import MeteredEnvironment, Payment, PaymentRefused, Settler  # noqa: E402
from serve import MAX_RESPONSE_TOKENS, counts_agree  # noqa: E402
from serve.client import GatewayError, Session  # noqa: E402
from serve.gateway import Backend, Gateway, Spool  # noqa: E402
from solders.keypair import Keypair  # noqa: E402
from test_gateway import FakeBackend  # noqa: E402

KEY_FILES = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"x"}


def channel_of(key: Keypair, mint, payee, *, status="Open", deposit=1000, settled=0, grace=3600, terms=None):
    return Channel(Keypair().pubkey(), status, session_salt(terms or {}), deposit, settled, 0, 0, 0, grace,
                   Keypair().pubkey(), payee, key.pubkey(), mint, Keypair().pubkey(), 0)


class Book:
    """The chain as a seller's environment and settler see it."""

    def __init__(self, ch: Channel, seller):
        self.ch, self.seller, self.mint = ch, seller, ch.mint
        self.calls: list[tuple] = []

    def get_channel(self, address):
        return self.ch

    def get_offer(self, address):
        return SimpleNamespace(kind=1, seller=self.seller)

    def seller_pda(self, authority):
        return self.seller

    def settle_channel(self, caller, ch, voucher):
        self.calls.append(("settle", voucher))
        self.ch.settled = int.from_bytes(voucher[0][34:42], "little")

    def settle_and_seal(self, payee, ch, voucher=None):
        self.calls.append(("seal", voucher))
        self.ch.status = "Sealed"

    def distribute(self, caller, ch):
        self.calls.append(("distribute", None))


class PerOutputToken:
    """An offer that charges 1 for each output token, so the numbers stay readable."""

    @staticmethod
    def usage_cost(in_tokens: int, out_tokens: int) -> int:
        return out_tokens


class Rig:
    def __init__(self, env, ch, key, session, backend, reports, terms):
        self.env, self.ch, self.key, self.session, self.backend = env, ch, key, session, backend
        self.reports, self.terms = reports, terms

    def voucher(self, cumulative: int, final: bool = False) -> None:
        _, sig = sign_voucher(self.key, self.ch.address, cumulative)
        self.session.pipe.send({"t": "voucher", "channel": str(self.ch.address), "terms": self.terms,
                                "cumulative": cumulative, "sig": sig.hex(), "final": final}, tiered=True)

    async def ask(self, **params) -> str:
        text = [p async for p in self.session.chat([{"role": "user", "content": "hi"}], model="base", **params)]
        return "".join(text)


@asynccontextmanager
async def rig(tmp_path, *, grace=3600, delay=0.0, backend_cls=FakeBackend, offer=PerOutputToken, **options):
    terms = {"offer": str(Keypair().pubkey()), "rev": 0, "step": 1}
    key, payee = Keypair(), Keypair().pubkey()
    ch = channel_of(key, Keypair().pubkey(), payee, grace=grace, terms=terms)
    book = Book(ch, Keypair().pubkey())
    book.get_offer = lambda address: SimpleNamespace(kind=1, seller=book.seller, usage_cost=offer.usage_cost)
    reports: list[dict] = []
    async with backend_cls(delay) as backend:
        env = MeteredEnvironment(Backend(backend.url), {"base"}, Spool(tmp_path / "spool", 1 << 30),
                                 tmp_path / "scratch", chain=book, payee=payee, report=reports.append, **options)
        seller, peer = await Peer.bind(), await Peer.bind()
        front = Gateway(seller, env, [f"127.0.0.1:{seller.port}"])
        ours, theirs = MemoryRendezvous.pair()
        serving = asyncio.create_task(front.serve(theirs))
        session = await Session.open(ours, peer=peer, addrs=[f"127.0.0.1:{peer.port}"], mode="burst")
        try:
            yield Rig(env, ch, key, session, backend, reports, terms)
        finally:
            serving.cancel()
            await peer.close()
            await seller.close()


def test_nothing_is_served_without_a_voucher(tmp_path):
    async def run():
        async with rig(tmp_path) as r:
            with pytest.raises(GatewayError) as e:
                await r.ask()
            assert e.value.code == "payment_required" and r.backend.requests == []

    asyncio.run(run())


def test_a_buyer_cannot_widen_a_request(tmp_path):
    async def run():
        async with rig(tmp_path) as r:
            r.voucher(100)
            await r.ask(n=50, max_tokens=10**9, chat_template="{{ evil }}", temperature=0.1)
            sent = r.backend.requests[-1]
            assert "n" not in sent and "chat_template" not in sent
            assert sent["max_tokens"] == MAX_RESPONSE_TOKENS and sent["temperature"] == 0.1

    asyncio.run(run())


def test_a_response_stops_where_the_voucher_ends(tmp_path, monkeypatch):
    monkeypatch.setattr(metered, "VOUCHER_WAIT", 0.2)

    async def run():
        async with rig(tmp_path) as r:
            r.voucher(2)
            text = await r.ask(max_tokens=100)
            assert text == "I do "           # the fake answers four words, 1 each
            assert r.session.stopped and r.session.receipt["out_tokens"] == 2

    asyncio.run(run())


def test_a_voucher_during_the_answer_keeps_it_going(tmp_path, monkeypatch):
    monkeypatch.setattr(metered, "VOUCHER_WAIT", 5.0)

    async def run():
        async with rig(tmp_path, delay=0.05) as r:
            r.voucher(2)
            pieces = []
            async for piece in r.session.chat([{"role": "user", "content": "hi"}], model="base"):
                pieces.append(piece)
                if len(pieces) == 2:
                    r.voucher(10)
            assert "".join(pieces) == "I do not know " and not r.session.stopped

    asyncio.run(run())


def test_every_voucher_reaches_the_daemon_and_the_end_is_reported(tmp_path):
    async def run():
        async with rig(tmp_path) as r:
            r.voucher(5)
            await r.ask()
            assert r.reports[0]["cumulative"] == 5 and bc.verify(
                bytes(r.ch.authorized_signer), bytes.fromhex(r.reports[0]["sig"]),
                voucher_message(r.ch.address, 5))
            await r.session.pipe.link.close()
            r.session.closing = True
            session = next(iter(r.env.sessions.values()))
            session.task.cancel()
            await asyncio.sleep(0.1)
            assert r.reports[-1] == {"t": "ended", "channel": str(r.ch.address)}

    asyncio.run(run())


def test_a_channel_with_a_short_grace_is_refused(tmp_path):
    async def run():
        async with rig(tmp_path, grace=30) as r:
            r.voucher(100)
            with pytest.raises(GatewayError) as e:
                await r.ask()
            assert e.value.code == "payment_refused" and "grace" in str(e.value)
            assert r.backend.requests == []

    asyncio.run(run())


def test_an_adapter_upload_takes_a_paid_session(tmp_path):
    async def run():
        async with rig(tmp_path) as r:
            with pytest.raises(GatewayError) as e:
                await asyncio.wait_for(r.session.upload_adapter_files(KEY_FILES), 5)
            assert e.value.code == "payment_required"

    asyncio.run(run())


def test_without_tmpfs_an_adapter_is_refused_rather_than_written(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway, "is_tmpfs", lambda path: False)

    async def run():
        async with rig(tmp_path, require_tmpfs=True) as r:
            r.voucher(100)
            with pytest.raises(GatewayError) as e:
                await asyncio.wait_for(r.session.upload_adapter_files(KEY_FILES), 5)
            assert e.value.code == "adapter_refused"
            assert not list((tmp_path / "scratch").glob("*")) if (tmp_path / "scratch").exists() else True

    asyncio.run(run())


def test_an_adapter_is_exactly_two_named_files(tmp_path):
    async def run():
        async with rig(tmp_path) as r:
            with pytest.raises(ValueError):
                await r.session.upload_adapter_files({"adapter_model.bin": b"pickle"})

    asyncio.run(run())


def test_a_scratch_nobody_can_write_is_named_not_a_permission_error(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway.os, "access", lambda path, mode: False)

    async def go():
        async with rig(tmp_path) as r:
            assert "can't write" in r.env.scratch_problem()
    asyncio.run(go())


def test_tmpfs_is_told_from_a_disk():
    assert not gateway.is_tmpfs(ROOT) or sys.platform == "linux"
    if Path("/proc/mounts").exists() and Path("/dev/shm").is_dir():
        assert gateway.is_tmpfs(Path("/dev/shm"))


def test_each_process_gets_its_own_session_key(tmp_path):
    a = MeteredEnvironment(None, set(), Spool(tmp_path, 1), tmp_path, chain=None, payee=None, report=print)
    b = MeteredEnvironment(None, set(), Spool(tmp_path, 1), tmp_path, chain=None, payee=None, report=print)
    assert a.enc_pk != b.enc_pk
    front = Gateway(SimpleNamespace(id="x"), a, ["203.0.113.1:1"])
    assert front.contact() == {"id": "x", "addrs": ["203.0.113.1:1"], "enc_pk": a.enc_pk.hex()}


# --- vouchers on their own -----------------------------------------------------------


def payment(deposit: int = 1000):
    key = Keypair()
    ch = channel_of(key, Keypair().pubkey(), Keypair().pubkey(), deposit=deposit)
    offer = SimpleNamespace(usage_cost=lambda i, o: i + o)
    return Payment(ch, offer), key


def signed(key, p: Payment, cumulative: int) -> bytes:
    return sign_voucher(key, p.channel.address, cumulative)[1]


def test_vouchers_must_be_signed_covered_and_only_rise():
    p, key = payment()
    with pytest.raises(PaymentRefused):
        p.take(10, signed(Keypair(), p, 10), 0, False)
    with pytest.raises(PaymentRefused):
        p.take(2000, signed(key, p, 2000), 0, False)
    p.take(100, signed(key, p, 100), 0, False)
    with pytest.raises(PaymentRefused):
        p.take(50, signed(key, p, 50), 0, False)


def test_a_closing_voucher_may_name_the_buyers_smaller_count_but_not_much_smaller():
    p, key = payment()
    p.take(500, signed(key, p, 500), 0, False)
    with pytest.raises(PaymentRefused):
        p.take(100, signed(key, p, 100), 400, True)
    p.take(300, signed(key, p, 300), 400, True)
    assert p.covered == 300


# --- settling ------------------------------------------------------------------------


def settler(tmp_path, ch: Channel) -> tuple[Settler, Book]:
    book = Book(ch, Keypair().pubkey())
    return Settler(book, Keypair(), tmp_path / "vouchers", lambda m: None), book


def held(s: Settler, key: Keypair, ch: Channel, cumulative: int, used: int = 0) -> None:
    s.record({"channel": str(ch.address), "cumulative": cumulative, "used": used,
              "sig": sign_voucher(key, ch.address, cumulative)[1].hex()})


def test_only_earned_vouchers_go_on_chain_during_a_session(tmp_path):
    key = Keypair()
    ch = channel_of(key, Keypair().pubkey(), Keypair().pubkey())
    s, book = settler(tmp_path, ch)
    held(s, key, ch, 300)
    s.watch([ch])
    assert book.calls == []                     # prepaid, not earned
    held(s, key, ch, 600, used=300)
    s.watch([ch])
    assert [c[0] for c in book.calls] == ["settle"] and ch.settled == 300
    s.watch([ch])
    assert len(book.calls) == 1                 # nothing new to settle


def test_a_small_earned_voucher_goes_on_chain_in_time(tmp_path, monkeypatch):
    key = Keypair()
    ch = channel_of(key, Keypair().pubkey(), Keypair().pubkey())
    s, book = settler(tmp_path, ch)
    held(s, key, ch, 10)
    held(s, key, ch, 20, used=10)
    s.watch([ch])
    assert book.calls == []
    monkeypatch.setattr(metered, "SETTLE_EVERY", -1.0)
    s.watch([ch])
    assert [c[0] for c in book.calls] == ["settle"]


def test_a_quiet_session_can_still_pay_what_it_counted(tmp_path, monkeypatch):
    """The prepaid step stays off chain, so the closing voucher can land below it."""
    monkeypatch.setattr(metered, "SETTLE_EVERY", -1.0)
    key = Keypair()
    ch = channel_of(key, Keypair().pubkey(), Keypair().pubkey())
    s, book = settler(tmp_path, ch)
    held(s, key, ch, 100, used=0)
    s.watch([ch])
    held(s, key, ch, 3, used=3)
    s.finish(str(ch.address))
    assert book.calls[0][0] == "seal" and book.calls[0][1][0] == voucher_message(ch.address, 3)


def test_a_channel_starting_to_close_is_settled_at_once(tmp_path):
    key = Keypair()
    ch = channel_of(key, Keypair().pubkey(), Keypair().pubkey())
    s, book = settler(tmp_path, ch)
    held(s, key, ch, 100)
    ch.status = "Closing"
    s.watch([ch])
    assert [c[0] for c in book.calls] == ["seal", "distribute"]
    assert book.calls[0][1] is not None


def test_a_voucher_below_the_watermark_is_not_replayed(tmp_path):
    key = Keypair()
    ch = channel_of(key, Keypair().pubkey(), Keypair().pubkey(), settled=500)
    s, book = settler(tmp_path, ch)
    held(s, key, ch, 300)
    s.finish(str(ch.address))
    assert book.calls[0] == ("seal", None)


def test_a_restart_loses_no_voucher(tmp_path):
    key = Keypair()
    ch = channel_of(key, Keypair().pubkey(), Keypair().pubkey())
    s, _ = settler(tmp_path, ch)
    held(s, key, ch, 400)
    again, book = settler(tmp_path, ch)
    ch.status = "Closing"
    again.watch([ch])
    assert book.calls[0][0] == "seal" and book.calls[0][1] is not None


# --- what the buyer pays -------------------------------------------------------------


def test_counts_within_tolerance_agree_and_a_tall_claim_does_not():
    assert counts_agree(100, 100) and counts_agree(150, 80) and counts_agree(0, 0)
    assert not counts_agree(10_000, 100)


class Claims(FakeBackend):
    """A server that says what the rig tells it to about usage."""

    claim = {"prompt_tokens": 7, "completion_tokens": 4}

    async def stream(self, writer, body):
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n")
        for word in "I do not know".split():
            event = {"choices": [{"delta": {"content": word + " "}}]}
            writer.write(f"data: {json.dumps(event)}\n\n".encode())
            await writer.drain()
            await asyncio.sleep(self.delay)
        tail = {"choices": [], "usage": self.claim}
        writer.write(f"data: {json.dumps(tail)}\n\ndata: [DONE]\n\n".encode())


def buyer(r: Rig, tmp_path: Path, deposit: int = 1000, step: int = 10) -> RemoteSession:
    remote = RemoteSession(SimpleNamespace(ledger=Ledger(tmp_path / "ledger.json")),
                           SimpleNamespace(usage_cost=PerOutputToken.usage_cost), None)
    remote.core.ledger.entries["k"] = {}
    remote.ledger_key, remote.session, remote.key = "k", r.session, r.key
    remote.channel, remote.terms = r.ch.address, r.terms
    remote.deposit, remote.step = deposit, step
    remote.served = lambda: "base"
    remote.counter = TokenCounter()
    return remote


async def talk(remote: RemoteSession) -> str:
    return "".join([p async for p in remote.chat([{"role": "user", "content": "hi"}], {})])


def test_the_smaller_count_is_what_gets_paid(tmp_path):
    class Low(Claims):
        claim = {"prompt_tokens": 7, "completion_tokens": 1}

    async def run():
        async with rig(tmp_path, backend_cls=Low) as r:
            remote = buyer(r, tmp_path)
            await talk(remote)
            assert remote.usage["out_tokens"] == 1 and remote.used == 1

    asyncio.run(run())


def test_a_tall_receipt_stops_the_session_and_signs_nothing_for_it(tmp_path):
    class High(Claims):
        claim = {"prompt_tokens": 7, "completion_tokens": 900}

    async def run():
        async with rig(tmp_path, backend_cls=High) as r:
            remote = buyer(r, tmp_path)
            with pytest.raises(BeetleError) as e:
                await talk(remote)
            assert e.value.code == "usage_mismatch" and not remote.pipe_alive()
            assert remote.used == 0 and remote.covered <= remote.step + 10

    asyncio.run(run())


def test_every_voucher_is_on_disk_before_a_crash_can_lose_it(tmp_path):
    async def run():
        async with rig(tmp_path) as r:
            remote = buyer(r, tmp_path)
            remote._pay(30)
            assert Ledger(tmp_path / "ledger.json").entries["k"]["signed"] == 30 / 1e6

    asyncio.run(run())


def test_the_buyer_pays_in_steps_during_a_long_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(metered, "VOUCHER_WAIT", 3.0)

    async def run():
        async with rig(tmp_path, delay=0.02, backend_cls=Claims) as r:
            remote = buyer(r, tmp_path, deposit=20, step=2)
            text = await talk(remote)
            assert text == "I do not know " and not r.session.stopped
            assert remote.covered >= 4

    asyncio.run(run())


def test_an_empty_deposit_refuses_before_sending(tmp_path):
    async def run():
        async with rig(tmp_path) as r:
            remote = buyer(r, tmp_path, deposit=5, step=2)
            remote.used = 5
            with pytest.raises(BeetleError) as e:
                await talk(remote)
            assert e.value.code == "budget_exceeded" and r.backend.requests == []

    asyncio.run(run())


def test_the_closing_voucher_names_what_was_counted(tmp_path):
    async def run():
        async with rig(tmp_path, backend_cls=Claims) as r:
            remote = buyer(r, tmp_path)
            remote.peer = SimpleNamespace(close=lambda: asyncio.sleep(0))
            await talk(remote)
            used = remote.used
            await remote.close()
            await asyncio.sleep(0.05)
            vouchers = [m for m in r.reports if m["t"] == "voucher"]
            assert vouchers[-1]["cumulative"] == used
            assert "closed_at" in remote.core.ledger.entries["k"]

    asyncio.run(run())
