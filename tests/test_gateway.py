"""Gateway tests: real iroh links on loopback against a small fake OpenAI server."""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import net  # noqa: E402,F401

import serve.client as client  # noqa: E402
from net import channel, echo  # noqa: E402
from net.peer import Peer  # noqa: E402
from net.rendezvous import FileRendezvous, MemoryRendezvous  # noqa: E402
from serve.client import GatewayError, ReceiptError, Session, SessionLost  # noqa: E402
from serve.gateway import Backend, Environment, Gateway, Spool  # noqa: E402

SECRET_PROMPT = "what is the name of my cat"
ANSWER = "Your cat is called Pixel and she likes oolong tea".split(" ")
BASE_ANSWER = "I do not know".split(" ")


class FakeBackend:
    """Just enough of an OpenAI-compatible server, on a real socket."""

    def __init__(self, delay: float = 0.0):
        self.delay, self.adapters, self.requests, self.fail_next = delay, {}, [], False

    async def __aenter__(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *exc):
        self.server.close()

    async def handle(self, reader, writer):
        head = (await reader.readuntil(b"\r\n\r\n")).decode()
        line, *headers = head.split("\r\n")
        path = line.split(" ")[1]
        length = next((int(h.split(":")[1]) for h in headers if h.lower().startswith("content-length")), 0)
        body = json.loads(await reader.readexactly(length)) if length else {}

        def reply(status: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            writer.write(f"HTTP/1.1 {status} X\r\ncontent-length: {len(data)}\r\n"
                         "connection: close\r\n\r\n".encode() + data)

        if path == "/v1/models":
            reply(200, {"data": [{"id": "base"}]})
        elif path == "/v1/load_lora_adapter":
            files = sorted(p.name for p in Path(body["lora_path"]).iterdir())
            self.adapters[body["lora_name"]] = files
            reply(200, {})
        elif path == "/v1/unload_lora_adapter":
            del self.adapters[body["lora_name"]]
            reply(200, {})
        elif path == "/v1/chat/completions":
            self.requests.append(body)
            if self.fail_next:
                self.fail_next = False
                reply(500, {})
            else:
                await self.stream(writer, body)
        else:
            reply(404, {})
        await writer.drain()
        writer.close()

    async def stream(self, writer, body):
        writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n")
        words = ANSWER if body["model"] in self.adapters else BASE_ANSWER
        for word in words:
            event = {"choices": [{"delta": {"content": word + " "}}]}
            writer.write(f"data: {json.dumps(event)}\n\n".encode())
            await writer.drain()
            await asyncio.sleep(self.delay)
        usage = {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": len(words)}}
        writer.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())


class Stack:
    """A gateway, its fake backend and one way in."""

    def __init__(self, tmp: Path, backend: FakeBackend, seller: Peer, env: Environment, tap):
        self.tmp, self.backend, self.seller, self.env, self.tap = tmp, backend, seller, env, tap
        self.gateway = Gateway(seller, env, [f"127.0.0.1:{seller.port}"], tap)
        self.serving: list[asyncio.Task] = []

    def rendezvous(self) -> MemoryRendezvous:
        buyer, seller = MemoryRendezvous.pair()
        self.serving.append(asyncio.create_task(self.gateway.serve(seller)))
        return buyer

    async def session(self, **options) -> Session:
        options.setdefault("mode", "burst")
        peer = await Peer.bind()
        session = await Session.open(self.rendezvous(), self.env.enc_pk, peer=peer,
                                     addrs=[f"127.0.0.1:{peer.port}"], **options)
        session.owns_peer = True
        return session

    def tapped(self) -> list[tuple[bytes, bytes]]:
        records, blob, i = [], (self.tmp / "tap").read_bytes(), 0
        while i < len(blob):
            n = struct.unpack(">I", blob[i + 1:i + 5])[0]
            records.append((blob[i:i + 1], blob[i + 5:i + 5 + n]))
            i += 5 + n
        return records


@asynccontextmanager
async def stack(tmp_path, delay: float = 0.0, **options):
    async with FakeBackend(delay) as backend:
        env = Environment(Backend(backend.url), {"base"}, Spool(tmp_path / "spool", 1 << 30),
                          tmp_path / "scratch", **options)
        seller = await Peer.bind(services={echo.ALPN: echo.answer})
        with open(tmp_path / "tap", "ab") as tap:
            s = Stack(tmp_path, backend, seller, env, tap)
            try:
                yield s
            finally:
                for task in s.serving:
                    task.cancel()
                await seller.close()


async def chat(session: Session, model: str = "base", prompt: str = SECRET_PROMPT) -> str:
    return "".join([p async for p in session.chat([{"role": "user", "content": prompt}], model=model)])


def make_adapter(path: Path) -> Path:
    path.mkdir()
    (path / "adapter_config.json").write_text('{"marker": "PLAINTEXT-ADAPTER-CONFIG"}')
    (path / "adapter_model.safetensors").write_bytes(b"PLAINTEXT-WEIGHTS" * 100)
    (path / "README.md").write_text("not part of the adapter")
    return path


@pytest.mark.parametrize("mode", ["burst", "constant"])
def test_chat_streams_with_a_verified_receipt(tmp_path, mode):
    async def go():
        async with stack(tmp_path) as s:
            session = await s.session(mode=mode, tick=0.01, bucket=0.1)
            assert await chat(session) == "I do not know "
            first = session.receipt
            assert (first["seq"], first["in_tokens"], first["out_tokens"]) == (1, 7, 4)
            await chat(session)
            assert session.receipt["seq"] == 2 and session.receipt["out_tokens"] == 8
            await session.close()
    asyncio.run(go())


def test_gateway_declares_what_it_is(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            assert session.tier == "T3-C" and session.info["protected"] is False
            assert "root" in session.info["notice"]
            await session.close()
    asyncio.run(go())


def test_a_protected_tier_cannot_be_declared(tmp_path):
    with pytest.raises(ValueError, match="evidence"):
        Environment(None, set(), Spool(tmp_path, 1), tmp_path, tier="T2-GPU")


def test_a_client_refuses_a_tier_claimed_without_evidence(tmp_path, monkeypatch):
    async def go():
        async with stack(tmp_path) as s:
            monkeypatch.setattr("serve.gateway.UNPROVEN_TIERS", {"T2-GPU": "claimed"})
            s.env.tier = "T2-GPU"
            with pytest.raises(GatewayError, match="unproven_tier"):
                await s.session()
    asyncio.run(go())


def test_the_host_sees_only_ciphertext_in_equal_frames(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            session = await s.session(mode="constant", tick=0.01, bucket=0.1)
            await chat(session)
            await session.close()
            records = s.tapped()
            assert {len(data) for _, data in records} == {channel.FRAME}
            assert {d for d, _ in records} == {b"<", b">"}
            everything = b"".join(data for _, data in records)
            for needle in (SECRET_PROMPT, "know", "delta", "messages", "chat"):
                assert needle.encode() not in everything
            assert s.backend.requests[0]["messages"][0]["content"] == SECRET_PROMPT
    asyncio.run(go())


def test_constant_mode_covers_first_token_time_and_length(tmp_path):
    async def go():
        async with stack(tmp_path, delay=0.03) as s:
            session = await s.session(mode="constant", tick=0.01, bucket=0.5)
            await chat(session)
            await asyncio.sleep(0.6)
            sent = sum(1 for d, _ in s.tapped() if d == b">")
            await session.close()
            assert 40 <= sent <= 56  # about 0.5 s of ticks, plus "ready" and "close"
    asyncio.run(go())


def test_an_unknown_model_is_refused_and_the_session_lives(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            with pytest.raises(GatewayError, match="unknown_model"):
                await chat(session, model="nope")
            s.backend.fail_next = True
            with pytest.raises(GatewayError, match="backend"):
                await chat(session)
            assert await chat(session) == "I do not know "
            await session.close()
    asyncio.run(go())


def test_buyer_params_cannot_touch_what_the_gateway_sets(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            [p async for p in session.chat([], model="base", temperature=0.5, cache_salt="mine",
                                           stream=False)]
            body = s.backend.requests[0]
            assert body["temperature"] == 0.5 and body["stream"] is True
            assert body["cache_salt"] != "mine"
            await session.close()
    asyncio.run(go())


def test_sessions_do_not_share_a_prefix_cache(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            one, two = await s.session(), await s.session()
            await chat(one)
            await chat(two)
            a, b = s.backend.requests
            assert a["cache_salt"] != b["cache_salt"]
            await one.close()
            await two.close()
    asyncio.run(go())


def test_adapter_lifecycle(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            adapter = make_adapter(tmp_path / "adapter")
            owner, other = await s.session(), await s.session()
            name = await owner.upload_adapter(adapter)
            assert s.backend.adapters == {name: ["adapter_config.json", "adapter_model.safetensors"]}
            assert (s.env.scratch / name).is_dir()
            assert not list(s.env.spool.root.iterdir())  # ciphertext is gone once loaded

            assert "Pixel" in await chat(owner, model=name)
            with pytest.raises(GatewayError, match="unknown_model"):
                await chat(other, model=name)  # another buyer cannot reach it

            everything = b"".join(data for _, data in s.tapped())
            assert b"PLAINTEXT" not in everything and b"adapter_config" not in everything

            await owner.close()
            assert s.backend.adapters == {} and not list(s.env.scratch.iterdir())
            with pytest.raises(GatewayError, match="unknown_model"):
                await chat(other, model=name)
            assert "Pixel" not in await chat(other)
            await other.close()
    asyncio.run(go())


def test_adapter_tiers_hide_the_upload_size(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            await session.upload_adapter(make_adapter(tmp_path / "adapter"))
            chunks = [d for direction, d in s.tapped() if direction == b"<" and len(d) == 1 << 20]
            assert len(chunks) == 32  # 32 MiB tier for a few kilobytes
            await session.close()
    asyncio.run(go())


def test_an_abandoned_session_cleans_up_after_itself(tmp_path):
    async def go():
        async with stack(tmp_path, resume_window=0.2) as s:
            session = await s.session()
            await session.upload_adapter(make_adapter(tmp_path / "adapter"))
            session.closing = True  # this buyer will not come back
            await session.pipe.link.close()
            for _ in range(100):
                if not s.backend.adapters and not s.env.sessions:
                    break
                await asyncio.sleep(0.05)
            assert s.backend.adapters == {} and not s.env.sessions
            assert not list(s.env.scratch.iterdir())
    asyncio.run(go())


@pytest.mark.parametrize("mode", ["burst", "constant"])
def test_a_response_carries_on_across_a_new_link(tmp_path, mode):
    async def go():
        async with stack(tmp_path, delay=0.05) as s:
            session = await s.session(mode=mode, tick=0.01, bucket=0.1)
            await session.upload_adapter(make_adapter(tmp_path / "adapter"))
            name = list(s.backend.adapters)[0]
            pieces = []
            stream = session.chat([{"role": "user", "content": SECRET_PROMPT}], model=name)
            async for piece in stream:
                pieces.append(piece)
                if len(pieces) == 3:
                    await session.pipe.link.close()  # the link dies mid-response
            assert "".join(pieces).split() == ANSWER  # nothing lost, nothing twice
            assert session.receipt["seq"] == 1
            assert await chat(session) == "I do not know "  # and the session still works
            await session.close()
    asyncio.run(go())


def test_an_upload_carries_on_across_a_new_link(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            adapter = make_adapter(tmp_path / "adapter")
            upload = asyncio.create_task(session.upload_adapter(adapter))
            while (tmp_path / "tap").stat().st_size < 8 << 20:
                await asyncio.sleep(0.01)
            await session.pipe.link.close()  # cut while the ciphertext is moving
            name = await asyncio.wait_for(upload, 30)
            assert list(s.backend.adapters) == [name]
            assert "Pixel" in await chat(session, model=name)
            fetched = sum(1 for _, data in s.tapped() if len(data) == 1 << 20)
            assert fetched < 32 + 2 * 16  # what arrived before the cut was not fetched again
            await session.close()
    asyncio.run(go())


def test_an_unannounced_upload_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("serve.gateway.ANNOUNCE_WAIT", 0.1)

    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            stream = await session.pipe.link.open_stream()
            await stream.write(bytes(32))
            await asyncio.sleep(0.5)
            assert not list(s.env.spool.root.iterdir())
            assert await chat(session) == "I do not know "  # the session is unharmed
            await session.close()
    asyncio.run(go())


def test_a_malformed_message_gets_an_answer_not_silence(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            session.pipe.send({"t": "adapter"}, tiered=True)
            with pytest.raises(GatewayError, match="internal"):
                await session._expect("adapter_ok")
            assert await chat(session) == "I do not know "
            await session.close()
    asyncio.run(go())


def test_a_crash_cannot_leave_an_adapter_on_offer(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            name = "a0011223344556677"
            leftover = s.env.scratch / name
            leftover.mkdir(parents=True)
            (leftover / "adapter_config.json").write_text("{}")
            (s.env.scratch / "not-ours").mkdir()
            s.backend.adapters[name] = ["adapter_config.json"]
            s.env.models.add(name)  # what /v1/models would report after the crash
            await s.env.sweep()
            assert name not in s.env.models and name not in s.backend.adapters
            assert [p.name for p in s.env.scratch.iterdir()] == ["not-ours"]
    asyncio.run(go())


def test_frames_the_gateway_dropped_end_the_session_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "RESUME_TRIES", 2)

    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            session.window = 2.0
            await chat(session)
            gateway_side = next(iter(s.env.sessions.values())).ch
            gateway_side._sent.clear()  # as if the retention limit had passed
            session.pipe.ch.rx = 1      # and this end had missed frames it no longer holds
            await session.pipe.link.close()
            with pytest.raises(SessionLost):
                await chat(session)
    asyncio.run(go())


def test_links_that_keep_dying_end_the_session(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "RESUME_TRIES", 2)

    class Dying(client.Session):
        async def _resume(self):
            self._lost("the link died again")  # as if each new link failed at once

    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            session.__class__ = Dying
            await session.pipe.link.close()
            with pytest.raises(SessionLost, match="no progress"):
                await asyncio.wait_for(chat(session), 5)
    asyncio.run(go())


def test_a_seller_gone_for_good_is_given_up_on_in_time(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "RESUME_TRIES", 50)
    monkeypatch.setattr(client, "RESUME_BUDGET_S", 0.2)
    tries = []

    class Gone(client.Session):
        async def _dial(self):
            tries.append(1)
            raise TimeoutError

    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            session.__class__ = Gone
            await session.pipe.link.close()
            with pytest.raises(SessionLost, match="TimeoutError"):
                await asyncio.wait_for(chat(session), 5)
    asyncio.run(go())
    assert len(tries) < 5


def test_a_session_the_gateway_no_longer_has_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "RESUME_TRIES", 2)

    async def go():
        async with stack(tmp_path, delay=0.05) as s:
            session = await s.session()
            session.window = 2.0
            s.env.sessions.clear()
            await session.pipe.link.close()
            with pytest.raises(SessionLost):
                await chat(session)
    asyncio.run(go())


def test_an_opening_frame_replayed_on_a_new_link_is_refused(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            await chat(session)
            first = next(data for d, data in s.tapped() if d == b"<")
            sessions, requests = len(s.env.sessions), len(s.backend.requests)

            intruder = await Peer.bind()
            rendezvous = s.rendezvous()
            await rendezvous.publish({"id": intruder.id, "addrs": [f"127.0.0.1:{intruder.port}"]})
            them = await rendezvous.wait(5)
            link = await intruder.link(them["id"], them["addrs"], initiator=True, window=5)
            await link.write(first)
            with pytest.raises(Exception):
                await asyncio.wait_for(link.read_exact(channel.FRAME), 2)
            assert len(s.env.sessions) == sessions and len(s.backend.requests) == requests
            await intruder.close()
            await session.close()
    asyncio.run(go())


def test_a_forged_receipt_is_caught(tmp_path):
    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            session.transcript = b"\1" * 32  # this end saw a different conversation
            with pytest.raises(ReceiptError, match="transcript"):
                await chat(session)
            session.closing = True
    asyncio.run(go())


def test_two_buyers_at_once(tmp_path):
    async def go():
        async with stack(tmp_path, delay=0.01) as s:
            one, two = await asyncio.gather(s.session(), s.session())
            answers = await asyncio.gather(chat(one), chat(two))
            assert answers == ["I do not know "] * 2
            await asyncio.gather(one.close(), two.close())
    asyncio.run(go())


def test_echo_reports_the_callers_address(tmp_path):
    async def go():
        server = await Peer.bind(services={echo.ALPN: echo.answer})
        caller = await Peer.bind()
        result = await echo.probe(caller, [{"id": server.id, "addrs": [f"127.0.0.1:{server.port}"]}])
        assert result["observed"] == [f"127.0.0.1:{caller.port}"]
        assert result["addrs"][0] == f"127.0.0.1:{caller.port}"
        gone = await Peer.bind()
        node = {"id": gone.id, "addrs": [f"127.0.0.1:{gone.port}"]}
        await gone.close()
        silent = await echo.probe(caller, [node], timeout=0.5)
        assert silent["nat"] == "blocked" and silent["observed"] == []
        await server.close()
        await caller.close()
    asyncio.run(go())


def test_echo_is_refused_on_a_peer_that_does_not_offer_it(tmp_path):
    async def go():
        server, caller = await Peer.bind(), await Peer.bind()
        with pytest.raises(Exception):
            await echo.observe(caller, {"id": server.id, "addrs": [f"127.0.0.1:{server.port}"]}, 1)
        await server.close()
        await caller.close()
    asyncio.run(go())


@pytest.mark.parametrize("observed, local, nat", [
    ([], ["inner:1"], "blocked"),
    (["outer:5"], ["outer:5", "inner:5"], "open"),
    (["outer:5"], ["inner:5"], "unknown"),
    (["outer:5", "outer:5"], ["inner:5"], "eim"),
    (["outer:5", "outer:6"], ["inner:5"], "edm"),
])
def test_nat_classification(observed, local, nat):
    assert echo.classify(observed, local) == nat


def test_both_rendezvous_return_each_publication_once(tmp_path):
    async def go():
        a, b = MemoryRendezvous.pair()
        c = FileRendezvous(tmp_path / "c.json", tmp_path / "d.json", poll=0.01)
        d = FileRendezvous(tmp_path / "d.json", tmp_path / "c.json", poll=0.01)
        for left, right in ((a, b), (c, d)):
            contact = {"id": "x", "addrs": ["127.0.0.1:5"]}
            await left.publish(contact)
            got = await right.wait(1)
            assert (got["id"], got["addrs"]) == ("x", ["127.0.0.1:5"])
            with pytest.raises(TimeoutError):
                await right.wait(0.05)
            await left.publish(contact)  # the same contact again is a new event
            assert (await right.wait(1))["id"] == "x"
    asyncio.run(go())


def test_prompts_never_go_through_a_proxy_from_the_environment(tmp_path, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")

    async def go():
        async with stack(tmp_path) as s:
            session = await s.session()
            assert await chat(session) == "I do not know "
            await session.close()
    asyncio.run(go())


def test_backend_url_may_end_in_v1():
    assert str(Backend("http://h:1/v1/").http.base_url).rstrip("/") == "http://h:1"
