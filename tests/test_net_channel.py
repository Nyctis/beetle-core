"""Inner channel tests: keys, framing, replay, pacing and resume, with no network."""

from __future__ import annotations

import asyncio
import struct
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import net  # noqa: E402,F401

import beetle_crypto as bc  # noqa: E402
from net import channel  # noqa: E402
from net.channel import (FRAME, ChannelError, Pipe, ReplayError, accept, connect, peek,  # noqa: E402
                         DATA, PAD)

SID = bytes(range(32))


class MemLink:
    """One end of an in-memory byte pipe, with the Link interface."""

    def __init__(self, inbox: asyncio.StreamReader, outbox: asyncio.StreamReader):
        self.inbox, self.outbox = inbox, outbox
        self.written: list[tuple[float, bytes]] = []

    async def write(self, data: bytes) -> None:
        self.written.append((time.monotonic(), data))
        self.outbox.feed_data(data)

    async def read_exact(self, n: int) -> bytes:
        return await self.inbox.readexactly(n)

    def cut(self) -> None:
        self.inbox.feed_eof()
        self.outbox.feed_eof()


def mem_pair() -> tuple[MemLink, MemLink]:
    a, b = asyncio.StreamReader(limit=1 << 26), asyncio.StreamReader(limit=1 << 26)
    return MemLink(a, b), MemLink(b, a)


def opened(frame: int = FRAME):
    kp = bc.HpkeKeyPair()
    client, first = connect(kp.public_bytes, SID, b'{"hello":1}', frame)
    server, hello = accept(kp, first, set())
    assert hello == b'{"hello":1}'
    return client, server


def test_opening_frame_is_one_frame_long_and_names_the_session():
    kp = bc.HpkeKeyPair()
    _, first = connect(kp.public_bytes, SID, b"x")
    assert len(first) == FRAME
    kind, session_id, enc = peek(first)
    assert (kind, session_id, len(enc)) == (channel.OPEN, SID, 32)


def test_frames_round_trip_and_are_all_one_length():
    client, server = opened()
    frames = [server.seal(DATA, b"first"), server.seal(PAD), server.seal(DATA, b"x" * server.room)]
    assert {len(f) for f in frames} == {FRAME}
    assert client.open(frames[0]) == (DATA, b"first")  # carries the response nonce
    assert client.open(frames[1]) == (PAD, b"")
    assert client.open(frames[2])[1] == b"x" * server.room
    request = client.seal(DATA, b"question")
    assert len(request) == FRAME and server.open(request) == (DATA, b"question")


def test_opening_frame_cannot_be_replayed():
    kp = bc.HpkeKeyPair()
    _, first = connect(kp.public_bytes, SID, b"x")
    seen: set[bytes] = set()
    accept(kp, first, seen)
    with pytest.raises(ReplayError):
        accept(kp, first, seen)


def test_garbage_does_not_fill_the_replay_set():
    kp = bc.HpkeKeyPair()
    _, first = connect(kp.public_bytes, SID, b"x")
    seen: set[bytes] = set()
    with pytest.raises(ChannelError):
        accept(kp, first[:-1] + bytes([first[-1] ^ 1]), seen)
    assert not seen


def test_wrong_key_and_foreign_session_are_refused():
    kp = bc.HpkeKeyPair()
    _, first = connect(kp.public_bytes, SID, b"x")
    with pytest.raises(ChannelError):
        accept(bc.HpkeKeyPair(), first, set())
    swapped = first[:7] + bytes(32) + first[39:]
    with pytest.raises(ChannelError):
        accept(kp, swapped, set())  # session_id is in the HPKE info


def test_later_frames_cannot_be_replayed_reordered_or_reflected():
    client, server = opened()
    a, b = server.seal(DATA, b"a"), server.seal(DATA, b"b")
    with pytest.raises(ChannelError):
        client.open(b)  # out of order
    client.open(a)
    with pytest.raises(ChannelError):
        client.open(a)  # replay
    client.open(b)
    reflected = client.seal(DATA, b"c")
    with pytest.raises(ChannelError):
        client.open(reflected)  # a side reading its own frame


def test_a_tampered_frame_leaves_state_alone():
    client, server = opened()
    good = server.seal(DATA, b"a")
    bad = bytearray(good)
    bad[100] ^= 1
    with pytest.raises(ChannelError):
        client.open(bytes(bad))
    assert client.rx == 0 and client.open(good) == (DATA, b"a")


def test_two_sessions_share_no_keys():
    kp = bc.HpkeKeyPair()
    one, first = connect(kp.public_bytes, SID, b"x")
    other, first2 = connect(kp.public_bytes, SID, b"x")
    server_one, _ = accept(kp, first, set())
    server_two, _ = accept(kp, first2, set())
    with pytest.raises(ChannelError):
        other.open(server_one.seal(DATA, b"a"))
    with pytest.raises(ChannelError):
        server_two.open(one.seal(DATA, b"a"))


def test_wrong_length_is_refused():
    client, server = opened()
    with pytest.raises(ChannelError):
        client.open(server.seal(DATA, b"a")[:-1])


def test_every_info_and_aad_is_domain_separated():
    client, server = opened()
    assert server._tx.aad.startswith(b"beetle/v1/inner\0response\0" + SID)
    assert client._tx.aad.startswith(b"beetle/v1/inner\0request\0" + SID)
    assert channel._info(SID).startswith(b"beetle/v1/inner\0")


def test_acked_frames_are_dropped_and_the_rest_can_be_resent():
    client, server = opened()
    frames = [server.seal(DATA, bytes([i])) for i in range(4)]
    for f in frames[:3]:
        client.open(f)
    server.open(client.seal(PAD))  # carries ack = 3
    assert server.frames_from(3) == frames[3:]
    with pytest.raises(ChannelError):
        server.frames_from(2)


def test_resume_frame_carries_the_counter_and_is_authenticated():
    client, server = opened()
    client.open(server.seal(DATA, b"a"))
    frame = client.resume_frame()
    assert len(frame) == FRAME and peek(frame)[0] == channel.RESUME
    assert server.read_resume(frame) == 1
    assert client.read_resume(server.resume_frame()) == 0
    with pytest.raises(ChannelError):
        client.read_resume(client.resume_frame())  # reflected
    with pytest.raises(ChannelError):
        server.read_resume(frame[:-1] + bytes([frame[-1] ^ 1]))


def pipes(**server_options):
    client_ch, server_ch = opened()
    return Pipe(client_ch), Pipe(server_ch, **server_options)


def test_messages_cross_frames_in_both_directions():
    async def go():
        left, right = mem_pair()
        client, server = pipes()
        client.attach(left)
        server.attach(right)
        big = {"text": "y" * (3 * FRAME)}
        client.send(big)
        assert await server.recv() == big
        server.send({"a": 1})
        server.send({"b": 2})
        assert await client.recv() == {"a": 1}
        assert await client.recv() == {"b": 2}
        assert {len(d) for _, d in left.written + right.written} == {FRAME}
        client.detach()
        server.detach()
    asyncio.run(go())


@pytest.mark.parametrize("size, frames", [(10, 1), (3000, 1), (5000, 4), (20000, 16), (70000, 32)])
def test_requests_fill_a_tier(size, frames):
    async def go():
        left, right = mem_pair()
        client, server = pipes()
        client.attach(left)
        server.attach(right)
        msg = {"text": "z" * size}
        client.send(msg, tiered=True)
        assert await server.recv() == msg
        assert len(left.written) == frames
        client.detach()
        server.detach()
    asyncio.run(go())


def test_constant_mode_ticks_through_silence_and_to_the_bucket():
    async def go():
        left, right = mem_pair()
        client, server = pipes(mode="constant", tick=0.01, bucket=0.3)
        client.attach(left)
        server.attach(right)
        await asyncio.sleep(0.05)
        assert not right.written  # idle: nothing on the wire
        server.hold()
        await asyncio.sleep(0.1)
        assert len(right.written) >= 5  # dummies before the first token
        server.send({"t": "delta"})
        assert await client.recv() == {"t": "delta"}
        server.release()
        await asyncio.sleep(0.5)
        stamps = [t for t, _ in right.written]
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        assert {len(d) for _, d in right.written} == {FRAME}
        assert 25 <= len(stamps) <= 31  # 0.3 s at 10 ms, and then it stops
        assert sorted(gaps)[len(gaps) // 2] == pytest.approx(0.01, abs=0.004)
        client.detach()
        server.detach()
    asyncio.run(go())


def test_a_second_response_extends_the_first_ones_window():
    async def go():
        left, right = mem_pair()
        client, server = pipes(mode="constant", tick=0.01, bucket=0.2)
        client.attach(left)
        server.attach(right)
        server.hold()
        server.release()
        started = server._started
        await asyncio.sleep(0.05)
        server.hold()  # inside the first bucket
        assert server._started == started
        server.release()
        assert server._until == pytest.approx(started + 0.2)
        await server.settled()
        await asyncio.sleep(0.03)
        count = len(right.written)
        assert 17 <= count <= 21
        server.hold()  # after the window: a new one starts
        assert server._started > started
        client.detach()
        server.detach()
    asyncio.run(go())


def test_burst_mode_sends_nothing_without_data():
    async def go():
        left, right = mem_pair()
        client, server = pipes(mode="burst")
        client.attach(left)
        server.attach(right)
        server.hold()
        await asyncio.sleep(0.05)
        assert not right.written
        client.detach()
        server.detach()
    asyncio.run(go())


def test_a_new_link_resends_exactly_what_was_missed():
    async def go():
        left, right = mem_pair()
        lost = []
        client_ch, server_ch = opened()
        client = Pipe(client_ch, on_lost=lost.append)
        server = Pipe(server_ch, on_lost=lost.append)
        client.attach(left)
        server.attach(right)
        server.send({"n": 1})
        assert await client.recv() == {"n": 1}
        left.cut()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(lost) == 2 and client.link is None and server.link is None
        server.send({"n": 2})  # queued while no link exists
        client.send({"up": 1})
        left, right = mem_pair()
        peer_rx = server_ch.read_resume(client_ch.resume_frame())
        back = client_ch.read_resume(server_ch.resume_frame())
        server.attach(right, peer_rx)
        client.attach(left, back)
        assert await client.recv() == {"n": 2}
        assert await server.recv() == {"up": 1}
        client.detach()
        server.detach()
    asyncio.run(go())


def test_oversized_declared_message_is_refused():
    async def go():
        left, right = mem_pair()
        lost = []
        client_ch, server_ch = opened()
        server = Pipe(server_ch, on_lost=lost.append)
        server.attach(right)
        await left.write(client_ch.seal(DATA, struct.pack(">II", channel.MAX_MESSAGE, 1)))
        await asyncio.sleep(0.01)
        assert isinstance(lost[0], ChannelError)
    asyncio.run(go())


def test_session_id_must_be_32_bytes():
    with pytest.raises(ValueError):
        connect(bc.HpkeKeyPair().public_bytes, b"short", b"x")


def test_a_zero_bucket_cannot_divide_by_zero():
    async def go():
        client, server = pipes(mode="constant", tick=0.01, bucket=0)
        server.hold()
        server.release()
        assert server._until > 0
    asyncio.run(go())


def test_small_frames_work():
    client, server = opened(512)
    assert client.open(server.seal(DATA, b"a")) == (DATA, b"a")
