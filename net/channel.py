"""The sealed channel between a buyer and the seller's runtime environment.

The outer link ends at the host, so nothing on it is trusted. This layer ends
inside the environment. Its shape is RFC 9458 with chunked responses: the buyer
runs HPKE to the environment's enc_pk once, and both directions then use keys
exported from that context. The composition is ours and has not been audited.

    first frame   OPEN | suite | session_id | enc | HPKE.Seal(hello)
    request keys  Export("beetle/v1/request"),  salted with enc
    response keys Export("beetle/v1/response"), salted with enc | random nonce
    frame i       AEAD(key, base XOR i, aad = domain | direction | session_id | i)
    frame body    kind(1) ack(4) len(2) payload zeros

Every frame on the wire has the same length, in both directions, from the very
first one. In `constant` mode the environment also sends one frame per tick for
as long as a response runs, dummy frames included, then keeps going to the end
of a duration bucket. `burst` sends each frame as soon as there is data.

Protects against:
- a host or network reading or changing requests and responses;
- replay: an opening frame works once, and later frames must arrive in counter
  order, so neither can be processed twice;
- frames moved between sessions or directions, which fail authentication;
- truncation: a response is complete only when its authenticated end arrives;
- with `constant`, token timing and response length (to the bucket), which is
  what Whisper Leak reads. This has not been tested against that classifier.

Does not protect against:
- whoever runs the environment. Without attestation that is the host itself, so
  the guarantee is only as strong as the machine's tier;
- a stolen enc_pk secret: there is no forward secrecy within one key's lifetime;
- a host dropping or stalling traffic, or replaying a resume frame, which only
  forces a resend of ciphertext it has already seen;
- what stays visible: when activity starts, its bucketed length and the request
  tier in `constant`, and token timing in `burst`;
- GPU, memory and power side channels on the machine itself.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import struct
import time
from collections import deque

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
from pyhpke import PyHPKEError

import beetle_crypto as bc

FRAME = 4096
TICK = 0.05             # seconds between frames in constant mode
MIN_TICK = 0.005
DURATION_BUCKET = 5.0   # constant mode rounds activity up to a multiple of this
REQUEST_TIERS = (4096, 16384, 65536)  # a request fills the smallest tier it fits
RETAIN_BYTES = 64 << 20  # sent frames kept per session so a new link can resume
MAX_MESSAGE = 16 << 20

MODES = ("burst", "constant")
PAD, DATA = 0, 1        # frame kinds
OPEN, RESUME = 1, 2     # kinds of frame that start a link

PROTOCOL = b"beetle/v1/inner"
SUITE_ID = struct.pack(">HHH", 0x0020, 0x0001, 0x0003)  # X25519, HKDF-SHA256, ChaCha20-Poly1305
SESSION_ID_LEN = 32
ENC_LEN = 32
PREFIX_LEN = 1 + len(SUITE_ID) + SESSION_ID_LEN + ENC_LEN
RESPONSE_NONCE_LEN = 32
RESUME_NONCE_LEN = 12
TAG_LEN = 16
_BODY = struct.Struct(">BIH")      # kind, ack, payload length
_MESSAGE = struct.Struct(">II")    # body length, padding length


class ChannelError(Exception):
    pass


class ReplayError(ChannelError):
    pass


def _info(session_id: bytes) -> bytes:
    return PROTOCOL + b"\0" + SUITE_ID + session_id


def _prefix(kind: int, session_id: bytes, enc: bytes) -> bytes:
    return bytes([kind]) + SUITE_ID + session_id + enc


def peek(frame: bytes) -> tuple[int, bytes, bytes]:
    """Kind, session_id and enc of a link's first frame. None of it is authenticated yet."""
    if len(frame) < PREFIX_LEN or frame[1:7] != SUITE_ID:
        raise ChannelError("not a frame this version understands")
    return frame[0], frame[7:7 + SESSION_ID_LEN], frame[7 + SESSION_ID_LEN:PREFIX_LEN]


class _Direction:
    """One way of a session: a key, a base nonce and the additional data."""

    def __init__(self, secret: bytes, salt: bytes, label: bytes, session_id: bytes):
        prk = bc.hmac_sha256(salt, secret)  # HKDF-Extract
        expand = lambda info, n: HKDFExpand(hashes.SHA256(), n, info).derive(prk)  # noqa: E731
        self.aead = ChaCha20Poly1305(expand(b"key", 32))
        self.base = int.from_bytes(expand(b"nonce", 12), "big")
        self.aad = PROTOCOL + b"\0" + label + b"\0" + session_id

    def _bind(self, seq: int) -> tuple[bytes, bytes]:
        return (self.base ^ seq).to_bytes(12, "big"), self.aad + seq.to_bytes(8, "big")

    def seal(self, seq: int, plaintext: bytes) -> bytes:
        nonce, aad = self._bind(seq)
        return self.aead.encrypt(nonce, plaintext, aad)

    def open(self, seq: int, ciphertext: bytes) -> bytes:
        nonce, aad = self._bind(seq)
        return self.aead.decrypt(nonce, ciphertext, aad)


class Channel:
    """One session's frames, from one end. Everything a lost link must not lose is here."""

    def __init__(self, ctx, enc: bytes, session_id: bytes, frame: int, server: bool):
        self.session_id, self.enc, self.frame, self.server = session_id, enc, frame, server
        self._response = ctx.export(b"beetle/v1/response", 32)
        self._resume = ChaCha20Poly1305(ctx.export(b"beetle/v1/resume", 32))
        request = _Direction(ctx.export(b"beetle/v1/request", 32), enc, b"request", session_id)
        self._head = b""  # what goes in front of the next frame
        if server:
            self._head = os.urandom(RESPONSE_NONCE_LEN)
            self._tx, self._rx = self._response_direction(self._head), request
        else:
            self._tx, self._rx = request, None  # the nonce arrives with the first reply
        self.tx = self.rx = 0
        self._sent: deque[bytes] = deque(maxlen=RETAIN_BYTES // frame)

    def _response_direction(self, nonce: bytes) -> _Direction:
        return _Direction(self._response, self.enc + nonce, b"response", self.session_id)

    @property
    def room(self) -> int:
        """Payload bytes the next frame can carry."""
        return self.frame - len(self._head) - _BODY.size - TAG_LEN

    def seal(self, kind: int, payload: bytes = b"") -> bytes:
        body = _BODY.pack(kind, self.rx, len(payload)) + payload
        frame = self._head + self._tx.seal(self.tx, body.ljust(self.room + _BODY.size, b"\0"))
        self._head = b""
        self.tx += 1
        self._sent.append(frame)
        return frame

    def open(self, frame: bytes) -> tuple[int, bytes]:
        """Authenticate the next frame in order. Nothing advances if it fails."""
        if len(frame) != self.frame:
            raise ChannelError(f"frame of {len(frame)} bytes, expected {self.frame}")
        direction = self._rx
        if direction is None:
            nonce, frame = frame[:RESPONSE_NONCE_LEN], frame[RESPONSE_NONCE_LEN:]
            direction = self._response_direction(nonce)
        try:
            body = direction.open(self.rx, frame)
        except InvalidTag:
            raise ChannelError(f"frame {self.rx} failed authentication") from None
        self._rx = direction
        self.rx += 1
        kind, ack, n = _BODY.unpack_from(body)
        while self._sent and self.tx - len(self._sent) < ack:
            self._sent.popleft()  # the peer has these, so a resume will not need them
        return kind, body[_BODY.size:_BODY.size + n]

    def frames_from(self, seq: int) -> list[bytes]:
        """Sealed frames from `seq` on, unchanged, for a peer that missed them."""
        first = self.tx - len(self._sent)
        if not first <= seq <= self.tx:
            raise ChannelError(f"frame {seq} is not held (have {first} to {self.tx})")
        return list(self._sent)[seq - first:]

    def _resume_aad(self, server: bool, prefix: bytes) -> bytes:
        return PROTOCOL + (b"\0resume-server\0" if server else b"\0resume-client\0") + prefix

    def resume_frame(self) -> bytes:
        """First frame on a new link: names the session, says how far we have read."""
        prefix = _prefix(RESUME, self.session_id, self.enc)
        nonce = os.urandom(RESUME_NONCE_LEN)
        body = self.rx.to_bytes(8, "big").ljust(
            self.frame - PREFIX_LEN - RESUME_NONCE_LEN - TAG_LEN, b"\0")
        sealed = self._resume.encrypt(nonce, body, self._resume_aad(self.server, prefix))
        return prefix + nonce + sealed

    def read_resume(self, frame: bytes) -> int:
        """The peer's receive counter, from its resume frame."""
        prefix, rest = frame[:PREFIX_LEN], frame[PREFIX_LEN:]
        nonce, sealed = rest[:RESUME_NONCE_LEN], rest[RESUME_NONCE_LEN:]
        try:
            body = self._resume.decrypt(nonce, sealed, self._resume_aad(not self.server, prefix))
        except InvalidTag:
            raise ChannelError("resume frame failed authentication") from None
        return int.from_bytes(body[:8], "big")


def connect(enc_pk: bytes, session_id: bytes, hello: bytes,
            frame: int = FRAME) -> tuple[Channel, bytes]:
    """Buyer: start a session. Returns the channel and the opening frame to send."""
    if len(session_id) != SESSION_ID_LEN:
        raise ValueError(f"session_id must be {SESSION_ID_LEN} bytes")
    pk = bc._SUITE.kem.deserialize_public_key(enc_pk)
    enc, ctx = bc._SUITE.create_sender_context(pk, info=_info(session_id))
    prefix = _prefix(OPEN, session_id, enc)
    room = frame - len(prefix) - TAG_LEN
    if len(hello) + 2 > room:
        raise ValueError("hello does not fit in the opening frame")
    body = (len(hello).to_bytes(2, "big") + hello).ljust(room, b"\0")
    return Channel(ctx, enc, session_id, frame, server=False), prefix + ctx.seal(body, aad=prefix)


def accept(keypair: bc.HpkeKeyPair, first: bytes, seen: set[bytes]) -> tuple[Channel, bytes]:
    """Environment: take an opening frame. Returns the channel and the buyer's hello.

    `seen` holds every enc this key has accepted. It only grows on success, so
    garbage cannot fill it, and it dies with the key, so a restart cannot be
    used to replay an old opening frame.
    """
    kind, session_id, enc = peek(first)
    if kind != OPEN:
        raise ChannelError("expected an opening frame")
    if enc in seen:
        raise ReplayError("opening frame replayed")
    try:
        ctx = bc._SUITE.create_recipient_context(enc, keypair._kp.private_key,
                                                 info=_info(session_id))
        body = ctx.open(first[PREFIX_LEN:], aad=first[:PREFIX_LEN])
    except PyHPKEError:
        raise ChannelError("opening frame failed authentication") from None
    seen.add(enc)
    return (Channel(ctx, enc, session_id, len(first), server=True),
            body[2:2 + int.from_bytes(body[:2], "big")])


class Pipe:
    """A channel's messages, carried over whichever link is attached right now.

    Messages are length-prefixed JSON in a byte stream that is cut into frames,
    so a message can span frames and a frame can hold several. Pacing happens
    here, on the sealed side of the host: by the time the host sees a frame its
    size and time are already fixed.
    """

    def __init__(self, channel: Channel, *, mode: str = "burst", tick: float = TICK,
                 bucket: float = DURATION_BUCKET, on_lost=lambda exc: None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.ch, self.mode, self.on_lost = channel, mode, on_lost
        self.tick = max(tick, MIN_TICK)
        self.bucket = max(bucket, self.tick)
        self.link = None
        self.attached = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._out = bytearray()
        self._in = bytearray()
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._wake = asyncio.Event()
        self._started = self._until = 0.0

    def attach(self, link, peer_rx: int | None = None) -> None:
        """Move onto `link`, first resending whatever the peer has not read."""
        self.detach()
        resend = [] if peer_rx is None else self.ch.frames_from(peer_rx)
        self.link = link
        self._tasks = [self._run(self._read(link), link), self._run(self._write(link, resend), link)]
        self.attached.set()

    def detach(self) -> None:
        self.link = None
        self.attached.clear()
        for task in self._tasks:
            task.cancel()
        self._tasks = []

    def _run(self, coro, link) -> asyncio.Task:
        task = asyncio.create_task(coro)
        task.add_done_callback(lambda t: self._ended(t, link))
        return task

    def _ended(self, task: asyncio.Task, link) -> None:
        if task.cancelled() or link is not self.link:
            return
        self.detach()
        self.on_lost(task.exception())

    def send(self, msg: dict, tiered: bool = False) -> bytes:
        """Queue a message; `tiered` pads it to whole frames of a request tier."""
        body = bc.canonical_json(msg)
        skip = self._tier_fill(len(body) + _MESSAGE.size) if tiered else 0
        self._out += _MESSAGE.pack(len(body), skip) + body + bytes(skip)
        self._wake.set()
        return body

    def _tier_fill(self, size: int) -> int:
        room = self.ch.room
        for t in REQUEST_TIERS:
            if size <= (full := t // self.ch.frame * room):
                return full - size
        step = REQUEST_TIERS[-1] // self.ch.frame * room
        return math.ceil(size / step) * step - size

    async def recv(self) -> dict:
        item = await self._inbox.get()
        if isinstance(item, Exception):
            raise item
        return item

    def fail(self, exc: Exception) -> None:
        """Make the next recv() raise: this session cannot continue."""
        self._inbox.put_nowait(exc)

    def hold(self) -> None:
        """Constant mode: a response is starting, so send frames from now on.

        A response that starts inside an earlier one's bucket extends that
        window, or the host could time the earlier one by where its padding ended.
        """
        now = time.monotonic()
        if now >= self._until:
            self._started = now
        self._until = math.inf
        self._wake.set()

    def release(self) -> None:
        """The response ended; keep sending to the end of its duration bucket."""
        elapsed = max(time.monotonic() - self._started, 1e-9)
        self._until = self._started + math.ceil(elapsed / self.bucket) * self.bucket

    async def settled(self) -> None:
        """Wait out the bucket, so ending a session cannot cut the padding short."""
        if self.mode == "constant":
            await asyncio.sleep(max(0.0, self._until - time.monotonic()))

    async def _read(self, link) -> None:
        while True:
            kind, payload = self.ch.open(await link.read_exact(self.ch.frame))
            if kind != DATA:
                continue
            self._in += payload
            while len(self._in) >= _MESSAGE.size:
                n, skip = _MESSAGE.unpack_from(self._in)
                if n + skip > MAX_MESSAGE:
                    raise ChannelError("message too large")
                end = _MESSAGE.size + n + skip
                if len(self._in) < end:
                    break
                self._inbox.put_nowait(json.loads(bytes(self._in[_MESSAGE.size:_MESSAGE.size + n])))
                del self._in[:end]

    async def _write(self, link, resend: list[bytes]) -> None:
        for frame in resend:
            await link.write(frame)
        constant = self.mode == "constant"
        due = time.monotonic()
        while True:
            await self._wake.wait()
            if constant:
                now = time.monotonic()
                if due > now:
                    await asyncio.sleep(due - now)
                due = max(due, now) + self.tick
            payload = bytes(self._out[:self.ch.room])
            if payload:
                del self._out[:len(payload)]
                await link.write(self.ch.seal(DATA, payload))
            elif constant and time.monotonic() < self._until:
                await link.write(self.ch.seal(PAD))
            else:
                self._wake.clear()
