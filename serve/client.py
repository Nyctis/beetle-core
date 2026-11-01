"""The buyer's side of a private inference session.

    session = await Session.open(rendezvous)
    name = await session.upload_adapter(path)
    async for chunk in session.chat(messages, model=name):
        ...
    await session.close()

A lost link is handled here: the session dials again and both ends resend what
the other missed, so a response carries on from where it stopped, or the call
raises SessionLost and nothing is delivered twice.

This checks that every receipt is signed by the session's key and covers exactly
what was sent and received. It cannot check who holds that key, or what a
receipt's token counts are worth: enc_pk must come from somewhere the seller's
host cannot change, and without attestation the buyer is trusting the machine,
which is what the declared tier says. Pass enc_pk if you hold one; otherwise it
is taken from the rendezvous, as trustworthy as that is.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path

import net  # noqa: F401  (puts common/ on the import path)

import beetle_archive
import beetle_crypto as bc
from net import channel, transfer
from net.peer import Peer
from net.rendezvous import Rendezvous
from serve import ADAPTER_FILES, UNPROVEN_TIERS, adapter_aad, next_transcript, receipt_digest

RESUME_TRIES = 5
RESUME_BUDGET_S = 60.0  # a seller gone this long has gone: tell the buyer, who has others


class GatewayError(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


class SessionLost(Exception):
    """The link could not be restored. The gateway will clean up on its own."""


class ReceiptError(Exception):
    """A receipt does not match what this end saw. Stop paying this seller."""


class Session:
    def __init__(self, rendezvous: Rendezvous, peer: Peer, addrs: list[str], window: float,
                 owns_peer: bool):
        self.rendezvous, self.peer, self.addrs = rendezvous, peer, addrs
        self.window, self.owns_peer = window, owns_peer
        self.pipe: channel.Pipe | None = None
        self.ready = False
        self.closing = False
        self.info: dict = {}
        self.receipt: dict | None = None
        self.sign_pk: bytes | None = None
        self.transcript = bytes(32)
        self._turn = asyncio.Lock()
        self._resuming: asyncio.Task | None = None
        self._progress = (-1, 0)  # frames read at the last loss, and losses since then
        self.links_lost = 0
        self.contact: dict = {}
        self.stopped = False  # the last answer was cut off, not finished

    @classmethod
    async def open(cls, rendezvous: Rendezvous, enc_pk: bytes | None = None, *, peer: Peer | None = None,
                   addrs: list[str] | None = None, session_id: bytes | None = None,
                   sign_pk: bytes | None = None, mode: str = "constant",
                   tick: float = channel.TICK, bucket: float = channel.DURATION_BUCKET,
                   frame: int = channel.FRAME, window: float = 30.0) -> "Session":
        """Link to the seller and open the sealed channel.

        `addrs` are this buyer's candidates, for instance from echo.probe().
        `sign_pk` pins the receipt key; otherwise the first one seen is used.
        """
        owns_peer = peer is None
        peer = peer or await Peer.bind()
        self = cls(rendezvous, peer, addrs if addrs is not None else peer.addrs, window, owns_peer)
        link = await self._dial()
        hello = bc.canonical_json({"v": 1, "mode": mode, "tick": tick, "bucket": bucket})
        ch, first = channel.connect(enc_pk or bytes.fromhex(self.contact["enc_pk"]),
                                    session_id or os.urandom(channel.SESSION_ID_LEN), hello, frame)
        self.pipe = channel.Pipe(ch, on_lost=self._lost)  # requests are sized by tier, not paced
        await link.write(first)
        self.pipe.attach(link)
        self.info = await asyncio.wait_for(self.pipe.recv(), window)
        if self.info.get("t") != "ready":
            raise GatewayError("no_session", str(self.info))
        if self.info["tier"] not in UNPROVEN_TIERS:
            raise GatewayError("unproven_tier", f"{self.info['tier']} was claimed without evidence")
        self.sign_pk = sign_pk or bytes.fromhex(self.info["sign_pk"])
        self.ready = True
        return self

    @property
    def tier(self) -> str:
        return self.info["tier"]

    async def _dial(self):
        await self.rendezvous.publish({"id": self.peer.id, "addrs": self.addrs})
        self.contact = await self.rendezvous.wait(self.window)
        return await self.peer.link(self.contact["id"], self.contact["addrs"], initiator=True,
                                    window=self.window)

    def _lost(self, exc) -> None:
        if self.closing:
            return
        if not self.ready:
            self.pipe.fail(SessionLost(f"link lost while opening: {exc}"))
            return
        self.links_lost += 1
        read, losses = self._progress
        losses = losses + 1 if read == self.pipe.ch.rx else 1
        self._progress = (self.pipe.ch.rx, losses)
        if losses > RESUME_TRIES:  # links come up and die with nothing getting through
            self.pipe.fail(SessionLost(f"no progress after {losses} links: {exc}"))
            return
        self._resuming = asyncio.create_task(self._resume())

    async def _resume(self) -> None:
        error, deadline = None, time.monotonic() + RESUME_BUDGET_S
        for _ in range(RESUME_TRIES):
            try:
                link = await self._dial()
                await link.write(self.pipe.ch.resume_frame())
                reply = await asyncio.wait_for(link.read_exact(self.pipe.ch.frame), self.window)
                self.pipe.attach(link, self.pipe.ch.read_resume(reply))
                return
            except Exception as e:
                error = e
                if time.monotonic() > deadline:
                    break
                await asyncio.sleep(1.0)
        self.pipe.fail(SessionLost(f"could not resume: {error!r}"))

    def _check(self, receipt: dict, request: bytes, text: str) -> None:
        if not bc.verify(self.sign_pk, bytes.fromhex(receipt["sig"]), receipt_digest(receipt)):
            raise ReceiptError("signature does not verify")
        self.transcript = next_transcript(self.transcript, request, text)
        if bytes.fromhex(receipt["transcript_hash"]) != self.transcript:
            raise ReceiptError("transcript hash differs from what was seen here")
        if receipt["channel_id"] != self.pipe.ch.session_id.hex():
            raise ReceiptError("receipt is for another session")
        last = self.receipt or {"seq": 0, "in_tokens": 0, "out_tokens": 0}
        if (receipt["seq"] != last["seq"] + 1 or receipt["in_tokens"] < last["in_tokens"]
                or receipt["out_tokens"] < last["out_tokens"]):
            raise ReceiptError("usage went backwards")
        self.receipt = receipt

    async def _expect(self, kind: str) -> dict:
        msg = await self.pipe.recv()
        if msg.get("t") == "error":
            raise GatewayError(msg["code"], msg["detail"])
        if msg.get("t") != kind:
            raise GatewayError("out_of_order", f"wanted {kind}, got {msg.get('t')}")
        return msg

    async def chat(self, messages: list[dict], *, model: str, **params):
        """Yield the answer piece by piece. Extra params go to the backend as they are.

        Read it to the end: the gateway finishes a response even if nobody is
        listening, and the next call would otherwise receive the rest of this one.
        """
        async with self._turn:
            request = self.pipe.send({"t": "chat", "model": model, "messages": messages, **params},
                                     tiered=True)
            pieces = []
            while True:
                msg = await self.pipe.recv()
                kind = msg.get("t")
                if kind == "delta":
                    pieces.append(msg["text"])
                    yield msg["text"]
                elif kind == "done":
                    self._check(msg["receipt"], request, "".join(pieces))
                    self.stopped = msg.get("stopped", False)
                    return
                elif kind == "error":
                    raise GatewayError(msg["code"], msg["detail"])
                else:
                    raise GatewayError("out_of_order", str(kind))

    async def upload_adapter(self, path: Path) -> str:
        """Hand a PEFT adapter directory to this session; returns its model name.

        The ciphertext goes over the link in the clear; the key goes through the
        sealed channel. A dropped link resumes the upload from the last good chunk.
        """
        files = {p.name: p.read_bytes() for p in Path(path).iterdir()
                 if p.is_file() and p.name in ADAPTER_FILES}
        return await self.upload_adapter_files(files)

    async def upload_adapter_files(self, files: dict[str, bytes]) -> str:
        """The same from memory, so a decrypted adapter never touches this disk."""
        if not files or not set(files) <= set(ADAPTER_FILES):
            raise ValueError(f"an adapter is {ADAPTER_FILES}")
        key = bc.new_key()
        blob = transfer.seal_tiered(key, beetle_archive.tar_bytes(files),
                                    adapter_aad(self.pipe.ch.session_id), transfer.ADAPTER_TIERS)
        with tempfile.TemporaryDirectory() as tmp:
            sealed = Path(tmp) / "adapter.enc"
            sealed.write_bytes(blob)
            del blob
            m = await asyncio.to_thread(transfer.manifest, sealed)
            root = transfer.root_of(m)
            async with self._turn:
                self.pipe.send({"t": "adapter", "root": root.hex(), "key": key.hex()}, tiered=True)
                sending = asyncio.create_task(self._send_file(sealed, root, m))
                try:  # a refusal comes at once, and nobody reads the file then
                    return (await self._expect("adapter_ok"))["name"]
                finally:
                    sending.cancel()

    async def _send_file(self, sealed: Path, root: bytes, m: dict) -> None:
        error = None
        for _ in range(RESUME_TRIES * 2):
            await asyncio.wait_for(self.pipe.attached.wait(), self.window * RESUME_TRIES)
            try:
                stream = await self.pipe.link.open_stream()
                await stream.write(root)
                await transfer.send(stream, sealed, m)
                return
            except Exception as e:  # the pipe notices the same loss and reconnects
                error = e
                await asyncio.sleep(1.0)
        raise SessionLost(f"upload could not finish: {error}")

    async def close(self) -> None:
        """End the session. The gateway unloads and deletes this session's adapters."""
        async with self._turn:
            self.pipe.send({"t": "close"}, tiered=True)
            try:
                msg = await asyncio.wait_for(self._expect("closed"), self.window)
                self.receipt = msg["receipt"]
            finally:
                self.closing = True
                if self._resuming:
                    self._resuming.cancel()
                link = self.pipe.link
                self.pipe.detach()
                if link:
                    await link.close()
                if self.owns_peer:
                    await self.peer.close()
