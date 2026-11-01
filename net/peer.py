"""Direct QUIC links between two peers, built on iroh with relays switched off.

Nothing here talks to a coordination server. Each side learns the other's
EndpointId and candidate addresses from the rendezvous (on chain in the real
flow), then both dial at once: each side's QUIC Initial opens its own NAT, so
the other side's Initial gets through. Whichever connection forms first wins,
and the initiator opens the first stream on it, so the two sides never have to
agree on a connection ahead of time.

The link is the outer layer only. It ends at the host, which sees ciphertext;
what travels over it is sealed to the sandbox (see net/channel.py).
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import iroh

ALPN = b"beetle/net/1"
HELLO = b"beetle/hello/1\n"
REDIAL_EVERY = 10.0  # seconds; a live connection can still turn out to be stale
EARLY_HOLD = 30.0    # how long a connection may wait for its link() call


class Link:
    """One QUIC connection and one of its streams."""

    def __init__(self, conn, send, recv):
        self.conn, self.send, self.recv = conn, send, recv

    async def write(self, data: bytes) -> None:
        await self.send.write_all(data)

    async def read_exact(self, n: int) -> bytes:
        return await self.recv.read_exact(n)

    def rtt_ms(self) -> int | None:
        return self.conn.rtt()  # None when no path is selected

    async def open_stream(self) -> "Link":
        """Another stream on this connection. The far side sees it on first write."""
        bi = await self.conn.open_bi()
        return Link(self.conn, bi.send(), bi.recv())

    async def accept_stream(self) -> "Link":
        bi = await self.conn.accept_bi()
        return Link(self.conn, bi.send(), bi.recv())

    async def close(self) -> None:
        self.conn.close(0, b"done")


async def _open_hello(conn) -> Link:
    link = await Link(conn, None, None).open_stream()
    await link.write(HELLO)  # a stream is invisible until it carries data
    await link.read_exact(len(HELLO))
    return link


async def _answer_hello(conn) -> Link:
    link = await Link(conn, None, None).accept_stream()
    await link.read_exact(len(HELLO))
    await link.write(HELLO)
    return link


class Peer:
    def __init__(self, endpoint, services: dict[bytes, Callable[..., Awaitable[None]]]):
        self.endpoint = endpoint
        self.mapped: str | None = None  # how a NAT shows this port, if anyone asked
        self.services = services        # other ALPNs on this endpoint: alpn -> handler(conn)
        self._links: dict[str, asyncio.Queue] = {}  # peer id -> connections for its link()
        self._early: dict[str, list] = {}           # connections ahead of their link()
        self._tasks: set[asyncio.Task] = set()
        self._spawn(self._accept())

    @classmethod
    async def bind(cls, port: int = 0, secret: bytes | None = None,
                   services: dict | None = None) -> "Peer":
        iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
        services = services or {}
        return cls(await iroh.Endpoint.bind(iroh.EndpointOptions(
            preset=iroh.preset_minimal(),  # the default preset calls n0's servers
            relay_mode=iroh.RelayMode.disabled(),
            secret_key=secret,
            alpns=[ALPN, *services],
            bind_addr=f"0.0.0.0:{port}",
        )), services)

    @property
    def id(self) -> str:
        return str(self.endpoint.id())

    @property
    def port(self) -> int:
        return int(self.endpoint.bound_sockets()[0].rsplit(":", 1)[1])

    @property
    def addrs(self) -> list[str]:
        """This machine's own interface addresses, which work on a shared LAN."""
        return list(self.endpoint.addr().direct_addresses())

    async def close(self) -> None:
        await self.endpoint.close()

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _accept(self) -> None:
        while (incoming := await self.endpoint.accept_next()) is not None:
            self._spawn(self._admit(incoming))

    async def _admit(self, incoming) -> None:
        try:
            accepting = await incoming.accept()
            alpn = await accepting.alpn()
            conn = await accepting.connect()
        except Exception:  # a failed handshake belongs to whoever dialed
            return
        if alpn in self.services:
            await self.services[alpn](conn)
            return
        who = str(conn.remote_id())
        if who in self._links:
            self._links[who].put_nowait(conn)
        else:  # it may simply have beaten our own link() call
            self._early.setdefault(who, []).append(conn)
            asyncio.get_running_loop().call_later(EARLY_HOLD, self._evict, who, conn)

    def _evict(self, who: str, conn) -> None:
        held = self._early.get(who, [])
        if conn in held:
            held.remove(conn)
            conn.close(1, b"wrong peer")

    async def link(self, peer_id: str, candidates: list[str], initiator: bool,
                   window: float = 30.0) -> Link:
        """Dial and accept at once until a stream is up, or raise TimeoutError."""
        addr = iroh.EndpointAddr(iroh.EndpointId.from_string(peer_id), None, candidates)
        ready: asyncio.Queue = asyncio.Queue()
        for conn in self._early.pop(peer_id, []):
            ready.put_nowait(conn)
        self._links[peer_id] = ready

        async def dial():
            while True:
                try:
                    ready.put_nowait(await self.endpoint.connect(addr, ALPN))
                    await asyncio.sleep(REDIAL_EVERY)
                except Exception:  # a timed-out dial still holds the NAT open
                    await asyncio.sleep(0.5)

        async def open_stream() -> Link:
            while True:
                conn = await ready.get()
                try:
                    return await _open_hello(conn)
                except Exception:  # left over from an earlier link; try the next
                    conn.close(2, b"stale")

        async def wait_stream() -> Link:
            waiting: dict[asyncio.Task, object] = {}
            getter = asyncio.create_task(ready.get())
            try:
                while True:
                    done, _ = await asyncio.wait({*waiting, getter},
                                                 return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        if task is getter:
                            conn = task.result()
                            waiting[asyncio.create_task(_answer_hello(conn))] = conn
                            getter = asyncio.create_task(ready.get())
                        else:
                            waiting.pop(task)
                            if task.exception() is None:
                                return task.result()
            finally:
                getter.cancel()
                for task, conn in waiting.items():
                    task.cancel()
                    conn.close(2, b"dup")

        dialer = asyncio.create_task(dial())
        try:
            return await asyncio.wait_for(open_stream() if initiator else wait_stream(), window)
        finally:
            dialer.cancel()
            if self._links.get(peer_id) is ready:
                del self._links[peer_id]
            while not ready.empty():
                ready.get_nowait().close(2, b"dup")
