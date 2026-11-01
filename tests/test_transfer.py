"""File transfer tests: hashes, tiers and resume, over an in-memory link."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import net  # noqa: E402,F401

import beetle_crypto as bc  # noqa: E402
from net import transfer  # noqa: E402
from net.transfer import ADAPTER_TIERS, CHUNK, FILE_TIERS, TransferError, manifest, root_of  # noqa: E402
from test_net_channel import mem_pair  # noqa: E402


def write_file(path: Path, size: int) -> dict:
    path.write_bytes(os.urandom(size))
    return manifest(path)


def transfer_once(src: Path, dest: Path, root: bytes, **kw):
    async def go():
        left, right = mem_pair()
        sender = asyncio.create_task(transfer.send(left, src))
        try:
            return await asyncio.wait_for(transfer.receive(right, root, dest, **kw), 10)
        finally:
            sender.cancel()
    return asyncio.run(go())


@pytest.mark.parametrize("size", [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK + 7])
def test_files_arrive_whole(tmp_path, size):
    src, dest = tmp_path / "src", tmp_path / "dest"
    m = write_file(src, size)
    assert transfer_once(src, dest, root_of(m)) == dest
    assert dest.read_bytes() == src.read_bytes()
    assert not (tmp_path / "dest.part").exists()


def test_wrong_root_is_refused(tmp_path):
    src = tmp_path / "src"
    write_file(src, 100)
    with pytest.raises(TransferError, match="root"):
        transfer_once(src, tmp_path / "dest", bytes(32))


def test_over_the_limit_is_refused(tmp_path):
    src = tmp_path / "src"
    m = write_file(src, 2 * CHUNK)
    with pytest.raises(TransferError, match="limit"):
        transfer_once(src, tmp_path / "dest", root_of(m), max_size=CHUNK)


class Counting:
    """Passes a link through, counting CHUNK_GET requests and corrupting chunks on demand."""

    def __init__(self, link, corrupt=0):
        self.link, self.corrupt, self.chunks = link, corrupt, 0

    async def read_exact(self, n):
        return await self.link.read_exact(n)

    async def write(self, data):
        if len(data) == CHUNK:
            self.chunks += 1
            if self.corrupt > 0:
                self.corrupt -= 1
                data = bytes([data[0] ^ 1]) + data[1:]
        await self.link.write(data)


def test_only_missing_chunks_are_fetched_after_an_interruption(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    m = write_file(src, 8 * CHUNK)
    whole = src.read_bytes()
    part = tmp_path / "dest.part"
    part.write_bytes(whole[:5 * CHUNK] + b"torn write")

    async def go():
        left, right = mem_pair()
        sender = Counting(left)
        task = asyncio.create_task(transfer.send(sender, src))
        await asyncio.wait_for(transfer.receive(right, root_of(m), dest), 10)
        await task
        return sender.chunks

    assert asyncio.run(go()) == 3
    assert dest.read_bytes() == whole


def test_a_link_cut_midway_resumes_on_the_next_one(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    m = write_file(src, 40 * CHUNK)

    class Cuts(Counting):
        async def write(self, data):
            await super().write(data)
            if self.chunks == 20:
                self.link.cut()

    async def go():
        left, right = mem_pair()
        sender = asyncio.create_task(transfer.send(Cuts(left), src))
        with pytest.raises(Exception):
            await asyncio.wait_for(transfer.receive(right, root_of(m), dest), 10)
        sender.cancel()
        left, right = mem_pair()
        counted = Counting(left)
        task = asyncio.create_task(transfer.send(counted, src))
        await asyncio.wait_for(transfer.receive(right, root_of(m), dest), 10)
        await task
        return counted.chunks

    resent = asyncio.run(go())
    assert resent <= 20 + transfer.BATCH
    assert dest.read_bytes() == src.read_bytes()


def test_a_corrupt_chunk_is_fetched_again(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    m = write_file(src, 3 * CHUNK)

    async def go():
        left, right = mem_pair()
        task = asyncio.create_task(transfer.send(Counting(left, corrupt=1), src))
        await asyncio.wait_for(transfer.receive(right, root_of(m), dest), 10)
        await task

    asyncio.run(go())
    assert dest.read_bytes() == src.read_bytes()


def test_a_sender_that_keeps_corrupting_is_dropped(tmp_path):
    src = tmp_path / "src"
    m = write_file(src, CHUNK)

    async def go():
        left, right = mem_pair()
        task = asyncio.create_task(transfer.send(Counting(left, corrupt=99), src))
        with pytest.raises(TransferError, match="hash"):
            await asyncio.wait_for(transfer.receive(right, root_of(m), tmp_path / "d"), 10)
        task.cancel()

    asyncio.run(go())


def test_a_request_past_the_end_stops_the_sender(tmp_path):
    src = tmp_path / "src"
    write_file(src, CHUNK)

    async def go():
        left, right = mem_pair()
        await right.write(bytes([transfer.CHUNK_GET]) + (5).to_bytes(4, "big") + (1).to_bytes(4, "big"))
        with pytest.raises(TransferError, match="past the end"):
            await transfer.send(left, src)

    asyncio.run(go())


def test_tiers_hide_the_exact_size():
    key = bc.new_key()
    small = transfer.seal_tiered(key, b"a" * 1000, b"aad", FILE_TIERS)
    large = transfer.seal_tiered(key, b"a" * 900_000, b"aad", FILE_TIERS)
    assert len(small) == len(large) == (1 << 20) + 28
    assert bc.open_blob(key, small, b"aad") == b"a" * 1000


def test_adapter_tiers_and_limits():
    assert transfer.tier(44 << 20, ADAPTER_TIERS) == 128 << 20
    with pytest.raises(ValueError):
        transfer.tier(600 << 20, ADAPTER_TIERS)
