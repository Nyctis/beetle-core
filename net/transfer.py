"""Large files over a Link: chunks, hashes and resume.

The receiver pulls. A file is named by its root, the hash of a manifest listing
every chunk's SHA-256, so each chunk is checked as it arrives and a transfer cut
off halfway picks up with only the chunks still missing.

Protects against corruption and substitution in transit, and against losing
work to a dropped link. Keeps nothing secret: files must be sealed before they
get here, and seal_tiered() pads them to a size tier first so the ciphertext
shows only the tier. Whoever carries a file still learns the tier, the timing
and the hash. The receiver must learn the root from somewhere the sender's host
cannot touch; a root taken from the sender proves nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import struct
from pathlib import Path

import beetle_crypto as bc

CHUNK = 1 << 20
BATCH = 16          # chunks per request
MAX_STRIKES = 3     # bad copies of one chunk before giving up on the sender
MAX_MANIFEST = 16 << 20

FILE_TIERS = tuple(m << 20 for m in (1, 16, 64, 256, 1024))
ADAPTER_TIERS = tuple(m << 20 for m in (32, 128, 512))

MANIFEST_GET, CHUNK_GET, DONE = 0x02, 0x03, 0x06


class TransferError(Exception):
    pass


def tier(size: int, tiers: tuple[int, ...] = FILE_TIERS) -> int:
    for t in tiers:
        if size <= t:
            return t
    raise ValueError(f"{size} bytes is over the largest tier ({tiers[-1]})")


def seal_tiered(key: bytes, data: bytes, aad: bytes,
                tiers: tuple[int, ...] = FILE_TIERS) -> bytes:
    """Seal `data` so every file in a tier has the same ciphertext size."""
    return bc.seal_blob(key, data, aad, bucket=tier(len(data) + 8, tiers))


def manifest(path: Path, chunk_size: int = CHUNK) -> dict:
    whole, leaves, size = hashlib.sha256(), [], 0
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            whole.update(block)
            leaves.append(bc.sha256(block).hex())
            size += len(block)
    return {"size": size, "chunk_size": chunk_size, "sha256": whole.hexdigest(), "leaves": leaves}


def root_of(m: dict) -> bytes:
    return bc.sha256(bc.canonical_json(m))


async def send(link, path: Path, m: dict | None = None) -> None:
    """Serve one file until the receiver says it is done."""
    m = m or await asyncio.to_thread(manifest, path)
    body = bc.canonical_json(m)
    with open(path, "rb") as f:
        while True:
            op = (await link.read_exact(1))[0]
            if op == MANIFEST_GET:
                await link.write(struct.pack(">I", len(body)) + body)
            elif op == CHUNK_GET:
                index, count = struct.unpack(">II", await link.read_exact(8))
                if index + count > len(m["leaves"]):
                    raise TransferError("request runs past the end of the file")
                f.seek(index * m["chunk_size"])
                for _ in range(count):
                    await link.write(f.read(m["chunk_size"]))
            elif op == DONE:
                return
            else:
                raise TransferError(f"unknown request {op}")


def _missing(part: Path, m: dict) -> list[int]:
    """Chunks not yet held. Re-hashing beats trusting a progress file after a crash."""
    if not part.exists():
        return list(range(len(m["leaves"])))
    missing = []
    with open(part, "rb") as f:
        for index, leaf in enumerate(m["leaves"]):
            if bc.sha256(f.read(m["chunk_size"])).hex() != leaf:
                missing.append(index)
    return missing


async def receive(link, root: bytes, dest: Path, max_size: int = 1 << 30) -> Path:
    """Fetch the file named `root` into `dest`, resuming from `dest`.part if present."""
    await link.write(bytes([MANIFEST_GET]))
    n = struct.unpack(">I", await link.read_exact(4))[0]
    if n > MAX_MANIFEST:
        raise TransferError("manifest too large")
    body = await link.read_exact(n)
    if bc.sha256(body) != root:
        raise TransferError("manifest does not match the root")
    m = json.loads(body)
    size, chunk = m["size"], m["chunk_size"]
    if size > max_size:
        raise TransferError(f"{size} bytes is over the limit of {max_size}")

    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    missing = await asyncio.to_thread(_missing, part, m)
    strikes: dict[int, int] = {}
    with open(part, "ab+") as f:  # create if absent, keep what is there
        f.truncate(size)
    with open(part, "r+b") as f:
        while missing:
            count = 1
            while count < min(BATCH, len(missing)) and missing[count] == missing[0] + count:
                count += 1
            batch, missing = missing[:count], missing[count:]
            await link.write(struct.pack(">BII", CHUNK_GET, batch[0], count))
            for index in batch:
                block = await link.read_exact(min(chunk, size - index * chunk))
                if bc.sha256(block).hex() == m["leaves"][index]:
                    f.seek(index * chunk)
                    f.write(block)
                    continue
                strikes[index] = strikes.get(index, 0) + 1
                if strikes[index] >= MAX_STRIKES:
                    raise TransferError(f"chunk {index} failed its hash {MAX_STRIKES} times")
                missing.append(index)

    if await asyncio.to_thread(bc.file_sha256, part) != m["sha256"]:
        part.unlink()
        raise TransferError("file does not match its hash")
    os.replace(part, dest)
    await link.write(bytes([DONE]))
    return dest
