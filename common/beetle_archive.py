"""Deterministic flat tar archives, shared by the requester, the gateway and the sandbox."""

from __future__ import annotations

import io
import tarfile

MAX_FILES = 64
MAX_BYTES = 8 << 30


def tar_bytes(files: dict[str, bytes]) -> bytes:
    """Sorted names, zero mtime, regular files only, so equal input gives equal bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size = len(files[name])
            info.mtime = 0
            tar.addfile(info, io.BytesIO(files[name]))
    return buf.getvalue()


def read_tar(blob: bytes, max_bytes: int = MAX_BYTES, max_files: int = MAX_FILES) -> dict[str, bytes]:
    """Read a flat tar into memory, refusing paths, links, special files and
    anything that would expand. Archives here come from the other side, so no
    compression and no sparse members, and the sizes are checked before reading."""
    out, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as tar:
        for m in tar:
            if not m.isfile() or m.issparse() or "/" in m.name or m.name.startswith("."):
                raise ValueError(f"unexpected entry in archive: {m.name}")
            total += m.size
            if total > max_bytes or len(out) >= max_files:
                raise ValueError("archive too large")
            out[m.name] = tar.extractfile(m).read()
    return out
