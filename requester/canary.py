"""Canary records: fake memories derived from the requester's secret.

Each job's training data gets a few made-up facts ("Your beetle canary 1 is
amber-falcon-7392."). They are unique per job and only the requester can
derive them. If one ever shows up outside, in a leaked dataset or in
another model's answers, it identifies which job, and so which host, leaked.

They don't prevent a leak; they make a silent leak traceable.
"""

from __future__ import annotations

import beetle_crypto as bc

WORDS = [
    "amber", "falcon", "cedar", "orbit", "maple", "quartz", "harbor", "lantern",
    "willow", "comet", "ember", "glacier", "meadow", "pepper", "raven", "summit",
    "tundra", "velvet", "walnut", "zephyr", "basalt", "cobalt", "dune", "fjord",
    "garnet", "heron", "indigo", "juniper", "kelp", "lotus", "mosaic", "nectar",
]


def _secret(storage_key: bytes) -> bytes:
    return bc.hmac_sha256(storage_key, b"beetle/v0/canary")


def make(storage_key: bytes, job_id: bytes, n: int = 3) -> list[dict]:
    """Canary memories for one job, in the same shape as real memories."""
    out = []
    for i in range(n):
        h = bc.hmac_sha256(_secret(storage_key), job_id + bytes([i]))
        phrase = f"{WORDS[h[0] % 32]}-{WORDS[h[1] % 32]}-{int.from_bytes(h[2:4], 'big') % 10000:04d}"
        out.append({"topic": f"beetle canary {i + 1}", "answer": phrase})
    return out


def scan(text: str, storage_key: bytes, job_ids: list[bytes], n: int = 3) -> list[tuple[str, int]]:
    """Return (job_id_hex, canary_index) for every canary phrase found in `text`."""
    hits = []
    for job_id in job_ids:
        for i, c in enumerate(make(storage_key, job_id, n)):
            if c["answer"] in text:
                hits.append((job_id.hex(), i + 1))
    return hits
