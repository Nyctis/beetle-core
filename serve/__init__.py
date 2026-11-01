"""Private inference over P2P: the seller's gateway and the buyer's client.

What both ends must agree on lives here: which tiers can be claimed without
evidence, and how a UsageReceipt and its transcript hash are computed.
"""

from __future__ import annotations

import net  # noqa: F401  (puts common/ on the import path)

import beetle_crypto as bc

# A tier comes from evidence, not from what a seller says. Nothing here checks
# evidence yet, so these are the only tiers either end accepts.
UNPROVEN_TIERS = {
    "T3": "no privacy promise",
    "T3-C": "ordinary cloud machine: the provider and anyone with root can read this session",
}
DEFAULT_TIER = "T3-C"
RECEIPT_FIELDS = ("channel_id", "seq", "in_tokens", "out_tokens", "transcript_hash")
# Exactly these: a pickled adapter_model.bin would be code on the seller's machine.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
# What a buyer may ask of the backend. Everything else stays here: `n` multiplies
# the work, and a server's own extensions are not this gateway's to pass along.
PASSED_PARAMS = ("max_tokens", "temperature", "top_p", "seed", "stop", "frequency_penalty",
                 "presence_penalty")
MAX_RESPONSE_TOKENS = 4096
BYTES_PER_TOKEN = 4       # a low count for English; only used before a server has reported
TOKEN_TOLERANCE = 1.5     # how far two honest counts of the same text can sit apart
TOKEN_SLACK = 64          # and the template's tokens, which neither side sees as text


def estimate_tokens(data: bytes | str) -> int:
    """Without a tokenizer. Good enough to meter a stream, not to bill one."""
    size = len(data.encode() if isinstance(data, str) else data)
    return max(1, -(-size // BYTES_PER_TOKEN))


def counts_agree(claimed: int, counted: int) -> bool:
    """False when `claimed` is more than an honest difference above `counted`."""
    return claimed <= counted * TOKEN_TOLERANCE + TOKEN_SLACK


def receipt_digest(receipt: dict) -> bytes:
    return bc.digest("usage_receipt", bc.canonical_json({k: receipt[k] for k in RECEIPT_FIELDS}))


def next_transcript(previous: bytes, request: bytes, response: str) -> bytes:
    """Hash chain over the session. Both ends compute it from what they hold."""
    return bc.sha256(b"beetle/v1/transcript\0" + previous
                     + bc.sha256(request) + bc.sha256(response.encode()))


def adapter_aad(session_id: bytes) -> bytes:
    return b"beetle/v1/adapter\0" + session_id
