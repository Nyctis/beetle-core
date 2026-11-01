"""Crypto helpers shared by the requester, the host and the sandbox.

Everything here wraps audited libraries (pyca/cryptography, pyhpke). Nothing
implements a primitive by hand. Formats:

- HPKE: RFC 9180, DHKEM(X25519, HKDF-SHA256) + HKDF-SHA256 + ChaCha20-Poly1305.
  A sealed 32-byte key is enc (32 B) || ciphertext (48 B) = 80 bytes.
- Bulk data: AES-256-GCM, nonce (12 B) prefixed, AAD binds job id and purpose.
- Every sealed blob is length-prefixed and zero-padded to a size bucket first,
  so its ciphertext size reveals only a coarse size class.
- Signed messages: SHA256("beetle/v0/" || type || program_id || cluster || body).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pyhpke import AEADId, CipherSuite, KDFId, KEMId

DOMAIN = b"beetle/v0/"
# Offline runs (host/run_local.py) use these placeholders. Chain runs call
# configure() with the deployed program id so digests match the program's.
PROGRAM_ID = b"Beetle111111111111111111111111111111111111"
CLUSTER = b"local"

SMALL_BUCKET = 64 * 1024          # data, code, spec body, private report
ADAPTER_BUCKET = 4 * 1024 * 1024  # adapter archives
# What a sandbox's outputs are padded to: a size that says a step count or a
# data size would say too much. The adapter tiers are net/transfer.py's.
ADAPTER_TIERS = tuple(m << 20 for m in (32, 128, 512))
REPORT_BUCKET = 1 << 20
SEALED_KEY_LEN = 80

_SUITE = CipherSuite.new(
    KEMId.DHKEM_X25519_HKDF_SHA256, KDFId.HKDF_SHA256, AEADId.CHACHA20_POLY1305
)


def configure(program_id: bytes, cluster: bytes) -> None:
    global PROGRAM_ID, CLUSTER
    PROGRAM_ID, CLUSTER = program_id, cluster


def configure_from_spec(spec: dict) -> None:
    chain = spec.get("chain")
    if chain:
        configure(bytes.fromhex(chain["program_id_hex"]), chain["cluster"].encode())


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(type_name: str, body: bytes) -> bytes:
    """Domain-separated digest that signatures are made over."""
    return sha256(DOMAIN + type_name.encode() + PROGRAM_ID + CLUSTER + body)


def settle_message(job_id: bytes, report: dict, report_hash: bytes) -> bytes:
    """Compact outcome the program re-derives and checks via Ed25519 precompile.

    Only the outcome, the share of the work done and hashes go on chain; scores
    stay in the sealed report. The terms the sandbox ran under are signed too:
    the program compares them with the job. Must match Chain.settle_message in
    beetle_chain.py.
    """
    sealed = bytes.fromhex(report["sealed_adapter_key"]) if report["sealed_adapter_key"] else bytes(SEALED_KEY_LEN)
    body = (job_id + bytes([report["outcome"]]) + report["work_bps"].to_bytes(2, "little")
            + bytes.fromhex(report["sealed_dek_hash"]) + bytes.fromhex(report["adapter_ct_hash"])
            + bytes([report["sealed_adapter_key"] is not None]) + sealed + report_hash
            + report["units"].to_bytes(8, "little") + bytes([report["price_mode"]])
            + report["run_secs"].to_bytes(8, "little") + report["grace_s"].to_bytes(4, "little"))
    return digest("settle", body)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_manifest_sha256(root: Path) -> str:
    """Hash of the sorted (relative path, file hash) list of a model directory."""
    entries = sorted(
        (p.relative_to(root).as_posix(), file_sha256(p))
        for p in root.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    return sha256(canonical_json(entries)).hex()


def code_measurement(files: dict[str, bytes]) -> str:
    """Mock measurement: hash of the sandbox source files by basename.

    A real TDX/App Attest measurement comes from hardware. Here the host can
    recompute the same value from the public source, which is the property a
    reproducible build gives the real thing.
    """
    entries = sorted((name, sha256(content).hex()) for name, content in files.items())
    return sha256(canonical_json(entries)).hex()


# --- Size padding ------------------------------------------------------------


def pad(data: bytes, bucket: int) -> bytes:
    """8-byte length prefix, then zeros up to a multiple of `bucket`."""
    body = len(data).to_bytes(8, "big") + data
    return body + bytes(-len(body) % bucket)


def unpad(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:8], "big")
    if n > len(blob) - 8 or any(blob[8 + n:]):
        raise ValueError("invalid padding")
    return blob[8:8 + n]


def adapter_bucket(size: int) -> int:
    return next((t for t in ADAPTER_TIERS if size + 8 <= t), ADAPTER_TIERS[-1])


# --- HPKE (key envelopes and sealed messages) --------------------------------


class HpkeKeyPair:
    """One-time X25519 key pair; it lives only in this process's memory."""

    def __init__(self):
        self._kp = _SUITE.kem.derive_key_pair(os.urandom(32))

    @property
    def public_bytes(self) -> bytes:
        return self._kp.public_key.to_public_bytes()

    def open(self, sealed: bytes, info: bytes) -> bytes:
        enc, ct = sealed[:32], sealed[32:]
        ctx = _SUITE.create_recipient_context(enc, self._kp.private_key, info=info)
        return ctx.open(ct)

    def open_blob(self, sealed: bytes, info: bytes) -> bytes:
        return unpad(self.open(sealed, info))


def hpke_seal(recipient_pk: bytes, plaintext: bytes, info: bytes) -> bytes:
    pk = _SUITE.kem.deserialize_public_key(recipient_pk)
    enc, ctx = _SUITE.create_sender_context(pk, info=info)
    return enc + ctx.seal(plaintext)


def hpke_seal_blob(recipient_pk: bytes, data: bytes, info: bytes,
                   bucket: int = SMALL_BUCKET) -> bytes:
    return hpke_seal(recipient_pk, pad(data, bucket), info)


# --- Bulk encryption -------------------------------------------------------


def aead_encrypt(key: bytes, plaintext: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def aead_decrypt(key: bytes, blob: bytes, aad: bytes) -> bytes:
    return AESGCM(key).decrypt(blob[:12], blob[12:], aad)


def seal_blob(key: bytes, data: bytes, aad: bytes, bucket: int = SMALL_BUCKET) -> bytes:
    return aead_encrypt(key, pad(data, bucket), aad)


def open_blob(key: bytes, blob: bytes, aad: bytes) -> bytes:
    return unpad(aead_decrypt(key, blob, aad))


def new_key() -> bytes:
    return AESGCM.generate_key(bit_length=256)


def hmac_sha256(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


# --- Signatures --------------------------------------------------------------


def new_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def public_bytes(sk: Ed25519PrivateKey) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return sk.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def verify(pk: bytes, sig: bytes, msg: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(pk).verify(sig, msg)
        return True
    except Exception:
        return False
