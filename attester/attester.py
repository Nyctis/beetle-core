"""The optional third-party verifier: signs a registration the chain checks.

Buyers normally verify evidence themselves (verifier.py) and name no verifier.
A buyer who would rather trust someone else names that party's key in
`open_job`; `start` then fails without this registration (Ed25519 precompile).
The attester signs only after:

1. the evidence verifies for its type (TDX quote, App Attest assertion, TPM
   quote, or the development mock if the policy allows it);
2. the evidence's report_data equals SHA256("beetle/v0/start" || program ||
   cluster || job_id || spec_hash || sign_pk || enc_pk), i.e. it binds exactly
   this job and these sandbox keys;
3. the attested measurement is on the policy's allowlist.

Registration message (re-derived by the program from the job's own fields):
    digest("register", job_id || spec_hash || sign_pk || enc_pk || measurement
                       || tier (u8) || expiry (i64 LE) || host (32 bytes))
"""

from __future__ import annotations

import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "attester")]

import beetle_crypto as bc  # noqa: E402
from verifier import EvidenceVerifier, Rejected, b64d, b64e  # noqa: E402,F401


def registration_message(job_id: bytes, spec_hash: bytes, sign_pk: bytes, enc_pk: bytes,
                         measurement: bytes, tier: int, expiry: int, host: bytes) -> bytes:
    return bc.digest("register", job_id + spec_hash + sign_pk + enc_pk + measurement
                     + bytes([tier]) + struct.pack("<q", expiry) + host)


@dataclass
class Registration:
    job_id: bytes
    spec_hash: bytes
    sign_pk: bytes
    enc_pk: bytes
    measurement: bytes
    tier: int
    expiry: int
    host: bytes
    attester_pk: bytes
    signature: bytes
    claims: dict

    def message(self) -> bytes:
        return registration_message(self.job_id, self.spec_hash, self.sign_pk, self.enc_pk,
                                    self.measurement, self.tier, self.expiry, self.host)

    def to_json(self) -> dict:
        return {k: (v.hex() if isinstance(v, bytes) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_json(cls, d: dict) -> "Registration":
        hexed = {"job_id", "spec_hash", "sign_pk", "enc_pk", "measurement", "host", "attester_pk",
                 "signature"}
        return cls(**{k: (bytes.fromhex(v) if k in hexed else v) for k, v in d.items()})


class Attester(EvidenceVerifier):
    def __init__(self, key_path: Path, policy: dict, state_path: Path):
        super().__init__(policy, state_path)
        key_path = Path(key_path)
        if not key_path.exists():
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_bytes(os.urandom(32))
            key_path.chmod(0o600)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        self.key = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
        self.public_key = bc.public_bytes(self.key)

    def register(self, job_id: bytes, spec_hash: bytes, sign_pk: bytes, enc_pk: bytes,
                 host: bytes, evidence: dict) -> Registration:
        report_data = bc.digest("start", job_id + spec_hash + sign_pk + enc_pk)
        verdict = self.verify(evidence, report_data)
        expiry = int(time.time()) + int(self.policy["registration_ttl_seconds"])
        msg = registration_message(job_id, spec_hash, sign_pk, enc_pk, verdict.measurement,
                                   verdict.tier, expiry, host)
        return Registration(job_id, spec_hash, sign_pk, enc_pk, verdict.measurement, verdict.tier,
                            expiry, host, self.public_key, self.key.sign(msg),
                            {"evidence_type": verdict.evidence_type, **verdict.claims})
