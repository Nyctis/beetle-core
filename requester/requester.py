"""Requester side (the user's agent): prepare an encrypted job, release the
data key only to a sandbox whose hardware evidence verifies here, on this
machine, then verify the signed report and open the adapter.

The requester never trusts the host: everything it accepts is either signed
by the attested sandbox key or checked against a hash it computed itself.

What the host can read (spec.json, the public header): which public base
model to fetch, the resource budget, the chain ids, the requester's
encryption public key and hashes of the inputs. Everything else (recipe,
acceptance thresholds, training code, data, hidden eval set) is sealed with
the per-job data key and padded to a size bucket.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "common"), str(ROOT / "attester")]

import canary
import beetle_crypto as bc
from attestation import AttestationError, Verdict, verify_mock
from beetle_archive import read_tar, tar_bytes
from job_defaults import TIERS
from net import transfer
from verifier import EvidenceVerifier, Rejected

JOB_INPUTS = ("body", "train", "eval", "program")  # each stored as <name>.enc
NO_EVIDENCE_TIER = TIERS["t3c"]
MAX_PROGRAM_BYTES = 1024 * 1024


class AttestationRejected(Exception):
    pass


def load_or_create_storage_key(path: Path) -> bytes:
    """Long-term key that encrypts adapters at rest on the requester's device."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bc.new_key())
        path.chmod(0o600)
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError(f"{path} is not a 32-byte key")
    return key


def bundle_program(program_dir: Path | None) -> bytes:
    if program_dir is None:
        return tar_bytes({})
    files = {p.name: p.read_bytes() for p in sorted(Path(program_dir).iterdir())
             if p.is_file() and not p.name.startswith(".")}
    blob = tar_bytes(files)
    if len(blob) > MAX_PROGRAM_BYTES:
        raise ValueError("program bundle larger than 1 MiB")
    return blob


def hpke_from_seed(seed: bytes) -> bc.HpkeKeyPair:
    """The same one-time key pair, so a job can outlive this process."""
    pair = bc.HpkeKeyPair.__new__(bc.HpkeKeyPair)
    pair._kp = bc._SUITE.kem.derive_key_pair(seed)
    return pair


class Requester:
    def __init__(self, storage_key: bytes, wallet_pk: bytes | None = None):
        # On chain the job id is derived from the requester's Solana wallet
        # and a recent slot; offline runs use a throwaway key and slot 0.
        self.wallet_pk = wallet_pk or bc.public_bytes(bc.new_signing_key())
        self.storage_key = storage_key
        self.enc_seed = os.urandom(32)
        self.enc_key = hpke_from_seed(self.enc_seed)  # one-time key that receives the results
        self.dek = bc.new_key()                       # one-time data key for this job

    def secrets(self) -> dict:
        """What a restarted process needs. Seal it; it opens everything in this job."""
        return {"enc_seed": self.enc_seed.hex(), "dek": self.dek.hex(), "job_id": self.job_id.hex(),
                "spec_hash": self.spec_hash.hex(), "nonce": self.nonce.hex(),
                "open_slot": self.open_slot}

    @classmethod
    def restore(cls, storage_key: bytes, wallet_pk: bytes, secrets: dict) -> "Requester":
        self = cls(storage_key, wallet_pk)
        self.enc_seed = bytes.fromhex(secrets["enc_seed"])
        self.enc_key = hpke_from_seed(self.enc_seed)
        self.dek = bytes.fromhex(secrets["dek"])
        self.job_id = bytes.fromhex(secrets["job_id"])
        self.spec_hash = bytes.fromhex(secrets["spec_hash"])
        self.nonce = bytes.fromhex(secrets["nonce"])
        self.open_slot = secrets["open_slot"]
        return self

    def prepare_job(self, job_dir: Path, memories: list, eval_items: list, base_model: dict,
                    recipe: dict, acceptance: dict, budget: dict, chain: dict | None = None,
                    program_dir: Path | None = None, n_canaries: int = 3,
                    open_slot: int = 0) -> dict:
        nonce = os.urandom(32)
        job_id = bc.sha256(self.wallet_pk + nonce + struct.pack("<Q", open_slot))
        self.canaries = canary.make(self.storage_key, job_id, n_canaries)
        program = bundle_program(program_dir)
        body = {
            "mode": "program" if program_dir else "recipe",
            "recipe": recipe,
            "acceptance": acceptance,
            "program": {"entry": "train.py", "bundle_sha256": bc.sha256(program).hex()}
            if program_dir else None,
            "parent_adapter": None,
        }
        plaintexts = {
            "body": bc.canonical_json(body),
            "train": bc.canonical_json(memories + self.canaries),
            "eval": bc.canonical_json(eval_items),
            "program": program,  # an empty archive in recipe mode, so both modes look alike
        }
        inbox = job_dir / "in"
        inbox.mkdir(parents=True, exist_ok=True)
        # One size tier for all four, so the host can't compare their sizes.
        bucket = max(transfer.tier(len(p) + 8) for p in plaintexts.values())
        hashes, roots = {}, {}
        for name in JOB_INPUTS:
            path = inbox / f"{name}.enc"
            path.write_bytes(bc.seal_blob(self.dek, plaintexts[name], job_id + name.encode(), bucket))
            manifest = transfer.manifest(path)
            hashes[name], roots[name] = manifest["sha256"], transfer.root_of(manifest).hex()

        spec = {  # the public header
            "version": "beetle/v0",
            "base_model": base_model,
            "budget": budget,
            "inputs": hashes,
            "input_roots": roots,  # names each file for net/transfer; bound by spec_hash
            "requester_enc_pk": self.enc_key.public_bytes.hex(),
            "nonce": nonce.hex(),
            "open_slot": open_slot,
        }
        if chain:
            spec["chain"] = chain
        bc.configure_from_spec(spec)
        spec_hash = bc.sha256(bc.canonical_json(spec))
        (inbox / "spec.json").write_text(json.dumps(spec, indent=2))
        job = {"job_id": job_id.hex(), "spec_hash": spec_hash.hex()}
        (inbox / "job.json").write_text(json.dumps(job, indent=2))
        self.job_id, self.spec_hash, self.nonce, self.open_slot = job_id, spec_hash, nonce, open_slot
        return job

    def check_attestation(self, att: dict, sig_hex: str, expected_measurement: str) -> None:
        """Release the key only if every check passes. Raises otherwise."""
        sign_pk = bytes.fromhex(att["sign_pk"])
        enc_pk = bytes.fromhex(att["enc_pk"])
        checks = {
            "signature": bc.verify(sign_pk, bytes.fromhex(sig_hex),
                                   bc.digest("attest", bc.canonical_json(att))),
            "job_id": att["job_id"] == self.job_id.hex(),
            "spec_hash": att["spec_hash"] == self.spec_hash.hex(),
            "measurement": att["measurement"] == expected_measurement,
            "report_data": att["report_data"] == bc.digest(
                "start", self.job_id + self.spec_hash + sign_pk + enc_pk).hex(),
            "network_isolated": att["network"]["isolated"] is True,
        }
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            raise AttestationRejected(f"attestation checks failed: {failed}")
        self.attested_sign_pk = sign_pk
        self.attested_enc_pk = enc_pk

    def check_evidence(self, verifier: EvidenceVerifier, evidence: dict, job, mainnet: bool):
        """The check that decides whether the data key is released; nothing else
        stands behind it. Takes the keys, measurement and tier from the job as
        started on chain, so what passes here is what the DEK gets sealed to.
        Raises AttestationRejected; the caller then uses reject_evidence."""
        if bc.sha256(bc.canonical_json(evidence)) != job.evidence_hash:
            raise AttestationRejected("evidence differs from the hash committed on chain")
        if evidence.get("type") == "MOCK" and mainnet:
            raise AttestationRejected("mock evidence is never valid on mainnet")
        report_data = bc.digest("start", self.job_id + self.spec_hash + job.sign_pk + job.enc_pk)
        if job.tier == NO_EVIDENCE_TIER:
            return self._nothing_to_verify(evidence, job, report_data)
        try:
            verdict = verifier.verify(evidence, report_data)
        except Rejected as e:
            raise AttestationRejected(str(e))
        if verdict.measurement != job.measurement:
            raise AttestationRejected("evidence shows a different measurement than the job")
        if verdict.tier != job.tier:
            raise AttestationRejected(f"evidence is tier {verdict.tier}, the offer claims {job.tier}")
        self.attested_sign_pk, self.attested_enc_pk = job.sign_pk, job.enc_pk
        return verdict

    def _nothing_to_verify(self, evidence: dict, job, report_data: bytes):
        """T3-C offers no evidence, and the floor is what lets a job get here. All
        that can be checked is that the keys on chain are the ones this claims."""
        if evidence.get("type") != "NONE":
            raise AttestationRejected("a T3-C job carries no evidence of any kind")
        try:
            claim = verify_mock(evidence["attestation"], evidence["signature"])
        except (AttestationError, KeyError, ValueError) as e:
            raise AttestationRejected(f"the sandbox's own claim does not hold: {e}")
        if claim.report_data != report_data or claim.measurement != job.measurement:
            raise AttestationRejected("the sandbox's own claim is not about this job")
        self.attested_sign_pk, self.attested_enc_pk = job.sign_pk, job.enc_pk
        return Verdict("NONE", NO_EVIDENCE_TIER, claim.measurement, report_data, {})

    def seal_dek(self, wrong_key: bool = False) -> bytes:
        key = bc.new_key() if wrong_key else self.dek
        return bc.hpke_seal(self.attested_enc_pk, key, info=b"dek" + self.job_id)

    def verify_report(self, report: dict, sig_hex: str, adapter_ct: bytes,
                      private_ct: bytes, expected_measurement: str) -> dict:
        return {
            "signature_from_attested_key": bc.verify(
                self.attested_sign_pk, bytes.fromhex(sig_hex),
                bc.digest("report", bc.canonical_json(report))),
            "job_id": report["job_id"] == self.job_id.hex(),
            "spec_hash": report["spec_hash"] == self.spec_hash.hex(),
            "measurement": report["measurement"] == expected_measurement,
            "adapter_ct_hash": report["adapter_ct_hash"] == bc.sha256(adapter_ct).hex(),
            "private_report_hash": report["private_report_ct_hash"] == bc.sha256(private_ct).hex(),
        }

    def open_private_report(self, private_ct: bytes) -> dict:
        return json.loads(self.enc_key.open_blob(private_ct, info=b"private-report" + self.job_id))

    def open_adapter(self, report: dict, adapter_ct: bytes, job_dir: Path) -> list[str]:
        """Decrypt the adapter, check it, and store it re-encrypted at rest."""
        key = self.enc_key.open(bytes.fromhex(report["sealed_adapter_key"]),
                                info=b"adapter" + self.job_id)
        bundle = bc.open_blob(key, adapter_ct, self.job_id + b"adapter")
        if bc.sha256(bundle).hex() != report["adapter_pt_hash"]:
            raise ValueError("adapter plaintext hash mismatch")
        names = sorted(read_tar(bundle))
        stored = bc.seal_blob(self.storage_key, bundle, b"adapter-store" + self.job_id,
                              bc.ADAPTER_BUCKET)
        (job_dir / "adapter.store").write_bytes(stored)
        return names


def load_stored_adapter(storage_key: bytes, path: Path, job_id: bytes) -> dict[str, bytes]:
    """Decrypt a stored adapter into memory; nothing plaintext touches disk."""
    bundle = bc.open_blob(storage_key, Path(path).read_bytes(), b"adapter-store" + job_id)
    return read_tar(bundle)
