"""Checks hardware evidence against a policy. This is the actual trust decision.

The chain doesn't verify evidence. A buyer runs this locally before releasing
the data key (requester.py); a third-party attester (attester.py) runs the same
code before signing. A policy is plain JSON and rejects by default:

    {"mock":       {"allow": false, "allowed_measurements": []},
     "tdx":        {"allowed_tcb_status": ["UpToDate"], "allowed_measurements": ["<hex>"]},
     "app_attest": {"team_id": "...", "bundle_id": "...", "production": true,
                    "require_sip_policy": true, "allowed_measurements": []},
     "tpm":        {"trusted_ek_ca_files": ["ca.pem"], "expected_pcrs": {"23": "<hex>"},
                    "allowed_measurements": ["<hex>"]}}

A missing section or an empty allowlist accepts nothing; there is no "skip".
MOCK proves nothing and passes only if the policy explicitly allows it, and
never on mainnet (requester.py).

Devices that need a one-time enrollment (App Attest keys, TPM attestation
keys) are kept in a small JSON state file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common")]

import attestation as att  # noqa: E402
import tpm_attest as tpm  # noqa: E402
from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402


class Rejected(Exception):
    pass


def b64e(b: bytes) -> str:
    return base64.b64encode(b).decode()


def b64d(s: str) -> bytes:
    return base64.b64decode(s)


class EvidenceVerifier:
    def __init__(self, policy: dict, state_path: Path):
        self.policy = policy
        self.state_path = Path(state_path)
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() \
            else {"app_attest": {}, "tpm": {}, "tpm_pending": {}}

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1))

    # --- one-time device enrollment ---------------------------------------------

    def app_attest_enroll(self, key_id: bytes, attestation_object: bytes, challenge: bytes) -> str:
        """`challenge` must be one this service issued; a real deployment tracks
        and expires them. Returns the device id (hex key id)."""
        p = self.policy["app_attest"]
        try:
            pub, _ = att.verify_app_attest_key(attestation_object, key_id, challenge, p["team_id"],
                                               p["bundle_id"], p["production"])
        except att.AttestationError as e:
            raise Rejected(f"App Attest enrollment: {e}")
        pem = pub.public_bytes(serialization.Encoding.PEM,
                               serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        self.state["app_attest"][key_id.hex()] = {"public_key": pem, "counter": 0}
        self._save()
        return key_id.hex()

    def tpm_enroll_begin(self, ek_cert: bytes, ak_public: bytes) -> tuple[str, bytes]:
        """Returns (enrollment id, credential blob for tpm2_activatecredential)."""
        cas = [x509.load_pem_x509_certificate(Path(f).read_bytes())
               for f in self.policy["tpm"]["trusted_ek_ca_files"]]
        try:
            ek_pub = tpm.verify_ek_cert(ek_cert, cas)
            ak = tpm.parse_ak_public(ak_public)
        except tpm.TpmError as e:
            raise Rejected(f"TPM enrollment: {e}")
        secret = os.urandom(32)
        enrollment_id = hashlib.sha256(ak.name).hexdigest()[:32]
        self.state["tpm_pending"][enrollment_id] = {
            "ak_public": ak_public.hex(), "secret_sha256": hashlib.sha256(secret).hexdigest()}
        self._save()
        return enrollment_id, tpm.make_credential(ek_pub, ak.name, secret)

    def tpm_enroll_finish(self, enrollment_id: str, activated_secret: bytes) -> None:
        pending = self.state["tpm_pending"].pop(enrollment_id, None)
        if pending is None:
            raise Rejected("unknown TPM enrollment")
        if hashlib.sha256(activated_secret).hexdigest() != pending["secret_sha256"]:
            self._save()
            raise Rejected("TPM did not activate the credential: AK is not in this TPM")
        self.state["tpm"][enrollment_id] = {"ak_public": pending["ak_public"]}
        self._save()

    # --- per-job verification ---------------------------------------------------------

    def verify(self, evidence: dict, report_data: bytes) -> att.Verdict:
        """Accepts only evidence that verifies for its type, binds exactly
        `report_data`, and shows an allowed measurement. Raises Rejected."""
        verdict = self._verify(evidence, report_data)
        if verdict.report_data != report_data:
            raise Rejected("evidence does not bind this job's sandbox keys")
        allowed = {bytes.fromhex(m) for m in self._allowlist(verdict.evidence_type)}
        if verdict.measurement not in allowed:
            raise Rejected(f"{verdict.evidence_type} measurement {verdict.measurement.hex()[:16]}... "
                           "is not on the allowlist")
        return verdict

    def _allowlist(self, evidence_type: str) -> list[str]:
        return {"MOCK": self.policy.get("mock", {}).get("allowed_measurements", []),
                "TDX": self.policy.get("tdx", {}).get("allowed_measurements", []),
                "APP_ATTEST": self.policy.get("app_attest", {}).get("allowed_measurements", []),
                "TPM2": self.policy.get("tpm", {}).get("allowed_measurements", [])}[evidence_type]

    def _verify(self, ev: dict, report_data: bytes) -> att.Verdict:
        kind = ev.get("type")
        try:
            if kind == "MOCK":
                if not self.policy.get("mock", {}).get("allow"):
                    raise Rejected("mock evidence is not accepted by this policy")
                return att.verify_mock(ev["attestation"], ev["signature"])
            if kind == "TDX":
                p = self.policy["tdx"]
                return att.verify_tdx(b64d(ev["quote"]), ev["collateral"], int(time.time()),
                                      report_data, tuple(p["allowed_tcb_status"]))
            if kind == "APP_ATTEST":
                p = self.policy["app_attest"]
                dev = self.state["app_attest"].get(ev["key_id"])
                if dev is None:
                    raise Rejected("App Attest key not enrolled")
                pub = serialization.load_pem_public_key(dev["public_key"].encode())
                counter, verdict = att.verify_app_attest_job(
                    b64d(ev["assertion"]), report_data, pub, p["team_id"], p["bundle_id"],
                    dev["counter"])
                if p.get("require_sip_policy") and not verdict.claims["sip_policy_checked"]:
                    raise Rejected("policy requires the macOS 27 SIP key policy, not verified yet")
                dev["counter"] = counter
                self._save()
                return verdict
            if kind == "TPM2":
                dev = self.state["tpm"].get(ev["enrollment_id"])
                if dev is None:
                    raise Rejected("TPM not enrolled")
                ak = tpm.parse_ak_public(bytes.fromhex(dev["ak_public"]))
                expected = {int(i): bytes.fromhex(v)
                            for i, v in self.policy["tpm"]["expected_pcrs"].items()}
                pcrs = {int(i): bytes.fromhex(v) for i, v in ev["pcrs"].items()}
                tpm.verify_quote(b64d(ev["quote"]), b64d(ev["signature"]), ak, pcrs,
                                 report_data, expected)
                # The image is identified by PCR 23, which only our locked boot extends.
                return att.Verdict("TPM2", att.TIER_T3B, pcrs[23], report_data,
                                   {"enrollment_id": ev["enrollment_id"]})
        except (att.AttestationError, tpm.TpmError) as e:
            raise Rejected(f"{kind} evidence: {e}")
        except KeyError as e:
            raise Rejected(f"{kind} evidence or policy is missing {e}")
        raise Rejected(f"unknown evidence type {kind!r}")
