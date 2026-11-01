"""Attester tests: registration rules, using real hardware evidence where we have it.

No machine here can produce TDX or App Attest evidence bound to one of our
jobs, so those tests check that the attester rejects evidence that doesn't
bind the job. The accept path is covered for MOCK here and for TPM2 against
a software TPM in tests/tpm_interop_test.py and the end-to-end run.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "attester")]

import beetle_crypto as bc  # noqa: E402
from attester import Attester, Registration, Rejected  # noqa: E402
from server import dev_policy  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


@pytest.fixture
def attester(tmp_path):
    return Attester(tmp_path / "attester.key", dev_policy(None), tmp_path / "state.json")


def mock_evidence(job_id, spec_hash, measurement_hex):
    """What the Docker sandbox writes: a self-signed attestation bound to the job."""
    sk = bc.new_signing_key()
    sign_pk, enc_pk = bc.public_bytes(sk), os.urandom(32)
    att = {"tier": "MOCK", "mock": True, "measurement": measurement_hex, "job_id": job_id.hex(),
           "spec_hash": spec_hash.hex(), "sign_pk": sign_pk.hex(), "enc_pk": enc_pk.hex(),
           "report_data": bc.digest("start", job_id + spec_hash + sign_pk + enc_pk).hex(),
           "network": {"isolated": True}}
    sig = sk.sign(bc.digest("attest", bc.canonical_json(att))).hex()
    return sign_pk, enc_pk, {"type": "MOCK", "attestation": att, "signature": sig}


def code_measurement():
    return dev_policy(None)["mock"]["allowed_measurements"][0]


def test_registration_signed_and_round_trips(attester):
    job_id, spec_hash, host = os.urandom(32), os.urandom(32), os.urandom(32)
    sign_pk, enc_pk, ev = mock_evidence(job_id, spec_hash, code_measurement())
    reg = attester.register(job_id, spec_hash, sign_pk, enc_pk, host, ev)
    assert reg.tier == 0 and reg.measurement.hex() == code_measurement()
    again = Registration.from_json(json.loads(json.dumps(reg.to_json())))
    assert bc.verify(attester.public_key, again.signature, again.message())


def test_rejects_evidence_for_other_keys(attester):
    job_id, spec_hash = os.urandom(32), os.urandom(32)
    _, _, ev = mock_evidence(job_id, spec_hash, code_measurement())
    with pytest.raises(Rejected, match="bind"):
        attester.register(job_id, spec_hash, os.urandom(32), os.urandom(32), os.urandom(32), ev)


def test_rejects_evidence_for_other_spec(attester):
    job_id, spec_hash = os.urandom(32), os.urandom(32)
    sign_pk, enc_pk, ev = mock_evidence(job_id, spec_hash, code_measurement())
    with pytest.raises(Rejected, match="bind"):
        attester.register(job_id, os.urandom(32), sign_pk, enc_pk, os.urandom(32), ev)


def test_rejects_unknown_image(attester):
    job_id, spec_hash = os.urandom(32), os.urandom(32)
    sign_pk, enc_pk, ev = mock_evidence(job_id, spec_hash, os.urandom(32).hex())
    with pytest.raises(Rejected, match="allowlist"):
        attester.register(job_id, spec_hash, sign_pk, enc_pk, os.urandom(32), ev)


def test_rejects_mock_when_policy_forbids(tmp_path):
    policy = dev_policy(None)
    policy["mock"]["allow"] = False
    a = Attester(tmp_path / "k", policy, tmp_path / "s.json")
    job_id, spec_hash = os.urandom(32), os.urandom(32)
    sign_pk, enc_pk, ev = mock_evidence(job_id, spec_hash, code_measurement())
    with pytest.raises(Rejected, match="not accepted"):
        a.register(job_id, spec_hash, sign_pk, enc_pk, os.urandom(32), ev)


def test_rejects_real_tdx_quote_not_bound_to_job(attester):
    attester.policy["tdx"]["allowed_tcb_status"] = ["UpToDate", "OutOfDate"]
    ev = {"type": "TDX",
          "quote": base64.b64encode((FIX / "tdx" / "tdx_quote").read_bytes()).decode(),
          "collateral": (FIX / "tdx" / "tdx_quote_collateral.json").read_text()}
    with pytest.raises(Rejected, match="TDX"):
        attester.register(os.urandom(32), os.urandom(32), os.urandom(32), os.urandom(32),
                          os.urandom(32), ev)


def test_rejects_app_attest_from_unenrolled_key(attester):
    ev = {"type": "APP_ATTEST", "key_id": "00" * 32, "assertion": base64.b64encode(b"x").decode()}
    with pytest.raises(Rejected, match="not enrolled"):
        attester.register(os.urandom(32), os.urandom(32), os.urandom(32), os.urandom(32),
                          os.urandom(32), ev)


def test_app_attest_enrollment_checks_app_identity(attester):
    """The real iOS fixture is for another developer's app, so our policy rejects it."""
    fx = next(f for f in json.loads((FIX / "appattest" / "ios-14.4.json").read_text())
              if f["type"] == "attestation")
    b = lambda s: base64.b64decode(s + "=" * (-len(s) % 4))  # noqa: E731
    with pytest.raises(Rejected, match="App Attest"):
        attester.app_attest_enroll(b(fx["keyIdBase64"]), b(fx["attestationBase64"]),
                                   b(fx["clientDataBase64"]))


def test_rejects_unknown_evidence_type(attester):
    with pytest.raises(Rejected, match="unknown"):
        attester.register(os.urandom(32), os.urandom(32), os.urandom(32), os.urandom(32),
                          os.urandom(32), {"type": "SELF_REPORTED"})


def test_digests_are_bound_to_one_network(attester):
    """Evidence signed for one program/cluster is rejected by an attester serving another.
    (The end-to-end run caught an attester that hadn't been configured for its network.)"""
    saved = (bc.PROGRAM_ID, bc.CLUSTER)
    try:
        job_id, spec_hash = os.urandom(32), os.urandom(32)
        bc.configure(b"A" * 32, b"devnet")
        sign_pk, enc_pk, ev = mock_evidence(job_id, spec_hash, code_measurement())
        bc.configure(b"B" * 32, b"devnet")
        with pytest.raises(Rejected, match="mock signature"):
            attester.register(job_id, spec_hash, sign_pk, enc_pk, os.urandom(32), ev)
    finally:
        bc.configure(*saved)
