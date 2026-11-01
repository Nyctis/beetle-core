"""Attestation verifier tests against evidence produced by real hardware.

TDX: a quote from real Intel TDX hardware (Phala's dcap-qvl samples).
App Attest: an attestation and an assertion from a real iOS 14.4 device
(veehaitch/devicecheck-appattest test resources). See tests/fixtures/*/NOTICE.

Run from the beetle directory:  uv run --with pytest pytest -q tests/
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common")]

import attestation as att  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
TDX_QUOTE = (FIX / "tdx" / "tdx_quote").read_bytes()
TDX_COLLATERAL = (FIX / "tdx" / "tdx_quote_collateral.json").read_text()
TDX_VALID_AT = int(datetime(2025, 6, 20, tzinfo=timezone.utc).timestamp())  # inside every window


def b64(s: str) -> bytes:
    s = s.strip()
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)) if "-" in s or "_" in s \
        else base64.b64decode(s + "=" * (-len(s) % 4))


# --- TDX ---------------------------------------------------------------------------


def test_tdx_real_quote_verifies():
    v = att.verify_tdx(TDX_QUOTE, TDX_COLLATERAL, TDX_VALID_AT, expected_report_data=None)
    assert v.tier == att.TIER_T2 and v.claims["tcb_status"] == "UpToDate"
    assert len(v.measurement) == 32


def test_tdx_rejects_quote_not_bound_to_this_job():
    with pytest.raises(att.AttestationError, match="report_data"):
        att.verify_tdx(TDX_QUOTE, TDX_COLLATERAL, TDX_VALID_AT, expected_report_data=bytes(32))


# MRTD, REPORTDATA, the quote's ECDSA signature, and the QE report deep in the signature data
@pytest.mark.parametrize("offset", [48 + 136, 48 + 520, 636, 636 + 128 + 100])
def test_tdx_rejects_tampered_quote(offset):
    q = bytearray(TDX_QUOTE)
    q[offset] ^= 0x01
    with pytest.raises(att.AttestationError):
        att.verify_tdx(bytes(q), TDX_COLLATERAL, TDX_VALID_AT, expected_report_data=None)


def test_tdx_padding_rules():
    end = att.trim_tdx_quote(TDX_QUOTE)
    assert len(end) == 4936 and not any(TDX_QUOTE[4936:])  # the sample carries zero padding
    att.verify_tdx(end + bytes(100), TDX_COLLATERAL, TDX_VALID_AT, None)  # more zeros: fine
    with pytest.raises(att.AttestationError, match="unsigned bytes"):
        att.verify_tdx(end + b"\x01", TDX_COLLATERAL, TDX_VALID_AT, None)


def test_tdx_rejects_expired_collateral():
    now = int(datetime(2026, 10, 28, tzinfo=timezone.utc).timestamp())
    with pytest.raises(att.AttestationError, match="rejected"):
        att.verify_tdx(TDX_QUOTE, TDX_COLLATERAL, now, expected_report_data=None)


def test_tdx_rejects_disallowed_tcb_status():
    with pytest.raises(att.AttestationError, match="status"):
        att.verify_tdx(TDX_QUOTE, TDX_COLLATERAL, TDX_VALID_AT, None, allowed_status=())


# --- Apple App Attest ----------------------------------------------------------------


FIXTURES = json.loads((FIX / "appattest" / "ios-14.4.json").read_text())
ATT = next(f for f in FIXTURES if f["type"] == "attestation")
ASR = next(f for f in FIXTURES if f["type"] == "assertion")
AT = datetime.fromisoformat(ATT["timestamp"].replace("Z", "+00:00"))
TEAM, BUNDLE = ATT["teamIdentifier"], ATT["bundleIdentifier"]


def attest(**kw):
    args = dict(attestation_object=b64(ATT["attestationBase64"]), key_id=b64(ATT["keyIdBase64"]),
                client_data=b64(ATT["clientDataBase64"]), team_id=TEAM, bundle_id=BUNDLE,
                production=False, at=AT)
    args.update(kw)
    return att.verify_app_attest_key(**args)


def test_app_attest_real_attestation_verifies():
    pub, v = attest()
    expected = serialization.load_pem_public_key(ATT["publicKey"].encode())
    assert pub.public_numbers() == expected.public_numbers()
    assert v.tier == att.TIER_T1 and v.claims["environment"] == "development"


@pytest.mark.parametrize("change,match", [
    (dict(client_data=b"another challenge"), "nonce"),
    (dict(team_id="AAAAAAAAAA"), "app id"),
    (dict(production=True), "environment"),
    (dict(key_id=bytes(32)), "key id"),
    (dict(at=datetime(2026, 10, 28, tzinfo=timezone.utc)), "not valid"),
])
def test_app_attest_rejects(change, match):
    with pytest.raises(att.AttestationError, match=match):
        attest(**change)


def test_app_attest_rejects_tampered_auth_data():
    obj = att.cbor2.loads(b64(ATT["attestationBase64"]))
    auth = bytearray(obj["authData"])
    auth[0] ^= 0x01  # rpIdHash: the nonce no longer matches the certificate
    obj["authData"] = bytes(auth)
    with pytest.raises(att.AttestationError, match="nonce"):
        attest(attestation_object=att.cbor2.dumps(obj))


def test_app_attest_real_assertion_and_replay():
    pub, _ = attest()
    assertion, client_data = b64(ASR["assertionBase64"]), b64(ASR["clientDataBase64"])
    assert att.verify_app_attest_assertion(assertion, client_data, pub, TEAM, BUNDLE, 0) == 1
    with pytest.raises(att.AttestationError, match="replay"):
        att.verify_app_attest_assertion(assertion, client_data, pub, TEAM, BUNDLE, 1)
    with pytest.raises(att.AttestationError, match="signature"):
        att.verify_app_attest_assertion(assertion, client_data + b"x", pub, TEAM, BUNDLE, 0)
