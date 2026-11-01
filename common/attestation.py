"""Verifying hardware evidence that a sandbox key belongs to a real, unmodified sandbox.

One entry point per evidence type. Each returns a Verdict or raises
AttestationError; nothing here trusts the host. What each type can prove:

| type        | tier | proves                                                     | keeps data from the owner |
|-------------|------|------------------------------------------------------------|---------------------------|
| TDX         | T2   | CPU-signed: this exact VM image runs, memory is encrypted  | yes (not physical attacks)|
| APP_ATTEST  | T1   | Apple-signed: our app, on a genuine Apple device           | only from a curious owner |
| TPM2        | T3B  | TPM-signed: this machine booted our locked image           | no                        |
| MOCK        | MOCK | nothing; self-signed by the sandbox, for development only  | no                        |

Every verdict carries `report_data`, the 32-byte digest that binds the
evidence to one job and one pair of sandbox keys (see enclave.py). The buyer
and the attester re-derive it, so evidence for one job can't be reused for
another. The chain stores only the evidence's hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cbor2
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

TIER_MOCK, TIER_T1, TIER_T2, TIER_T3B = 0, 1, 2, 3
TIER_NAMES = {TIER_MOCK: "MOCK", TIER_T1: "T1 Mac", TIER_T2: "T2 TDX", TIER_T3B: "T3-B TPM"}

APPLE_ROOT = Path(__file__).with_name("certs") / "Apple_App_Attestation_Root_CA.pem"
APPLE_NONCE_OID = x509.ObjectIdentifier("1.2.840.113635.100.8.2")
AAGUID_PRODUCTION = b"appattest" + bytes(7)
AAGUID_DEVELOPMENT = b"appattestdevelop"


class AttestationError(Exception):
    pass


@dataclass
class Verdict:
    evidence_type: str
    tier: int
    measurement: bytes          # 32 bytes: what code/identity is attested
    report_data: bytes          # 32 bytes: the job binding
    claims: dict = field(default_factory=dict)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AttestationError(msg)


# --- Intel TDX -----------------------------------------------------------------


def tdx_measurement(report) -> bytes:
    """MRTD (initial image) plus the four runtime measurement registers."""
    parts = [bytes(report.mr_td), bytes(report.rt_mr0), bytes(report.rt_mr1),
             bytes(report.rt_mr2), bytes(report.rt_mr3)]
    return hashlib.sha256(b"".join(parts)).digest()


TDX_SIGNED_END = 48 + 584 + 4  # v4 quote: header, TD report, then u32 signature-data length


def trim_tdx_quote(quote: bytes) -> bytes:
    """The quote is header + TD report + length-prefixed signature data. Some
    producers pad it with zeros; anything after the signature data is not
    covered by the signature, so allow only zero padding and drop it."""
    _require(len(quote) >= TDX_SIGNED_END, "quote too short")
    end = TDX_SIGNED_END + int.from_bytes(quote[TDX_SIGNED_END - 4:TDX_SIGNED_END], "little")
    _require(len(quote) >= end, "quote truncated")
    _require(not any(quote[end:]), "unsigned bytes after the quote")
    return quote[:end]


def verify_tdx(quote: bytes, collateral_json: str, now: int, expected_report_data: bytes | None,
               allowed_status=("UpToDate",)) -> Verdict:
    """Verify a TDX quote against Intel's root of trust (via Phala's dcap-qvl).

    `collateral_json` is Intel's signed TCB info, QE identity and CRLs for this
    platform (fetched from Intel PCS or a PCCS cache). `now` must fall inside
    all of their validity windows.
    """
    import dcap_qvl

    quote = trim_tdx_quote(quote)
    collateral = dcap_qvl.QuoteCollateralV3.from_json(collateral_json)
    try:
        verified = dcap_qvl.verify(quote, collateral, now)
    except Exception as e:
        raise AttestationError(f"TDX quote rejected: {e}")
    _require(verified.status in allowed_status, f"TDX TCB status {verified.status} not allowed")
    parsed = dcap_qvl.parse_quote(quote)
    _require(parsed.is_tdx, "not a TDX quote")
    report = parsed.report
    td_attributes = bytes(report.td_attributes)
    _require(td_attributes[0] & 0x01 == 0, "TD runs in debug mode")
    rd = bytes(report.report_data)
    # We put the 32-byte job binding first and zeros after it.
    if expected_report_data is not None:
        _require(rd[:32] == expected_report_data and rd[32:] == bytes(32),
                 "report_data does not bind this job's sandbox keys")
    return Verdict("TDX", TIER_T2, tdx_measurement(report), rd[:32], {
        "quote_sha256": hashlib.sha256(quote).hexdigest(), "tcb_status": verified.status, "advisory_ids": list(verified.advisory_ids),
        "mr_td": bytes(report.mr_td).hex(), "rt_mr3": bytes(report.rt_mr3).hex()})


# --- Apple App Attest ------------------------------------------------------------


def _verify_chain(leaf: x509.Certificate, intermediate: x509.Certificate,
                  root: x509.Certificate, at: datetime) -> None:
    for cert, issuer in ((leaf, intermediate), (intermediate, root)):
        _require(cert.issuer == issuer.subject, "certificate chain names do not link")
        try:
            issuer.public_key().verify(cert.signature, cert.tbs_certificate_bytes,
                                       ec.ECDSA(cert.signature_hash_algorithm))
        except InvalidSignature:
            raise AttestationError("certificate signature invalid")
        _require(cert.not_valid_before_utc <= at <= cert.not_valid_after_utc,
                 f"certificate not valid at {at.isoformat()}")
    bc = intermediate.extensions.get_extension_for_class(x509.BasicConstraints).value
    _require(bc.ca, "intermediate is not a CA")


def _apple_nonce(cert: x509.Certificate) -> bytes:
    """The extension value is DER: SEQUENCE { [1] EXPLICIT OCTET STRING (32 bytes) }."""
    der = cert.extensions.get_extension_for_oid(APPLE_NONCE_OID).value.value
    _require(len(der) == 38 and der[0] == 0x30 and der[2] == 0xA1 and der[4] == 0x04
             and der[5] == 32, "unexpected nonce extension encoding")
    return der[6:38]


def _rp_id_hash(team_id: str, bundle_id: str) -> bytes:
    return hashlib.sha256(f"{team_id}.{bundle_id}".encode()).digest()


def verify_app_attest_key(attestation_object: bytes, key_id: bytes, client_data: bytes,
                          team_id: str, bundle_id: str, production: bool,
                          at: datetime | None = None) -> tuple[ec.EllipticCurvePublicKey, Verdict]:
    """One-time attestation of a device's App Attest key (Apple's 9 server checks).

    Returns the attested public key; per-job proofs are assertions signed by it.
    """
    at = at or datetime.now(timezone.utc)
    obj = cbor2.loads(attestation_object)
    _require(obj.get("fmt") == "apple-appattest", "not an App Attest attestation")
    x5c = obj["attStmt"]["x5c"]
    _require(len(x5c) == 2, "expected leaf and intermediate certificates")
    leaf, inter = (x509.load_der_x509_certificate(c) for c in x5c)
    root = x509.load_pem_x509_certificate(APPLE_ROOT.read_bytes())
    _verify_chain(leaf, inter, root, at)                                     # 1

    auth = obj["authData"]
    nonce = hashlib.sha256(auth + hashlib.sha256(client_data).digest()).digest()  # 2-3
    _require(_apple_nonce(leaf) == nonce, "nonce does not match client data")     # 4
    pub = leaf.public_key()
    point = pub.public_bytes(serialization.Encoding.X962,
                             serialization.PublicFormat.UncompressedPoint)
    _require(hashlib.sha256(point).digest() == key_id, "key id is not the attested key")  # 5
    _require(auth[:32] == _rp_id_hash(team_id, bundle_id), "app id does not match")       # 6
    _require(int.from_bytes(auth[33:37], "big") == 0, "counter must be 0")                 # 7
    aaguid = auth[37:53]
    _require(aaguid == (AAGUID_PRODUCTION if production else AAGUID_DEVELOPMENT),
             "wrong App Attest environment")                                               # 8
    cred_len = int.from_bytes(auth[53:55], "big")
    _require(auth[55:55 + cred_len] == key_id, "credential id is not the key id")          # 9
    return pub, Verdict("APP_ATTEST", TIER_T1, auth[:32], hashlib.sha256(client_data).digest(), {
        "environment": "production" if production else "development",
        "app": f"{team_id}.{bundle_id}"})


def verify_app_attest_assertion(assertion: bytes, client_data: bytes,
                                pub: ec.EllipticCurvePublicKey, team_id: str, bundle_id: str,
                                prev_counter: int) -> int:
    """Per-request proof from an attested key. Returns the new counter."""
    obj = cbor2.loads(assertion)
    auth, sig = obj["authenticatorData"], obj["signature"]
    nonce = hashlib.sha256(auth + hashlib.sha256(client_data).digest()).digest()
    try:
        pub.verify(sig, nonce, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        raise AttestationError("assertion signature invalid")
    _require(auth[:32] == _rp_id_hash(team_id, bundle_id), "app id does not match")
    counter = int.from_bytes(auth[33:37], "big")
    _require(counter > prev_counter, "assertion counter did not increase (replay)")
    return counter


def verify_app_attest_job(assertion: bytes, report_data: bytes, pub: ec.EllipticCurvePublicKey,
                          team_id: str, bundle_id: str, prev_counter: int) -> tuple[int, Verdict]:
    """Binds one job: the assertion's client data is the job's report_data.

    Not checked yet: the macOS 27 key-policy property that proves Full Security
    and SIP were on. Apple documents it (WWDC26 session 201) but no macOS 27
    sample was available to test against; the policy must require it before
    T1 is offered with private data.
    """
    counter = verify_app_attest_assertion(assertion, report_data, pub, team_id, bundle_id,
                                          prev_counter)
    return counter, Verdict("APP_ATTEST", TIER_T1, _rp_id_hash(team_id, bundle_id), report_data,
                            {"app": f"{team_id}.{bundle_id}", "counter": counter,
                             "sip_policy_checked": False})


# --- Development mock ------------------------------------------------------------


def verify_mock(attestation: dict, sig_hex: str) -> Verdict:
    """The local Docker sandbox's self-signed attestation. Proves nothing about
    hardware; the policy must explicitly allow it (development only)."""
    import beetle_crypto as bc

    sign_pk = bytes.fromhex(attestation["sign_pk"])
    _require(bc.verify(sign_pk, bytes.fromhex(sig_hex),
                       bc.digest("attest", bc.canonical_json(attestation))), "bad mock signature")
    _require(attestation["network"]["isolated"] is True, "sandbox reported network access")
    return Verdict("MOCK", TIER_MOCK, bytes.fromhex(attestation["measurement"]),
                   bytes.fromhex(attestation["report_data"]), {"mock": True})


def load_json(path: Path):
    return json.loads(Path(path).read_text())
