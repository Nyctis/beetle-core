"""TPM 2.0 remote attestation, verifier side (tier T3-B).

Two protocols, both from the TCG TPM 2.0 specification:

1. Credential activation binds an attestation key (AK) to one TPM. The
   verifier checks the endorsement key (EK) certificate against the TPM
   vendor's CA, then encrypts a random secret to the EK so that it can only
   be recovered by a TPM that holds both the EK and a key with the AK's
   exact name (MakeCredential / ActivateCredential).
2. A quote is the AK's signature over PCR values (the boot measurements) and
   a nonce. The nonce is the job's report_data, so a quote can't be reused.

What this proves: the machine booted the measured, locked image and the
quote came from a real TPM. What it does not: that the owner cannot read
memory. A TPM has no memory protection, and its bus can be sniffed.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
from dataclasses import dataclass
from datetime import datetime, timezone

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

try:  # CFB moved to the 'decrepit' module in cryptography 47
    from cryptography.hazmat.decrepit.ciphers.modes import CFB
except ImportError:  # pragma: no cover
    from cryptography.hazmat.primitives.ciphers.modes import CFB

ALG_RSA, ALG_SHA256, ALG_NULL, ALG_AES, ALG_CFB, ALG_RSASSA = 0x0001, 0x000B, 0x0010, 0x0006, 0x0043, 0x0014
TPM_GENERATED_VALUE = 0xFF544347
ST_ATTEST_QUOTE = 0x8018
ATTR_FIXED_TPM, ATTR_FIXED_PARENT, ATTR_SENSITIVE_DATA_ORIGIN = 1 << 1, 1 << 4, 1 << 5
ATTR_RESTRICTED, ATTR_DECRYPT, ATTR_SIGN = 1 << 16, 1 << 17, 1 << 18
CRED_FILE_MAGIC, CRED_FILE_VERSION = 0xBADCC0DE, 1  # tpm2-tools credential file header


class TpmError(Exception):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise TpmError(msg)


class Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        _require(self.pos + n <= len(self.data), "truncated TPM structure")
        v = self.data[self.pos:self.pos + n]
        self.pos += n
        return v

    def u8(self):
        return self.take(1)[0]

    def u16(self):
        return struct.unpack(">H", self.take(2))[0]

    def u32(self):
        return struct.unpack(">I", self.take(4))[0]

    def u64(self):
        return struct.unpack(">Q", self.take(8))[0]

    def tpm2b(self) -> bytes:
        return self.take(self.u16())

    def done(self) -> bool:
        return self.pos == len(self.data)


# --- Attestation key public area -----------------------------------------------


@dataclass
class AkPublic:
    tpmt_public: bytes
    name: bytes
    attributes: int
    public_key: rsa.RSAPublicKey


def parse_ak_public(tpm2b_public: bytes) -> AkPublic:
    """Parse an RSA TPM2B_PUBLIC and check it is a TPM-bound restricted signing key."""
    outer = Reader(tpm2b_public)
    tpmt = outer.tpm2b()
    _require(outer.done(), "trailing bytes after TPM2B_PUBLIC")
    r = Reader(tpmt)
    _require(r.u16() == ALG_RSA, "AK must be an RSA key")
    _require(r.u16() == ALG_SHA256, "AK name algorithm must be SHA-256")
    attrs = r.u32()
    r.tpm2b()                                   # authPolicy
    sym = r.u16()
    if sym != ALG_NULL:
        r.u16(), r.u16()                        # keyBits, mode
    scheme = r.u16()
    if scheme != ALG_NULL:
        _require(scheme == ALG_RSASSA and r.u16() == ALG_SHA256, "AK scheme must be RSASSA-SHA256")
    bits = r.u16()
    exponent = r.u32() or 65537
    modulus = r.tpm2b()
    _require(r.done() and len(modulus) * 8 == bits, "malformed RSA public area")
    required = ATTR_FIXED_TPM | ATTR_FIXED_PARENT | ATTR_SENSITIVE_DATA_ORIGIN | ATTR_RESTRICTED | ATTR_SIGN
    _require(attrs & required == required and not attrs & ATTR_DECRYPT,
             "AK must be a fixed, TPM-generated, restricted signing key")
    pub = rsa.RSAPublicNumbers(exponent, int.from_bytes(modulus, "big")).public_key()
    name = struct.pack(">H", ALG_SHA256) + hashlib.sha256(tpmt).digest()
    return AkPublic(tpmt, name, attrs, pub)


# --- Endorsement key certificate -------------------------------------------------


def _check_signed_by(cert: x509.Certificate, issuer: x509.Certificate) -> None:
    key = issuer.public_key()
    try:
        if isinstance(key, rsa.RSAPublicKey):
            key.verify(cert.signature, cert.tbs_certificate_bytes, padding.PKCS1v15(),
                       cert.signature_hash_algorithm)
        elif isinstance(key, ec.EllipticCurvePublicKey):
            key.verify(cert.signature, cert.tbs_certificate_bytes,
                       ec.ECDSA(cert.signature_hash_algorithm))
        else:
            raise TpmError("unsupported CA key type")
    except InvalidSignature:
        raise TpmError("EK certificate chain signature invalid")


def trim_der(data: bytes) -> bytes:
    """NV reads return the whole index; keep just the DER certificate at its start."""
    _require(len(data) >= 4 and data[0] == 0x30, "not a DER certificate")
    if data[1] < 0x80:
        end = 2 + data[1]
    else:
        n = data[1] & 0x7F
        end = 2 + n + int.from_bytes(data[2:2 + n], "big")
    _require(end <= len(data) and not any(data[end:]), "unexpected bytes after the certificate")
    return data[:end]


def verify_ek_cert(ek_cert_der: bytes, ca_certs: list[x509.Certificate],
                   at: datetime | None = None) -> rsa.RSAPublicKey:
    """The EK certificate must chain to a trusted TPM manufacturer CA.

    For real hardware, ca_certs are the vendors' published EK CAs (Infineon,
    Nuvoton, STMicro, AMD fTPM, Intel PTT, Microsoft Pluton). For swtpm it is
    the local test CA, which proves nothing about hardware.
    """
    at = at or datetime.now(timezone.utc)
    cert = x509.load_der_x509_certificate(trim_der(ek_cert_der))
    by_subject = {c.subject: c for c in ca_certs}
    cur, depth = cert, 0
    while True:
        _require(cur.not_valid_before_utc <= at <= cur.not_valid_after_utc,
                 "EK certificate chain not valid now")
        issuer = by_subject.get(cur.issuer)
        _require(issuer is not None, "EK certificate does not chain to a trusted TPM CA")
        _check_signed_by(cur, issuer)
        if issuer.subject == issuer.issuer:      # reached a self-signed root
            _check_signed_by(issuer, issuer)
            break
        cur, depth = issuer, depth + 1
        _require(depth < 4, "EK certificate chain too long")
    pub = cert.public_key()
    _require(isinstance(pub, rsa.RSAPublicKey) and pub.key_size == 2048, "EK must be RSA-2048")
    return pub


# --- MakeCredential (TPM 2.0 Part 1, sections 24 and 11.4.10) ---------------------


def kdfa(key: bytes, label: bytes, context_u: bytes, context_v: bytes, bits: int) -> bytes:
    """SP800-108 counter-mode KDF with HMAC-SHA256, as the TPM defines it."""
    out, counter = b"", 1
    while len(out) * 8 < bits:
        out += hmac.new(key, struct.pack(">I", counter) + label + b"\x00" + context_u + context_v
                        + struct.pack(">I", bits), hashlib.sha256).digest()
        counter += 1
    return out[:bits // 8]


def make_credential(ek_pub: rsa.RSAPublicKey, ak_name: bytes, secret: bytes) -> bytes:
    """Encrypt `secret` so only the TPM holding this EK and a key named `ak_name`
    can recover it. Returns a tpm2-tools credential file for tpm2_activatecredential.

    The EK uses the default TCG template: SHA-256 name algorithm and AES-128-CFB.
    """
    _require(len(secret) <= 32, "secret too long")
    seed = os.urandom(32)
    enc_seed = ek_pub.encrypt(seed, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                                 algorithm=hashes.SHA256(), label=b"IDENTITY\x00"))
    sym_key = kdfa(seed, b"STORAGE", ak_name, b"", 128)
    plaintext = struct.pack(">H", len(secret)) + secret          # TPM2B_DIGEST
    enc = Cipher(algorithms.AES(sym_key), CFB(bytes(16))).encryptor()
    enc_identity = enc.update(plaintext) + enc.finalize()
    hmac_key = kdfa(seed, b"INTEGRITY", b"", b"", 256)
    outer_hmac = hmac.new(hmac_key, enc_identity + ak_name, hashlib.sha256).digest()
    id_object = struct.pack(">H", len(outer_hmac)) + outer_hmac + enc_identity
    return (struct.pack(">II", CRED_FILE_MAGIC, CRED_FILE_VERSION)
            + struct.pack(">H", len(id_object)) + id_object
            + struct.pack(">H", len(enc_seed)) + enc_seed)


# --- Quote ------------------------------------------------------------------------


@dataclass
class Quote:
    nonce: bytes
    pcr_selection: list[int]
    pcr_digest: bytes
    clock: int
    reset_count: int
    restart_count: int
    firmware_version: int


def parse_quote(attest: bytes) -> Quote:
    r = Reader(attest)
    _require(r.u32() == TPM_GENERATED_VALUE, "not generated by a TPM")
    _require(r.u16() == ST_ATTEST_QUOTE, "not a quote")
    r.tpm2b()                                   # qualifiedSigner
    nonce = r.tpm2b()                           # extraData
    clock, reset, restart = r.u64(), r.u32(), r.u32()
    r.u8()                                      # safe
    fw = r.u64()
    count = r.u32()
    _require(count == 1, "expected one PCR bank")
    _require(r.u16() == ALG_SHA256, "expected the SHA-256 PCR bank")
    bitmap = r.take(r.u8())
    pcrs = [i for i in range(len(bitmap) * 8) if bitmap[i // 8] >> (i % 8) & 1]
    digest = r.tpm2b()
    _require(r.done(), "trailing bytes after quote")
    return Quote(nonce, pcrs, digest, clock, reset, restart, fw)


def verify_quote(attest: bytes, signature: bytes, ak: AkPublic, pcr_values: dict[int, bytes],
                 expected_nonce: bytes, expected_pcrs: dict[int, bytes]) -> Quote:
    """Check the AK's signature, the nonce, and that the signed PCR digest
    matches PCR values that equal the policy's expected boot measurements."""
    try:
        ak.public_key.verify(signature, attest, padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature:
        raise TpmError("quote signature invalid")
    q = parse_quote(attest)
    _require(q.nonce == expected_nonce, "quote nonce does not bind this job")
    _require(q.pcr_selection == sorted(expected_pcrs), "quote covers the wrong PCRs")
    _require(sorted(pcr_values) == q.pcr_selection, "PCR values do not match the selection")
    composite = hashlib.sha256(b"".join(pcr_values[i] for i in q.pcr_selection)).digest()
    _require(composite == q.pcr_digest, "PCR values do not match the signed digest")
    for i, want in expected_pcrs.items():
        _require(pcr_values[i] == want, f"PCR {i} differs from the locked image's measurement")
    return q


def pcr_extend(value: bytes, measurement: bytes) -> bytes:
    """What PCR_Extend does: new = SHA256(old || measurement)."""
    return hashlib.sha256(value + measurement).digest()
