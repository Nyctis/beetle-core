"""TPM 2.0 attestation, end to end against a software TPM (swtpm) in Docker.

The host side runs real tpm2-tools commands inside the container; the
verifier side is common/tpm_attest.py. Then each attack must be rejected.
Run on a machine with Docker, from the beetle directory:

    docker build -t beetle-swtpm:dev -f host/tpm/Dockerfile host/tpm
    uv run python tests/tpm_interop_test.py
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "host")]

import tpm_attest as tpm  # noqa: E402
from evidence import SwTpm  # noqa: E402
from cryptography import x509  # noqa: E402

results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def rejects(name: str, fn, match: str) -> None:
    try:
        fn()
        check(name, False, "UNEXPECTEDLY ACCEPTED")
    except tpm.TpmError as e:
        check(name, match in str(e), f"rejected: {e}")


def enroll(t: SwTpm, ca_certs) -> tuple[tpm.AkPublic, bool]:
    """Verifier side of enrollment: EK cert, AK public area, credential activation."""
    t.run("enroll")
    ek_pub = tpm.verify_ek_cert(t.get("/work/ek.crt"), ca_certs)
    ak = tpm.parse_ak_public(t.get("/work/ak.pub"))
    secret = os.urandom(32)
    t.put(tpm.make_credential(ek_pub, ak.name, secret), "/work/cred.bin")
    try:
        activated = bytes.fromhex(t.run("activate").split()[-1])
    except RuntimeError as e:
        print(f"    activation error: {e}")
        activated = b""
    return ak, activated == secret


def quote(t: SwTpm, nonce: bytes) -> tuple[bytes, bytes, dict[int, bytes]]:
    t.run("quote", nonce.hex())
    return t.get("/work/quote.msg"), t.get("/work/quote.sig"), t.pcr_values()


def main() -> int:
    image_measurement = hashlib.sha256(b"beetle locked host image v0").digest()
    expected = {i: bytes(32) for i in range(8)}          # swtpm has no firmware measurements
    expected[23] = tpm.pcr_extend(bytes(32), image_measurement)
    a, b = SwTpm("beetle-tpm-a", fresh=True), SwTpm("beetle-tpm-b", fresh=True)
    try:
        ca_a = [x509.load_pem_x509_certificate(pem) for pem in a.ca_pems().values()]

        print("enrollment (EK certificate + credential activation)")
        ak, activated = enroll(a, ca_a)
        check("AK name computed by the verifier equals the TPM's", ak.name == a.get("/work/ak.name"))
        check("TPM recovered the verifier's secret: AK lives in this TPM", activated)

        print("quote over the measured boot, bound to a job")
        a.run("boot", image_measurement.hex())
        report_data = os.urandom(32)
        msg, sig, values = quote(a, report_data)
        try:
            q = tpm.verify_quote(msg, sig, ak, values, report_data, expected)
            check("valid quote verifies", True, f"PCR 23 = locked image, reset count {q.reset_count}")
        except tpm.TpmError as e:
            check("valid quote verifies", False, str(e))

        print("attacks")
        rejects("quote replayed for another job", lambda: tpm.verify_quote(
            msg, sig, ak, values, os.urandom(32), expected), "nonce")
        bad = bytearray(msg)
        bad[-1] ^= 1
        rejects("quote bytes tampered", lambda: tpm.verify_quote(
            bytes(bad), sig, ak, values, report_data, expected), "signature")
        forged = dict(values)
        forged[23] = expected[23]
        forged[0] = os.urandom(32)
        rejects("host lies about PCR values", lambda: tpm.verify_quote(
            msg, sig, ak, forged, report_data, expected), "signed digest")

        b.run("enroll")
        b.run("boot", hashlib.sha256(b"modified image that dumps memory").hexdigest())
        ak_b = tpm.parse_ak_public(b.get("/work/ak.pub"))
        msg_b, sig_b, values_b = quote(b, report_data)
        rejects("machine booted a different image", lambda: tpm.verify_quote(
            msg_b, sig_b, ak_b, values_b, report_data, expected), "PCR 23")
        rejects("quote signed by another TPM's AK", lambda: tpm.verify_quote(
            msg_b, sig_b, ak, values_b, report_data, expected), "signature")

        # TPM B tries to activate a credential made for TPM A's EK and AK name.
        ek_pub_a = tpm.verify_ek_cert(a.get("/work/ek.crt"), ca_a)
        secret = os.urandom(32)
        b.put(tpm.make_credential(ek_pub_a, ak_b.name, secret), "/work/cred.bin")
        try:
            got = bytes.fromhex(b.run("activate").split()[-1])
        except RuntimeError:
            got = b""
        check("credential for TPM A cannot be activated by TPM B", got != secret)

        rejects("EK certificate from an untrusted CA", lambda: tpm.verify_ek_cert(
            b.get("/work/ek.crt"), [c for c in ca_a if c.subject != c.issuer]), "chain")
        tpmt = bytearray(ak.tpmt_public)
        attrs = int.from_bytes(tpmt[4:8], "big") & ~tpm.ATTR_RESTRICTED
        tpmt[4:8] = attrs.to_bytes(4, "big")
        rejects("unrestricted key presented as an AK", lambda: tpm.parse_ak_public(
            len(tpmt).to_bytes(2, "big") + bytes(tpmt)), "restricted")
    finally:
        a.close()
        b.close()

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
