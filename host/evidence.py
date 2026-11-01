"""Evidence a host presents to the attester for each job.

NoEvidence:   an ordinary machine with nothing to prove, tier T3-C. It passes on
              the sandbox's self-signed attestation, which ties the sandbox's keys
              to this job and shows no one anything about the machine.
MockEvidence: the Docker sandbox's self-signed attestation (development only).
TpmEvidence:  a quote from the host's TPM, bound to the job's report_data.
              On this VPS the TPM is swtpm in a container (host/tpm/); a real
              T3-B host would use its own TPM through the same tpm2-tools calls.

TDX and App Attest evidence come from the sandbox itself (inside the
confidential VM, or the Mac app), so they have no host-side provider here.
"""

from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import beetle_crypto as bc

TPM_IMAGE = "beetle-swtpm:dev"
QUOTED_PCRS = [0, 1, 2, 3, 4, 5, 6, 7, 23]


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"attester refused: {json.loads(e.read()).get('error')}") from None


class SwTpm:
    """A software TPM in its own container, driven with tpm2-tools."""

    def __init__(self, name: str, fresh: bool = False):
        self.name = name
        running = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name],
                                 capture_output=True, text=True).stdout.strip() == "true"
        if fresh or not running:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            subprocess.run(["docker", "run", "-d", "--name", name, TPM_IMAGE], check=True,
                           capture_output=True)
            for _ in range(30):
                if "tpm ready" in subprocess.run(["docker", "logs", name], capture_output=True,
                                                 text=True).stdout:
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError("swtpm did not start")

    def run(self, *args: str) -> str:
        r = subprocess.run(["docker", "exec", self.name, "tpm_host.sh", *args],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{' '.join(args)} failed: {r.stderr.strip()[:300]}")
        return r.stdout

    def get(self, path: str) -> bytes:
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["docker", "cp", f"{self.name}:{path}", f"{d}/f"], check=True,
                           capture_output=True)
            return Path(f"{d}/f").read_bytes()

    def put(self, data: bytes, path: str) -> None:
        with tempfile.TemporaryDirectory() as d:
            Path(f"{d}/f").write_bytes(data)
            subprocess.run(["docker", "cp", f"{d}/f", f"{self.name}:{path}"], check=True,
                           capture_output=True)

    def ca_pems(self) -> dict[str, bytes]:
        return {f: self.get(f"/var/lib/swtpm-localca/{f}")
                for f in ("issuercert.pem", "swtpm-localca-rootca-cert.pem")}

    def pcr_values(self) -> dict[int, bytes]:
        pcrs = self.get("/work/pcrs.bin")
        return {i: pcrs[k * 32:(k + 1) * 32] for k, i in enumerate(QUOTED_PCRS)}

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


class MockEvidence:
    kind = "MOCK"

    def evidence(self, outbox: Path, report_data: bytes) -> dict:
        return {"type": self.kind, "attestation": json.loads((outbox / "attestation.json").read_text()),
                "signature": (outbox / "attestation.sig").read_text()}


class NoEvidence(MockEvidence):
    kind = "NONE"


class TpmEvidence:
    """Measured boot, one-time enrollment with the attester, then a quote per job."""

    def __init__(self, attester_url: str, container: str, image_measurement: bytes, log):
        self.url, self.log = attester_url, log
        self.tpm = SwTpm(container)
        # Locked boot: the image extends PCR 23 once per TPM power cycle.
        self.tpm.run("pcrs")
        if self.tpm.pcr_values()[23] == bytes(32):
            self.tpm.run("boot", image_measurement.hex())
            log("TPM: measured boot extended PCR 23 with the locked image")
        self.tpm.run("enroll")
        begin = post(f"{self.url}/tpm/enroll-begin", {
            "ek_cert": base64.b64encode(self.tpm.get("/work/ek.crt")).decode(),
            "ak_public": base64.b64encode(self.tpm.get("/work/ak.pub")).decode()})
        self.tpm.put(base64.b64decode(begin["credential"]), "/work/cred.bin")
        secret = bytes.fromhex(self.tpm.run("activate").split()[-1])
        post(f"{self.url}/tpm/enroll-finish", {"enrollment_id": begin["enrollment_id"],
                                                "secret": base64.b64encode(secret).decode()})
        self.enrollment_id = begin["enrollment_id"]
        log(f"TPM: enrolled with the attester (credential activation passed), "
            f"id {self.enrollment_id[:12]}")

    def evidence(self, outbox: Path, report_data: bytes) -> dict:
        self.tpm.run("quote", report_data.hex())
        return {"type": "TPM2", "enrollment_id": self.enrollment_id,
                "quote": base64.b64encode(self.tpm.get("/work/quote.msg")).decode(),
                "signature": base64.b64encode(self.tpm.get("/work/quote.sig")).decode(),
                "pcrs": {str(i): v.hex() for i, v in self.tpm.pcr_values().items()}}


def register(attester_url: str, job_id: bytes, spec_hash: bytes, att: dict, host: bytes,
             evidence: dict) -> dict:
    """Ask the attester for a registration; returns its JSON (hex fields)."""
    return post(f"{attester_url}/register", {
        "job_id": job_id.hex(), "spec_hash": spec_hash.hex(), "sign_pk": att["sign_pk"],
        "enc_pk": att["enc_pk"], "host": host.hex(), "evidence": evidence})


def report_data_for(job_id: bytes, spec_hash: bytes, att: dict) -> bytes:
    return bc.digest("start", job_id + spec_hash + bytes.fromhex(att["sign_pk"])
                     + bytes.fromhex(att["enc_pk"]))
