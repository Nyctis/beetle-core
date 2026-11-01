"""The buyer's local check of TPM evidence, against a software TPM (swtpm) in Docker.

No attester and no chain: the buyer enrolls the seller's TPM itself and decides
from the quote whether the data key may go. Run on a machine with Docker, from
the beetle directory:

    docker build -t beetle-swtpm:dev -f host/tpm/Dockerfile host/tpm
    uv run python tests/tpm_buyer_test.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "host"), str(ROOT / "requester"), str(ROOT / "attester")]

import beetle_crypto as bc  # noqa: E402
import tpm_attest as tpm  # noqa: E402
from beetle_chain import evidence_hash  # noqa: E402
from evidence import SwTpm  # noqa: E402
from job_defaults import sandbox_code_files  # noqa: E402
from requester import AttestationRejected, Requester  # noqa: E402
from server import dev_policy  # noqa: E402
from verifier import EvidenceVerifier  # noqa: E402

TIER_T3B = 3
results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def refused(name: str, req: Requester, verifier: EvidenceVerifier, evidence: dict, job) -> None:
    try:
        req.check_evidence(verifier, evidence, job, mainnet=False)
        check(name, False, "KEY WOULD HAVE BEEN RELEASED")
    except AttestationRejected as e:
        check(name, True, f"refused: {str(e)[:70]}")


def main() -> int:
    t = SwTpm("beetle-buyer-tpm-test", fresh=True)
    try:
        with tempfile.TemporaryDirectory() as d:
            ca_dir = Path(d)
            for name, pem in t.ca_pems().items():
                (ca_dir / name).write_bytes(pem)
            verifier = EvidenceVerifier(dev_policy(ca_dir), ca_dir / "state.json")

            image = bytes.fromhex(bc.code_measurement(sandbox_code_files()))
            t.run("pcrs")
            t.run("boot", image.hex())
            t.run("enroll")
            eid, credential = verifier.tpm_enroll_begin(t.get("/work/ek.crt"), t.get("/work/ak.pub"))
            t.put(credential, "/work/cred.bin")
            verifier.tpm_enroll_finish(eid, bytes.fromhex(t.run("activate").split()[-1]))
            check("the buyer enrolled the seller's TPM itself", eid in verifier.state["tpm"])

            req = Requester(os.urandom(32), os.urandom(32))
            req.job_id, req.spec_hash = os.urandom(32), os.urandom(32)
            sign_pk, enc_pk = os.urandom(32), os.urandom(32)
            t.run("quote", bc.digest("start", req.job_id + req.spec_hash + sign_pk + enc_pk).hex())
            evidence = {"type": "TPM2", "enrollment_id": eid,
                        "quote": base64.b64encode(t.get("/work/quote.msg")).decode(),
                        "signature": base64.b64encode(t.get("/work/quote.sig")).decode(),
                        "pcrs": {str(i): v.hex() for i, v in t.pcr_values().items()}}
            job = SimpleNamespace(sign_pk=sign_pk, enc_pk=enc_pk, tier=TIER_T3B,
                                  measurement=tpm.pcr_extend(bytes(32), image),
                                  evidence_hash=evidence_hash(evidence))

            verdict = req.check_evidence(verifier, evidence, job, mainnet=False)
            check("a quote of this job on the locked image passes", verdict.evidence_type == "TPM2"
                  and len(req.seal_dek()) == 80)

            def variant(**over):
                return SimpleNamespace(**{**vars(job), **over})

            refused("evidence other than the committed hash", req, verifier, evidence,
                    variant(evidence_hash=os.urandom(32)))
            refused("a quote made for other sandbox keys", req, verifier, evidence,
                    variant(enc_pk=os.urandom(32)))
            other = Requester(os.urandom(32), os.urandom(32))
            other.job_id, other.spec_hash = os.urandom(32), req.spec_hash
            refused("a quote replayed from another job", other, verifier, evidence, job)
            bad_pcrs = json.loads(json.dumps(evidence))
            bad_pcrs["pcrs"]["23"] = os.urandom(32).hex()
            refused("PCR values the quote doesn't cover", req, verifier, bad_pcrs,
                    variant(evidence_hash=evidence_hash(bad_pcrs)))
            bad_sig = dict(evidence, signature=base64.b64encode(os.urandom(70)).decode())
            refused("a forged quote signature", req, verifier, bad_sig,
                    variant(evidence_hash=evidence_hash(bad_sig)))
            stranger = dict(evidence, enrollment_id="0" * 32)
            refused("a TPM the buyer never enrolled", req, verifier, stranger,
                    variant(evidence_hash=evidence_hash(stranger)))
            strict = EvidenceVerifier({**dev_policy(ca_dir), "tpm": {
                **dev_policy(ca_dir)["tpm"], "allowed_measurements": []}}, ca_dir / "state.json")
            refused("a measurement this buyer doesn't allow", req, strict, evidence, job)
            refused("the offer claims a tier the evidence isn't", req, verifier, evidence, variant(tier=2))
    finally:
        t.close()
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
