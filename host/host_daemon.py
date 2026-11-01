"""Host daemon: watch the beetle program for jobs taken from this seller's
offers, run each one in the offline sandbox, and post start/result on chain.

The host never holds a usable key: the requester's data key arrives on chain
sealed to the sandbox, and the result is signed by the sandbox. Before `start`
the host collects the sandbox's hardware evidence (MOCK on a plain VPS, TPM2
with --evidence tpm), commits to its hash on chain and hands it to the buyer,
who verifies it locally. A job that names a verifier also needs that verifier's
registration.

Usage (on the host machine, from the beetle directory):
    uv run python host/host_daemon.py --chain chain.json --once [--evidence tpm] [--post-offer]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "host"), str(ROOT / "attester")]

import beetle_crypto as bc  # noqa: E402
import tpm_attest as tpm  # noqa: E402
from attester import Registration  # noqa: E402  (message layout shared with the attester)
from evidence import MockEvidence, TpmEvidence, register, report_data_for  # noqa: E402
from job_defaults import BUDGET, MODEL_DIR, PRICE, TIERS, sandbox_code_files  # noqa: E402
from beetle_chain import (FINISHED, KIND_TRAINING, NO_VERIFIER, Chain, Job,  # noqa: E402
                          OfferTerms, evidence_hash, load_keypair)
from sandbox_host import ensure_image, ensure_model, sandbox_cmd, wait_for  # noqa: E402
from solders.pubkey import Pubkey  # noqa: E402

OFFER_IDX = 0
STAKE = 100_000_000  # 100 test-USDC
DECLINE_MARGIN = 2   # seconds


def log(msg: str) -> None:
    print(f"[host]    {time.strftime('%H:%M:%S')} {msg}", flush=True)


def ensure_offer(chain: Chain, kp, measurement: bytes, tier: int) -> None:
    """Registers, stakes and posts one training offer unless already done."""
    me = kp.pubkey()
    if chain.get_seller(me) is None:
        chain.register_seller(kp)
        chain.stake(kp, STAKE)
        log(f"registered as a seller, staked {STAKE / 1e6:.0f} test-USDC")
    if chain.get_offer(chain.offer_pda(me, OFFER_IDX)) is None:
        tokens = BUDGET["max_train_tokens"]
        chain.post_offer(kp, me, OFFER_IDX, OfferTerms(
            KIND_TRAINING, tier, measurement, price_a=PRICE * 1_000_000 // tokens,
            start_window_s=900, grace_s=BUDGET["max_wall_seconds"],
            valid_until=chain.chain_time() + 7 * 86_400))
        log(f"offer posted: tier {tier}, {PRICE * 1_000_000 / tokens / 1e6:.2f} test-USDC per million tokens")


def run_job(chain: Chain, host_kp, job: Job, job_dir: Path, provider, attester_url: str) -> None:
    job_id, job_host = job.job_id, job.host
    inbox, outbox = job_dir / "in", job_dir / "out"
    spec = json.loads((inbox / "spec.json").read_text())
    if bc.sha256(bc.canonical_json(spec)) != job.spec_hash:
        log("spec in object storage does not match the on-chain spec_hash; skipping")
        return
    bc.configure_from_spec(spec)

    base = ensure_model(spec["base_model"]["revision"])
    if base["manifest_sha256"] != spec["base_model"]["manifest_sha256"]:
        log("cannot serve this base model (manifest differs); skipping")
        return
    ensure_image()
    outbox.mkdir(exist_ok=True)
    outbox.chmod(0o777)

    container = f"beetle-{job.job_id.hex()[:12]}"
    subprocess.run(["docker", "rm", "-f", container], capture_output=True)
    subprocess.run(sandbox_cmd(container, MODEL_DIR, inbox, outbox), check=True, capture_output=True)
    log(f"sandbox {container} started (--network none)")

    try:
        got = wait_for([outbox / "attestation.json", outbox / "error.json"], container, 120)
        if got is None or got.name == "error.json":
            log("sandbox did not attest; the requester can reclaim via timeout")
            return
        att = json.loads(got.read_text())
        report_data = report_data_for(job.job_id, job.spec_hash, att)
        evidence = provider.evidence(outbox, report_data)
        (outbox / "evidence.json").write_text(json.dumps(evidence))  # P2P to the buyer
        registration = None
        if job.verifier != NO_VERIFIER:
            try:
                reg = register(attester_url, job.job_id, job.spec_hash, att, bytes(job.host), evidence)
            except RuntimeError as e:
                log(f"{e}; not starting (the requester can reclaim via timeout)")
                return
            registration = (bytes.fromhex(reg["attester_pk"]), bytes.fromhex(reg["signature"]),
                            Registration.from_json(reg).message(), reg["expiry"])
        sig = chain.start(host_kp, job, bytes.fromhex(att["sign_pk"]), bytes.fromhex(att["enc_pk"]),
                          job.measurement, evidence_hash(evidence), registration)
        log(f"start tx {sig[:20]}...  sandbox keys declared, {evidence['type']} evidence committed")

        job = chain.get_job(job_id)
        job = chain.wait_job(job_id, lambda j: j.state != "Started",
                             timeout=job.release_by - chain.chain_time() + 60)
        if job is None or not job.has_dek:
            log(f"the requester did not release the key ({job.state if job else 'job closed'})")
            return
        tmp = inbox / ".sealed_dek.tmp"
        tmp.write_bytes(job.sealed_dek)
        tmp.rename(inbox / "sealed_dek.bin")
        log("sealed DEK read from chain and handed to the sandbox")

        log("training inside the sandbox...")
        subprocess.run(["docker", "wait", container], capture_output=True)
        if not (outbox / "report.json").exists():
            # A signed report covers every outcome the chain can act on. Without
            # one (dek_timeout, a crash) the job ends through the timeouts.
            reason = json.loads((outbox / "error.json").read_text())["reason"] \
                if (outbox / "error.json").exists() else "no report"
            log(f"sandbox aborted: {reason}")
            return

        report = json.loads((outbox / "report.json").read_text())
        if "outcome" not in report:
            log("sandbox image predates the v0.2 report format (no outcome); cannot submit")
            return
        report_hash = bc.sha256(bc.canonical_json(report))
        sig = chain.submit_result(host_kp, job, report, report_hash,
                                  bytes.fromhex((outbox / "settle.sig").read_text()))
        log(f"submit_result tx {sig[:20]}...  outcome={report['outcome']} "
            f"work_bps={report['work_bps']} (scores are sealed to the requester)")

        # Get paid: the requester may accept early; otherwise finalize after
        # the settle delay. finalize also pays out; claim_due covers the case
        # where someone else settled but left this seller's claim.
        job = chain.get_job(job.job_id)
        while job is not None:
            try:
                if job.state == "Delivered" and chain.chain_time() >= job.challenge_end:
                    log(f"finalize tx {chain.finalize(host_kp, job)[:20]}...")
                elif job.state != "Delivered":
                    chain.claim_due(host_kp, job.job_id)
            except RuntimeError as e:
                log(f"settlement not accepted yet: {str(e)[:80]}")
            time.sleep(3)
            job = chain.get_job(job.job_id)
        log(f"job settled; host balance {chain.token_balance(job_host):.2f} test-USDC")
    finally:
        logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
        (job_dir / "sandbox.log").write_text(logs.stdout + logs.stderr)
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        # Delete-after-use: once the job is over, no ciphertext or sealed key
        # for it stays on this machine. The sandbox's keys died with it.
        final = chain.get_job(job_id)
        if final is None or final.state in FINISHED:
            for sub in ("in", "out"):
                shutil.rmtree(job_dir / sub, ignore_errors=True)
            log("job inputs and outputs deleted from this host")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="chain.json")
    ap.add_argument("--key", default=".keys/host.json", help="the seller's operator key")
    ap.add_argument("--authority", help="the seller's authority, if it isn't --key")
    ap.add_argument("--post-offer", action="store_true",
                    help="register, stake and post an offer first (--key is also the authority)")
    ap.add_argument("--once", action="store_true", help="exit after one job")
    ap.add_argument("--attester", default="http://127.0.0.1:8700",
                    help="verifier for TPM enrollment and for jobs that name one")
    ap.add_argument("--evidence", choices=["mock", "tpm"], default="mock")
    ap.add_argument("--tpm-container", default="beetle-host-tpm")
    args = ap.parse_args()

    chain = Chain.from_file(ROOT / args.chain)
    host_kp = load_keypair(ROOT / args.key)
    authority = Pubkey.from_string(args.authority) if args.authority else host_kp.pubkey()
    image = bytes.fromhex(bc.code_measurement(sandbox_code_files()))
    if args.evidence == "tpm":
        provider = TpmEvidence(args.attester, args.tpm_container, image, log)
        measurement, tier = tpm.pcr_extend(bytes(32), image), TIERS["t3b"]
    else:
        provider = MockEvidence()
        measurement, tier = image, TIERS["mock"]
    if args.post_offer:
        ensure_offer(chain, host_kp, measurement, tier)
    log(f"watching program {chain.program_id} for jobs of seller {authority} "
        f"(evidence: {args.evidence})")
    seen: set[bytes] = set()
    while True:
        for job in chain.jobs_for_host(authority):
            if job.state != "Opened" or job.job_id in seen:
                continue
            job_dir = ROOT / "jobs" / job.job_id.hex()[:16]
            if not (job_dir / "in" / "spec.json").exists():  # inputs not uploaded yet
                # Silence past the start window is slashed; declining in time isn't.
                free_until = min(job.opened_at + chain.get_config()["decline_window"], job.start_by)
                if chain.chain_time() >= free_until - DECLINE_MARGIN:
                    log(f"job {job.job_id.hex()[:16]}: no inputs arrived, declining")
                    chain.decline_job(host_kp, job)
                    chain.claim_due(host_kp, job.job_id)
                continue
            seen.add(job.job_id)
            log(f"job {job.job_id.hex()[:16]} opened: {job.price / 1e6:.2f} test-USDC in escrow, "
                f"{job.bond / 1e6:.2f} of stake locked")
            run_job(chain, host_kp, job, job_dir, provider, args.attester)
            if args.once:
                return 0
        time.sleep(3)


if __name__ == "__main__":
    sys.exit(main())
