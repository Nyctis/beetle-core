"""Requester (the user's agent) side of one job on chain.

Picks an offer from the chain, encrypts the data, uploads ciphertext to the
host's storage, takes the offer (escrow), verifies the sandbox's hardware
evidence on this machine, releases the data key only if that passes, verifies
the sandbox-signed result, opens the adapter and accepts.

Object storage and the P2P evidence channel are stood in for by the host's job
directory over SSH; only ciphertext and public metadata travel that way.

Usage (from the beetle directory):
    uv run python requester/run_requester.py --chain chain.json --host-ssh HOST --dev-policy

There's no way to skip verification: pass a policy (see attester/verifier.py)
or, on test networks, --dev-policy for MOCK and test-TPM evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "requester"), str(ROOT / "host"),
                str(ROOT / "attester")]

import beetle_crypto as bc  # noqa: E402
from job_defaults import (ACCEPTANCE, BUDGET, PRICE, RECIPE, TIER_LABELS, TIERS,  # noqa: E402
                          sandbox_code_files)
from beetle_chain import KIND_TRAINING, Chain, load_keypair  # noqa: E402
from requester import (AttestationRejected, Requester, load_or_create_storage_key,  # noqa: E402
                       load_stored_adapter)
from sandbox_host import ensure_model  # noqa: E402
from server import dev_policy  # noqa: E402
from verifier import EvidenceVerifier  # noqa: E402


def log(msg: str) -> None:
    print(f"[agent]   {time.strftime('%H:%M:%S')} {msg}", flush=True)


def ssh_quiet(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="chain.json")
    ap.add_argument("--key", default=".keys/requester.json")
    ap.add_argument("--host-ssh", required=True, help="ssh destination of the seller's machine")
    ap.add_argument("--host-dir", default="beetle")
    ap.add_argument("--program", help="directory with your own train.py (program mode)")
    ap.add_argument("--accept-tiers", default="mock,t3b",
                    help=f"comma-separated tiers to shop in: {', '.join(TIERS)}")
    policy = ap.add_mutually_exclusive_group(required=True)
    policy.add_argument("--policy", type=Path, help="evidence policy JSON (attester/verifier.py)")
    policy.add_argument("--dev-policy", action="store_true", help="test networks only")
    ap.add_argument("--tpm-ca-dir", type=Path, help="with --dev-policy: test TPM CA certificates")
    ap.add_argument("--verifier-state", default=".keys/attester_state.json",
                    help="enrolled devices, written by attester/server.py")
    args = ap.parse_args()
    accept = [TIERS[t.strip()] for t in args.accept_tiers.split(",")]

    chain = Chain.from_file(ROOT / args.chain)
    req_kp = load_keypair(ROOT / args.key)
    mainnet = chain.is_mainnet()
    if args.dev_policy and mainnet:
        log("REFUSED: --dev-policy accepts mock evidence, which is never valid on mainnet")
        return 1
    policy = dev_policy(args.tpm_ca_dir) if args.dev_policy else json.loads(args.policy.read_text())
    expected_measurement = bc.code_measurement(sandbox_code_files())

    # --- 1. Pick an offer ------------------------------------------------------
    units = BUDGET["max_train_tokens"]
    offers = [o for t in accept for o in chain.list_offers(kind=KIND_TRAINING, tier=t)]
    offers = sorted((o for o in offers if o.quote(units)[0] <= PRICE),
                    key=lambda o: o.quote(units)[0])
    if not offers:
        log(f"no live offer in tiers {accept} at or below {PRICE / 1e6:.2f} test-USDC")
        return 1
    offer = offers[0]
    log(f"{len(offers)} offers; taking {str(offer.address)[:12]}... tier {TIER_LABELS[offer.tier]} "
        f"at {offer.quote(units)[0] / 1e6:.2f} test-USDC")

    # --- 2. Encrypt locally ----------------------------------------------------
    storage_key = load_or_create_storage_key(ROOT / ".keys/storage.key")
    req = Requester(storage_key, wallet_pk=bytes(req_kp.pubkey()))
    memories = json.loads((ROOT / "demo/memories.json").read_text())
    eval_items = json.loads((ROOT / "demo/eval_hidden.json").read_text())
    base_model = ensure_model()
    tmp_dir = ROOT / "jobs" / f"req-{time.strftime('%Y%m%d-%H%M%S')}"
    job = req.prepare_job(tmp_dir, memories, eval_items, base_model, RECIPE, ACCEPTANCE, BUDGET,
                          chain=chain.spec_chain(),
                          program_dir=Path(args.program) if args.program else None,
                          open_slot=chain.current_slot())
    job_id = bytes.fromhex(job["job_id"])
    job_dir = ROOT / "jobs" / job["job_id"][:16]
    tmp_dir.rename(job_dir)
    log(f"job {job['job_id'][:16]}: data, hidden eval, recipe and "
        f"{'training code' if args.program else 'empty code slot'} encrypted locally, "
        f"plus {len(req.canaries)} canary records")

    # --- 3. Upload ciphertext (object-storage stand-in) ---------------------
    remote = f"{args.host_dir}/jobs/{job['job_id'][:16]}"
    ssh_quiet(["ssh", "-o", "BatchMode=yes", args.host_ssh, f"mkdir -p {remote}/in"])
    ssh_quiet(["rsync", "-az", f"{job_dir}/in/", f"{args.host_ssh}:{remote}/in/"])
    log(f"uploaded ciphertext only: {sorted(p.name for p in (job_dir / 'in').iterdir())}")

    # --- 4. Take the offer ------------------------------------------------------
    bal0 = chain.token_balance(req_kp.pubkey())
    sig = chain.open_job(req_kp, req_kp, offer, job_id, req.nonce, req.open_slot, req.spec_hash,
                         units, max_price=PRICE)
    log(f"open_job tx {sig[:20]}...  escrowed; balance {bal0:.2f} -> "
        f"{chain.token_balance(req_kp.pubkey()):.2f} test-USDC")

    # --- 5. Wait for the host to start; verify the evidence here ---------------
    onchain = chain.wait_job(job_id, lambda j: j.state != "Opened",
                             timeout=offer.start_window_s + 60)
    if onchain is None or onchain.state != "Started":
        log(f"job is {onchain.state if onchain else 'closed'}")
        return 1
    out = job_dir / "out"
    out.mkdir(exist_ok=True)
    ssh_quiet(["rsync", "-az", f"{args.host_ssh}:{remote}/out/evidence.json", f"{out}/"])
    evidence = json.loads((out / "evidence.json").read_text())
    verifier = EvidenceVerifier(policy, ROOT / args.verifier_state)
    try:
        verdict = req.check_evidence(verifier, evidence, onchain, mainnet)
    except AttestationRejected as e:
        log(f"REFUSED to release the key: {e}")
        sig = chain.reject_evidence(req_kp, onchain)
        chain.claim_due(req_kp, job_id)
        log(f"reject_evidence tx {sig[:20]}...  refunded less the capped fee")
        return 1
    log(f"{verdict.evidence_type} evidence verified locally: tier {TIER_LABELS[verdict.tier]}, "
        "measurement on the policy's allowlist, bound to this job and these sandbox keys")

    sealed = req.seal_dek()
    sig = chain.release_key(req_kp, job_id, sealed, onchain.evidence_hash)
    log(f"release_key tx {sig[:20]}...  ({len(sealed)}-byte DEK sealed to the sandbox key)")

    # --- 6. Wait for the sandbox-signed result -------------------------------
    onchain = chain.wait_job(job_id, lambda j: j.state != "Running",
                             timeout=onchain.run_secs + 60, every=5)
    if onchain is None or onchain.state != "Delivered":
        log(f"no result; job is {onchain.state if onchain else 'closed'}")
        return 1
    ssh_quiet(["rsync", "-az", f"{args.host_ssh}:{remote}/out/report.json",
               f"{args.host_ssh}:{remote}/out/report.sig", f"{out}/"])
    report = json.loads((out / "report.json").read_text())
    if report["private_report_ct_hash"] is None:
        # The sandbox stopped on an error: a signed outcome, and nothing else to fetch.
        ok = bc.verify(req.attested_sign_pk, bytes.fromhex((out / "report.sig").read_text()),
                       bc.digest("report", bc.canonical_json(report)))
        log(f"sandbox stopped: outcome {onchain.outcome}, signature {'OK' if ok else 'BAD'}")
        if ok:
            log(f"accept tx {chain.accept(req_kp, onchain)[:20]}...  settled by that outcome")
        return 1
    ssh_quiet(["rsync", "-az", f"{args.host_ssh}:{remote}/out/adapter.enc",
               f"{args.host_ssh}:{remote}/out/private_report.enc", f"{out}/"])
    adapter_ct = (out / "adapter.enc").read_bytes()
    private_ct = (out / "private_report.enc").read_bytes()
    checks = req.verify_report(report, (out / "report.sig").read_text(), adapter_ct,
                               private_ct, expected_measurement)
    checks["matches_chain"] = (
        onchain.report_hash == bc.sha256(bc.canonical_json(report))
        and onchain.outcome == report["outcome"]
        and onchain.work_bps == report["work_bps"]
        and onchain.adapter_ct_hash.hex() == report["adapter_ct_hash"])
    log(f"result on chain: outcome={onchain.outcome} work_bps={onchain.work_bps} (no scores on "
        f"chain); checks {'all OK' if all(checks.values()) else checks}")
    if not all(checks.values()):
        log("result does not verify; not accepting (it settles on its own after the delay)")
        return 1
    private = req.open_private_report(private_ct)
    qa, tr = private["personal_qa"], private["training"]
    log(f"sealed private report: personal_qa {qa['before']:.2f} -> {qa['after']:.2f}, "
        f"regression loss {private['regression_loss']['delta']:+.3f}, {tr['steps']} steps")

    # --- 7. Open the adapter and settle ---------------------------------------
    if report["passed"]:
        names = req.open_adapter(report, adapter_ct, job_dir)
        ok = sorted(load_stored_adapter(storage_key, job_dir / "adapter.store", job_id)) == names
        log(f"adapter decrypted, hash verified, stored encrypted at rest (reload OK: {ok}): {names}")
    sig = chain.accept(req_kp, onchain)
    log(f"accept tx {sig[:20]}...  job closed: {chain.get_job(job_id) is None}; requester "
        f"balance {chain.token_balance(req_kp.pubkey()):.2f}, host "
        f"{chain.token_balance(onchain.host):.2f} test-USDC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
