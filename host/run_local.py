"""Run one job on this machine: requester -> host -> offline Docker sandbox -> requester.

The host only moves files and starts the container; it never holds a key.
The on-chain steps are written to a local ledger file (no rule checks; the
Solana version is host/host_daemon.py plus requester/run_requester.py).

After every job it scans everything the host can see (in/, out/, the sandbox
log, the ledger) for any plaintext memory, hidden eval question or canary.

Usage (on a machine with Docker, from the beetle directory):
    uv run python host/run_local.py --scenario honest          # built-in recipe
    uv run python host/run_local.py --scenario program         # requester's own train.py
    uv run python host/run_local.py --scenario program-attack  # hostile train.py
    uv run python host/run_local.py --scenario tamper-model
    uv run python host/run_local.py --scenario tamper-code
    uv run python host/run_local.py --scenario wrong-dek
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "requester"), str(ROOT / "host")]

import beetle_crypto as bc  # noqa: E402
from job_defaults import (ACCEPTANCE, BOND_RATIO_BPS, BUDGET, COMPUTE_FLOOR_BPS,  # noqa: E402
                          LATE_SLASH_BPS, MODEL_DIR, MODEL_REPO, PRICE, PROTOCOL_FEE_BPS,
                          REQUESTER_FAULTS, sandbox_code_files)
from model_profiles import get as get_profile  # noqa: E402
from requester import (AttestationRejected, Requester, load_or_create_storage_key,  # noqa: E402
                       load_stored_adapter)
from sandbox_host import ensure_image, ensure_model, sandbox_cmd, wait_for  # noqa: E402

SCENARIOS = ["honest", "program", "program-attack", "tamper-model", "tamper-code", "wrong-dek"]
PROGRAMS = {"program": ROOT / "demo/program", "program-attack": ROOT / "demo/program_attack"}
EXPECTED_ATTACKS = 8  # attempts in demo/program_attack/train.py
# The hostile program barely trains, and a job that misses its bar returns no output.
ATTACK_ACCEPTANCE = {**ACCEPTANCE, "personal_qa_min_gain": 0}


class Ledger:
    """Stand-in for the Solana program: an append-only log of the steps."""

    def __init__(self, path: Path):
        self.path = path

    def write(self, ix: str, **fields) -> None:
        entry = {"ts": round(time.time(), 1), "instruction": ix, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[chain]   {ix}: " + ", ".join(f"{k}={v}" for k, v in fields.items()))


def usdc(amount: int) -> str:
    return f"{amount / 1e6:.2f} USDC"


def entropy(data: bytes) -> float:
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    return -sum(c / len(data) * math.log2(c / len(data)) for c in counts if c)


def leak_scan(job_dir: Path, secrets: list[str]) -> list[str]:
    """Problems in the files the host can see.

    *.enc files must look like ciphertext (near 8 bits/byte of entropy);
    searching random bytes for short strings only produces chance matches.
    Every other file must not contain any secret string (case-insensitive).
    """
    problems = []
    for p in job_dir.rglob("*"):
        if not p.is_file() or p.name == "adapter.store":  # requester-side, encrypted
            continue
        if "model_tampered" in p.parts:  # a copy of the public base model
            continue
        data = p.read_bytes()
        if p.suffix == ".enc":
            if entropy(data) < 7.99:
                problems.append(f"{p.relative_to(job_dir)}: not ciphertext-like")
            continue
        low = data.lower()
        problems += [f"{p.relative_to(job_dir)}: {s!r}" for s in secrets if s.lower().encode() in low]
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="honest", choices=SCENARIOS)
    args = ap.parse_args()

    print(f"=== scenario: {args.scenario} ===")
    base_model = ensure_model()
    image_id = ensure_image()
    expected_measurement = bc.code_measurement(sandbox_code_files())
    print(f"[host]    base model {MODEL_REPO}@{base_model['revision'][:10]}, "
          f"image {image_id[7:19]}, expected measurement {expected_measurement[:16]}...")

    # --- Requester prepares the job; the job is opened on chain -------------
    storage_key = load_or_create_storage_key(ROOT / ".keys/storage.key")
    req = Requester(storage_key)
    memories = json.loads((ROOT / "demo/memories.json").read_text())
    eval_items = json.loads((ROOT / "demo/eval_hidden.json").read_text())
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job_dir = ROOT / "jobs" / f"{stamp}-{args.scenario}"
    job = req.prepare_job(job_dir, memories, eval_items, base_model, get_profile(MODEL_REPO).recipe(),
                          ATTACK_ACCEPTANCE if args.scenario == "program-attack" else ACCEPTANCE, BUDGET,
                          program_dir=PROGRAMS.get(args.scenario))
    secrets = ([m["answer"] for m in memories] + [c["answer"] for c in req.canaries]
               + [e["question"] for e in eval_items])
    inbox, outbox = job_dir / "in", job_dir / "out"
    outbox.mkdir(parents=True, exist_ok=True)
    outbox.chmod(0o777)
    print("[agent]   encrypted inputs: " + ", ".join(
        f"{p.name} {p.stat().st_size} B" for p in sorted(inbox.glob("*.enc"))))
    ledger = Ledger(job_dir / "ledger.jsonl")
    bond = PRICE * BOND_RATIO_BPS // 10_000
    ledger.write("open_job", job_id=job["job_id"][:16], spec_hash=job["spec_hash"][:16],
                 escrow=usdc(PRICE), host_bond_locked=usdc(bond))

    # --- Host starts the sandbox ---------------------------------------------
    model_mount, extra = MODEL_DIR, []
    if args.scenario == "tamper-model":
        model_mount = job_dir / "model_tampered"
        shutil.copytree(MODEL_DIR, model_mount, ignore=shutil.ignore_patterns(".*"))
        cfg = json.loads((model_mount / "config.json").read_text())
        cfg["num_hidden_layers"] -= 1   # a "cheaper" model
        (model_mount / "config.json").write_text(json.dumps(cfg))
    if args.scenario == "tamper-code":
        tampered = job_dir / "train_eval_tampered.py"
        tampered.write_text((ROOT / "sandbox/train_eval.py").read_text()
                            + "\n# host-injected code: copy decrypted data to /job/out\n")
        extra = ["-v", f"{tampered}:/app/train_eval.py:ro"]

    container = f"beetle-{stamp}"
    subprocess.run(sandbox_cmd(container, model_mount, inbox, outbox, extra),
                   check=True, capture_output=True)
    print("[host]    sandbox started (--network none, read-only root, no capabilities)")

    def finish(ret: int) -> int:
        logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
        (job_dir / "sandbox.log").write_text(logs.stdout + logs.stderr)
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        hits = leak_scan(job_dir, secrets)
        n_files = sum(1 for p in job_dir.rglob("*") if p.is_file() and p.name != "adapter.store"
                      and "model_tampered" not in p.parts)
        n_enc = sum(1 for p in job_dir.rglob("*.enc"))
        print(f"[check]   leak scan of {n_files} host-visible files ({n_enc} ciphertexts checked for "
              f"entropy, the rest for {len(secrets)} secret strings): "
              + ("CLEAN" if not hits else f"PROBLEMS {hits}"))
        print(f"[host]    job files in {job_dir.relative_to(ROOT)}")
        return 2 if hits else ret

    def refund_and_slash(reason: str) -> int:
        slash = bond * LATE_SLASH_BPS // 10_000
        ledger.write("timeout", reason=reason, refund_to_requester=usdc(PRICE),
                     host_slashed=usdc(slash))
        return finish(1)

    # --- Attestation; the requester releases the key only if it checks out ---
    got = wait_for([outbox / "attestation.json", outbox / "error.json"], container, 120)
    if got is None or got.name == "error.json":
        reason = json.loads(got.read_text())["reason"] if got else "no output"
        print(f"[host]    sandbox did not attest: {reason}")
        return refund_and_slash("no_attestation")
    att = json.loads(got.read_text())
    ledger.write("start", enclave_sign_pk=att["sign_pk"][:16], measurement=att["measurement"][:16])
    try:
        req.check_attestation(att, (outbox / "attestation.sig").read_text(), expected_measurement)
    except AttestationRejected as e:
        print(f"[agent]   REFUSED to release the key: {e}")
        return refund_and_slash("attestation_rejected")
    print("[agent]   attestation OK (signature, job binding, measurement, network isolated)")
    sealed = req.seal_dek(wrong_key=args.scenario == "wrong-dek")
    tmp = inbox / ".sealed_dek.tmp"
    tmp.write_bytes(sealed)
    tmp.rename(inbox / "sealed_dek.bin")  # atomic: the sandbox sees all or nothing
    ledger.write("release_key", sealed_dek=f"{len(sealed)} bytes")

    # --- Training happens in the sandbox --------------------------------------
    print("[host]    waiting for the sandbox to train and evaluate...")
    code = int(subprocess.run(["docker", "wait", container], capture_output=True,
                              text=True).stdout.strip() or -1)
    if (outbox / "error.json").exists():
        err = json.loads((outbox / "error.json").read_text())
        signed = bc.verify(req.attested_sign_pk, bytes.fromhex((outbox / "error.sig").read_text()),
                           bc.digest("error", bc.canonical_json(err)))
        print(f"[agent]   sandbox aborted: {err['reason']} (signed by attested key: {signed})")
        if signed and err["reason"] in REQUESTER_FAULTS:
            floor = PRICE * COMPUTE_FLOOR_BPS // 10_000
            ledger.write("finalize", outcome="requester_fault", reason=err["reason"],
                         host_paid=usdc(floor), refund_to_requester=usdc(PRICE - floor),
                         bond_released=usdc(bond))
            return finish(1)
        return refund_and_slash(err["reason"])
    if code != 0 or not (outbox / "report.json").exists():
        return refund_and_slash(f"sandbox_exit_{code}")

    report = json.loads((outbox / "report.json").read_text())
    adapter_ct = (outbox / "adapter.enc").read_bytes()
    private_ct = (outbox / "private_report.enc").read_bytes()
    ledger.write("submit_result", passed=report["passed"],
                 adapter_ct_hash=report["adapter_ct_hash"][:16])

    # --- Requester verifies, reads its sealed report, settles ----------------
    checks = req.verify_report(report, (outbox / "report.sig").read_text(), adapter_ct,
                               private_ct, expected_measurement)
    print(f"[agent]   public report checks: {'all OK' if all(checks.values()) else checks}")
    if not all(checks.values()):
        ledger.write("challenge", reason="report_check_failed")
        return finish(1)
    private = req.open_private_report(private_ct)
    qa, reg, tr = private["personal_qa"], private["regression_loss"], private["training"]
    print(f"[agent]   private report (only the requester can read it): personal_qa "
          f"{qa['before']:.2f} -> {qa['after']:.2f}, regression loss {reg['delta']:+.3f}, "
          f"{tr['steps']} steps, {tr['train_tokens']} tokens, {tr['wall_seconds']}s")

    ret = 0
    if private["program"]:
        p = private["program"]
        first_line = p["output"].splitlines()[0] if p["output"] else "(no output)"
        print(f"[agent]   program exit {p['exit_code']}, stop reason {p['stop_reason']}; {first_line}")
        for line in private["log"]:
            print(f"[agent]     program log: {line}")
    if args.scenario == "program-attack":
        blocked = [l for l in private["log"] if ": blocked" in l]
        allowed = [l for l in private["log"] if ": ALLOWED" in l]
        leaked_file = (outbox / "leak.txt").exists()
        ok = len(blocked) == EXPECTED_ATTACKS and not allowed and not leaked_file
        print(f"[check]   attack attempts blocked: {len(blocked)}/{EXPECTED_ATTACKS}, allowed: "
              f"{len(allowed)}, leak.txt in out/: {leaked_file} -> {'PASS' if ok else 'FAIL'}")
        ret = 0 if ok else 2

    if report["passed"]:
        names = req.open_adapter(report, adapter_ct, job_dir)
        reloaded = load_stored_adapter(storage_key, job_dir / "adapter.store", req.job_id)
        print(f"[agent]   adapter decrypted, re-encrypted at rest as adapter.store; "
              f"reloads in memory: {sorted(reloaded) == names} {names}")
        fee = PRICE * PROTOCOL_FEE_BPS // 10_000
        ledger.write("finalize", outcome="passed", host_paid=usdc(PRICE - fee),
                     protocol_fee=usdc(fee), bond_released=usdc(bond))
    else:
        floor = PRICE * COMPUTE_FLOOR_BPS // 10_000
        print("[agent]   failed acceptance: no adapter key was released")
        ledger.write("finalize", outcome="failed", host_paid=usdc(floor),
                     refund_to_requester=usdc(PRICE - floor), bond_released=usdc(bond))
    return finish(ret)


if __name__ == "__main__":
    sys.exit(main())
