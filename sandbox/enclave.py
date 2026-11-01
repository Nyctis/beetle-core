"""Entry point of the sealed sandbox (local Docker mock of a T2 enclave).

Runs one job end to end. Step numbers are those of the v0.1 protocol:

  4. self-check that there is no network, generate one-time keys, publish an
     attestation bound to the job
  5. wait for the requester's sealed data key (DEK), delivered by the host
  6. verify and decrypt the inputs, run baseline eval, train (built-in recipe
     or the requester's own program), evaluate
  7. encrypt the adapter and seal its key to the requester only if it passed;
     seal the detailed results to the requester; sign a minimal public report

The only I/O is files: /job/in (read-only, written by the host), /model
(read-only base model) and /job/out. The container runs with --network none.

What the host sees from this process: the attestation, the public report
(pass/fail and hashes), ciphertexts, and stdout. Stdout carries phase names
only, never numbers derived from the data, and unexpected errors are reported
by exception type only, since exception messages can quote data.

Everything that limits a job comes from three places: job_defaults.py, which is
measured with this code; the public spec; and the terms in job.json. The host
writes job.json, but the program compares those terms with the job when it
checks the signature here, so lying about them only loses the host the job.
"""

from __future__ import annotations

import json
import signal
import socket
import sys
import tarfile
import time
from pathlib import Path

import beetle_crypto as bc
import job_defaults as jd
from beetle_archive import read_tar

APP = Path("/app")
INBOX = Path("/job/in")
OUTBOX = Path("/job/out")
MODEL = Path("/model")
JOB_INPUTS = ("body", "train", "eval", "program")
TERMS = ("units", "price_mode", "run_secs", "grace_s")


def phase(msg: str) -> None:
    print(f"[enclave] {msg}", flush=True)


class JobFailed(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class OutOfTime(BaseException):
    """Not an Exception, so nothing between here and main() can swallow it."""


def network_check() -> dict:
    """The sandbox refuses to run if it can reach anything outside.

    The kernel creates placeholder tunnel devices (gre0, sit0, ...) in every
    network namespace; they are down and have no address, so only interfaces
    that are up count. Loopback is the only one allowed.
    """
    up = []
    for p in sorted(Path("/sys/class/net").iterdir()):
        if p.is_dir() and p.name != "lo" and int((p / "flags").read_text(), 16) & 0x1:
            up.append(p.name)
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
        tcp_blocked = False
    except OSError:
        tcp_blocked = True
    try:
        socket.getaddrinfo("huggingface.co", 443)
        dns_blocked = False
    except OSError:
        dns_blocked = True
    return {"interfaces_up": up, "tcp_blocked": tcp_blocked, "dns_blocked": dns_blocked,
            "isolated": not up and tcp_blocked and dns_blocked}


def measure_self() -> str:
    files = {p.name: p.read_bytes() for p in APP.glob("*.py")}
    files["requirements.txt"] = (APP / "requirements.txt").read_bytes()
    files.update({f"sdk/{p.name}": p.read_bytes() for p in (APP / "sdk").glob("*.py")})
    return bc.code_measurement(files)


def job_terms(job: dict, spec: dict) -> dict:
    """What the job paid for. An offline run has no chain and no order: it takes
    the spec's budget as the order."""
    if all(k in job for k in TERMS):
        return {k: job[k] for k in TERMS}
    if spec.get("chain"):
        raise JobFailed("job_terms_missing")
    wall = spec["budget"]["max_wall_seconds"]
    return {"units": spec["budget"]["max_train_tokens"], "price_mode": jd.PRICE_BY_TOKEN,
            "run_secs": wall + jd.RUN_RESERVE_S, "grace_s": jd.RUN_RESERVE_S}


def stop_at_deadline(terms: dict) -> float:
    """A result only counts if the seller can land it before the chain's deadline,
    which starts with the key. So nothing runs past that, whatever is running:
    the sandbox reports instead of leaving the seller to the timeout's slash.
    Returns when, on the monotonic clock."""
    def out_of_time(*_):
        raise OutOfTime

    left = max(terms["run_secs"] - jd.SUBMIT_RESERVE_S, 1)
    signal.signal(signal.SIGALRM, out_of_time)
    signal.setitimer(signal.ITIMER_REAL, left)
    return time.monotonic() + left


def stop_timer() -> None:
    signal.setitimer(signal.ITIMER_REAL, 0)


class Enclave:
    def __init__(self):
        self.sign_key = bc.new_signing_key()   # signs attestation, report, settlement
        self.enc_key = bc.HpkeKeyPair()        # receives the DEK
        self.sign_pk = bc.public_bytes(self.sign_key)
        self.sealed_dek_hash = None            # set once the host delivers it

    def sign(self, type_name: str, obj: dict) -> str:
        return self.sign_key.sign(bc.digest(type_name, bc.canonical_json(obj))).hex()

    def write_signed(self, name: str, type_name: str, obj: dict) -> None:
        (OUTBOX / f"{name}.json").write_text(json.dumps(obj, indent=2))
        (OUTBOX / f"{name}.sig").write_text(self.sign(type_name, obj))


def wait_for_dek(enclave: Enclave, job_id: bytes) -> bytes:
    sealed_path = INBOX / "sealed_dek.bin"
    deadline = time.time() + jd.DEK_WAIT_SECONDS
    while not sealed_path.exists():  # the host renames it into place atomically
        if time.time() > deadline:
            raise JobFailed("dek_timeout")
        time.sleep(0.5)
    sealed = sealed_path.read_bytes()
    enclave.sealed_dek_hash = bc.sha256(sealed).hex()  # the chain compares it with what it holds
    try:
        return enclave.enc_key.open(sealed, info=b"dek" + job_id)
    except Exception as e:
        raise JobFailed("dek_open_failed", type(e).__name__)


def open_inputs(spec: dict, dek: bytes, job_id: bytes) -> dict[str, bytes]:
    out = {}
    for name in JOB_INPUTS:
        path = INBOX / f"{name}.enc"
        if not path.exists() or bc.file_sha256(path) != spec["inputs"][name]:
            raise JobFailed(f"{name}_ciphertext_hash_mismatch")
        try:
            out[name] = bc.open_blob(dek, path.read_bytes(), job_id + name.encode())
        except Exception as e:
            raise JobFailed("data_decrypt_failed", type(e).__name__)
    return out


def read_inputs(plain: dict[str, bytes], terms: dict) -> tuple[dict, list, list, dict]:
    """Everything a job's inputs can get wrong is found here, before any work, so
    that it's the requester's fault and not a crash after hours of training."""
    if (len(plain["train"]) > jd.train_bytes_limit(terms)
            or len(plain["program"]) > jd.MAX_PROGRAM_BYTES):
        raise JobFailed("malformed_inputs")
    try:
        body = json.loads(plain["body"])
        memories = json.loads(plain["train"])
        eval_items = json.loads(plain["eval"])
    except (ValueError, RecursionError):
        raise JobFailed("malformed_inputs")
    problem = jd.inputs_problem(body, memories, eval_items)
    if problem:
        raise JobFailed(problem)
    program_files = {}
    if body["mode"] == "program":
        if bc.sha256(plain["program"]).hex() != body["program"]["bundle_sha256"]:
            raise JobFailed("program_hash_mismatch")
        try:
            program_files = read_tar(plain["program"], jd.MAX_PROGRAM_BYTES, jd.MAX_PROGRAM_FILES)
        except (tarfile.TarError, ValueError):
            raise JobFailed("malformed_inputs")
        if (body["program"]["entry"] not in program_files
                or any(len(name.encode()) > 255 for name in program_files)):  # no such file name
            raise JobFailed("malformed_inputs")
    return body, memories, eval_items, program_files


def seal_results(enclave: Enclave, job_id: bytes, spec: dict, terms: dict, spec_hash: bytes,
                 measurement: str, model, stats: dict, adapter_log: list[str], body: dict,
                 scores: dict, outcome: int, work_bps: int) -> None:
    import train_eval as te

    passed = outcome == jd.OUTCOME_PASSED
    bundle = te.adapter_bundle(model)
    adapter_key = bc.new_key()
    adapter_ct = bc.seal_blob(adapter_key, bundle, job_id + b"adapter", bc.adapter_bucket(len(bundle)))
    (OUTBOX / "adapter.enc").write_bytes(adapter_ct)
    requester_pk = bytes.fromhex(spec["requester_enc_pk"])
    sealed_adapter_key = (bc.hpke_seal(requester_pk, adapter_key, info=b"adapter" + job_id).hex()
                          if passed else None)

    # A program's outputs are what it computed with the seller's GPU, so they're
    # part of what's paid for: a run that didn't pass gets counters, not content.
    program = body["mode"] == "program"
    shown = passed or not program
    private = {  # only the requester can read this
        "mode": body["mode"],
        "passed": passed,
        **scores,
        "training": {k: stats[k] for k in ("steps", "train_tokens", "wall_seconds")}
        | {"loss_log": stats["loss_log"] if shown else []},
        "log": (stats["log"] if program else adapter_log) if shown else [],
        "program": {"exit_code": stats["exit_code"], "stop_reason": stats["stop_reason"],
                    "output": stats["output"] if shown else ""} if program else None,
    }
    private_ct = bc.hpke_seal_blob(requester_pk, bc.canonical_json(private),
                                   info=b"private-report" + job_id, bucket=bc.REPORT_BUCKET)
    (OUTBOX / "private_report.enc").write_bytes(private_ct)
    sign_result(enclave, job_id, terms, {
        "spec_hash": spec_hash.hex(),
        "measurement": measurement,
        "passed": passed,
        "outcome": outcome,
        "work_bps": work_bps,
        "adapter_ct_hash": bc.sha256(adapter_ct).hex(),
        "adapter_pt_hash": bc.sha256(bundle).hex(),
        "checkpoint_root": stats["checkpoint_root"].hex(),
        "private_report_ct_hash": bc.sha256(private_ct).hex(),
        "sealed_adapter_key": sealed_adapter_key,
    })


def sign_result(enclave: Enclave, job_id: bytes, terms: dict, fields: dict) -> None:
    """report.json has the sealed adapter key in it, so the host must not hand it
    over before submit_result lands: until then a buyer could keep the key and
    still claim the timeout. Nothing here is signed past the deadline for that
    reason too (stop_at_deadline)."""
    report = {"job_id": job_id.hex(), "sealed_dek_hash": enclave.sealed_dek_hash, **terms, **fields}
    enclave.write_signed("report", "report", report)
    settle_msg = bc.settle_message(job_id, report, bc.sha256(bc.canonical_json(report)))
    (OUTBOX / "settle.sig").write_text(enclave.sign_key.sign(settle_msg).hex())


def run(enclave: Enclave, job_id: bytes, spec: dict, terms: dict, spec_hash: bytes,
        measurement: str) -> None:
    dek = wait_for_dek(enclave, job_id)
    stop_at = stop_at_deadline(terms)
    phase("data key opened")
    plain = open_inputs(spec, dek, job_id)
    body, memories, eval_items, program_files = read_inputs(plain, terms)
    if bc.dir_manifest_sha256(MODEL) != spec["base_model"]["manifest_sha256"]:
        raise JobFailed("base_model_hash_mismatch")
    phase("inputs verified")  # not the mode: both modes look alike to the host

    import torch  # heavy imports only after the checks pass
    import train_eval as te

    torch.set_num_threads(max(1, torch.get_num_threads()))
    acc_rule, recipe = body["acceptance"], body["recipe"]
    tok, model = te.load_base(MODEL)
    max_len = recipe["max_seq_len"]
    base_acc = te.qa_accuracy(model, tok, eval_items)
    base_reg = te.answer_loss(model, tok, te.REGRESSION_SET, max_len)
    if base_acc >= 1 and acc_rule["personal_qa_min_gain"] > 0:
        raise JobFailed("baseline_already_passes")
    phase("baseline evaluated")

    private_log: list[str] = []
    phase("training")
    by_token = terms["price_mode"] == jd.PRICE_BY_TOKEN
    tokens = spec["budget"]["max_train_tokens"]  # within what was paid: see spec_problem
    wall = spec["budget"]["max_wall_seconds"]
    limit = max(min(wall, int(stop_at - time.monotonic()) - jd.EVAL_RESERVE_S), 1)  # time to evaluate
    try:
        if body["mode"] == "recipe":
            pairs = te.augment(memories)
            planned = sum(len(te.encode(tok, q, a, max_len)[0]) for q, a in pairs) * recipe["optim"]["epochs"]
            planned = min(planned, tokens)
            model, stats = te.train_lora(model, tok, pairs, recipe,
                                         {"max_train_tokens": tokens, "max_wall_seconds": limit},
                                         private_log.append)
            done, fault = stats["train_tokens"], False
        else:
            import program_runner as pr

            session = pr.Session(model, tok, recipe, memories, tokens)
            stats = pr.run_program(session, program_files, body["program"]["entry"], limit)
            model = session.model
            # What the buyer paid for, not what it wrote in its budget.
            planned = terms["units"]
            done = stats["train_tokens"] if by_token else int(stats["wall_seconds"])
            fault = stats["stop_reason"] in pr.PROGRAM_FAULTS or (
                stats["stop_reason"] is None and stats["exit_code"] != 0)
    except te.BadRecipe:
        raise JobFailed("malformed_inputs")
    phase("training finished")

    acc = te.qa_accuracy(model, tok, eval_items)
    reg = te.answer_loss(model, tok, te.REGRESSION_SET, max_len)
    required = jd.required_gain(acc_rule, base_acc)
    gain, reg_delta = acc - base_acc, reg - base_reg
    passed = gain >= required and reg_delta <= acc_rule["regression_max_loss_increase"]
    phase("evaluated")
    # The host sets the machine's speed, but not a limit below what the job paid it to take.
    fault = fault or (stats["wall_seconds"] > limit and limit == wall < terms["run_secs"] - terms["grace_s"])
    outcome, work_bps = jd.outcome_for_run(passed, done, planned, stats["wall_seconds"], limit, fault)

    # --- Step 7: seal outputs and sign the report ----------------------------
    stop_timer()  # past this point the result exists, and the seller can still land it
    scores = {
        "personal_qa": {"before": base_acc, "after": acc, "gain": gain, "required_gain": required},
        "regression_loss": {"before": base_reg, "after": reg, "delta": reg_delta,
                            "max_increase": acc_rule["regression_max_loss_increase"]},
    }
    seal_results(enclave, job_id, spec, terms, spec_hash, measurement, model, stats, private_log,
                 body, scores, outcome, work_bps)
    phase("report signed")

    # Best effort: Python cannot guarantee memory zeroization. The real enclave
    # relies on the VM (and its memory) being destroyed after the job.
    del dek, plain, memories, eval_items


def sign_failure(enclave: Enclave, job_id: bytes, terms: dict, spec_hash: bytes,
                 measurement: str, reason: str) -> None:
    """A result the chain can settle by, so that a job that stopped here doesn't
    have to wait for a timeout. Without the key on chain there is none to bind."""
    sign_result(enclave, job_id, terms, {
        "spec_hash": spec_hash.hex(),
        "measurement": measurement,
        "passed": False,
        "outcome": jd.outcome_for_error(reason),
        "work_bps": 0,
        "adapter_ct_hash": (bytes(32)).hex(),
        "adapter_pt_hash": None,
        "checkpoint_root": None,
        "private_report_ct_hash": None,
        "sealed_adapter_key": None,
    })


def main() -> int:
    OUTBOX.mkdir(parents=True, exist_ok=True)
    enclave = Enclave()
    job = json.loads((INBOX / "job.json").read_text())
    job_id = bytes.fromhex(job["job_id"])
    spec, terms, spec_hash, measurement = None, None, None, None
    try:
        net = network_check()
        phase(f"network isolated: {net['isolated']}")
        if not net["isolated"]:
            raise JobFailed("network_not_isolated", json.dumps(net))
        spec = json.loads((INBOX / "spec.json").read_text())
        spec_hash = bc.sha256(bc.canonical_json(spec))
        if spec_hash.hex() != job["spec_hash"]:
            raise JobFailed("spec_hash_mismatch")
        bc.configure_from_spec(spec)  # chain jobs bind digests to the program id
        terms = job_terms(job, spec)
        problem = jd.spec_problem(spec, terms)
        if problem:
            raise JobFailed(problem)
        try:  # a key that can't be sealed to would only show after the training
            bc.hpke_seal(bytes.fromhex(spec["requester_enc_pk"]), bytes(32), info=b"check")
        except Exception:
            raise JobFailed("spec_malformed")

        # --- Step 4: attestation bound to this job ----------------------------
        measurement = measure_self()
        attestation = {
            "tier": "MOCK",
            "mock": True,  # no hardware root of trust in the local prototype
            "measurement": measurement,
            "job_id": job["job_id"],
            "spec_hash": spec_hash.hex(),
            "sign_pk": enclave.sign_pk.hex(),
            "enc_pk": enclave.enc_key.public_bytes.hex(),
            "report_data": bc.digest("start", job_id + spec_hash + enclave.sign_pk
                                     + enclave.enc_key.public_bytes).hex(),
            "network": net,
        }
        enclave.write_signed("attestation", "attest", attestation)
        phase("attestation published")
        run(enclave, job_id, spec, terms, spec_hash, measurement)
        return 0
    except JobFailed as e:
        failure = {"job_id": job["job_id"], "reason": e.reason, "detail": e.detail}
    except OutOfTime:
        failure = {"job_id": job["job_id"], "reason": "run_budget_exhausted", "detail": ""}
    except Exception as e:  # never print a traceback: messages can quote data
        failure = {"job_id": job["job_id"], "reason": "internal_error", "detail": type(e).__name__}
    stop_timer()
    phase(f"failed: {failure['reason']}")
    enclave.write_signed("error", "error", failure)
    if enclave.sealed_dek_hash:
        sign_failure(enclave, job_id, terms, spec_hash, measurement, failure["reason"])
    return 3


if __name__ == "__main__":
    sys.exit(main())
