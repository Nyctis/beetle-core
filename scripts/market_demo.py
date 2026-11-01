"""One round of the market on a local chain, end to end, with no attester.

    scripts/setup_chain.sh local
    uv run python scripts/market_demo.py

A seller registers, stakes and posts an offer. A buyer finds it with an account
filter, takes it, verifies the sandbox's evidence on its own machine and
releases the data key. One training job runs, and because this tier has no
hardware behind it, the buyer's accept is what pays the full price. Then the
buyer takes an offer from a seller that never shows up, and that seller is
slashed. Last,
a chat session pays per token through a Solana Foundation payment channel,
if chain.json names that program.

The chain, the job encryption and the signatures are the real ones. The sandbox
is the part that's simulated: DemoSandbox below runs in this process and trains
a small numpy model instead of a LoRA, with MOCK evidence. It writes the report
format the chain requires, so it's also the reference for sandbox/enclave.py.
Everyone is a throwaway key; the test-USDC comes from the mint set up by
setup_chain.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "requester"), str(ROOT / "host"),
                str(ROOT / "attester"), str(ROOT / "sandbox")]

import beetle_crypto as bc  # noqa: E402
from beetle_archive import tar_bytes  # noqa: E402
from beetle_chain import (KIND_INFERENCE, KIND_TRAINING, OFFER_PAUSED, Chain,  # noqa: E402
                          OfferTerms, evidence_hash, load_keypair, session_salt, sign_voucher)
from job_defaults import (ACCEPTANCE, RUN_RESERVE_S, TIERS, fit_budget, outcome_for_run,  # noqa: E402
                          sandbox_code_files, spec_problem)
from recipe_data import TRAIN_TEMPLATES  # noqa: E402
from requester import AttestationRejected, Requester  # noqa: E402
from server import dev_policy  # noqa: E402
from solders.keypair import Keypair  # noqa: E402
from verifier import EvidenceVerifier  # noqa: E402

USDC = 1_000_000
UNITS = 2_000                # train tokens ordered
PRICE_PER_MILLION = 500 * USDC
STAKE = 50 * USDC
FEATURES, EPOCHS, LR = 512, 40, 0.5


def log(who: str, msg: str) -> None:
    print(f"[{who:7}] {msg}", flush=True)


def tokens(text: str) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in text.lower()).split()


def features(text: str) -> np.ndarray:
    """Hashed bag of words."""
    x = np.zeros(FEATURES)
    for t in tokens(text):
        x[int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "little") % FEATURES] = 1.0
    return x


class TinyModel:
    """Softmax regression from a question to one of the known answers."""

    def __init__(self, answers: list[str]):
        self.answers = answers
        self.w = np.zeros((FEATURES, len(answers)))

    def predict(self, question: str) -> str:
        return self.answers[int(np.argmax(features(question) @ self.w))]

    def step(self, question: str, answer: str) -> None:
        x = features(question)
        z = x @ self.w
        p = np.exp(z - z.max())
        p /= p.sum()
        y = np.zeros(len(self.answers))
        y[self.answers.index(answer)] = 1.0
        self.w += LR * np.outer(x, y - p)

    def accuracy(self, items: list[dict]) -> float:
        return sum(self.predict(e["question"]) == e["answer"] for e in items) / len(items)


class DemoSandbox:
    """What sandbox/enclave.py does around training, minus the container."""

    def __init__(self, job_dir: Path, job: dict, spec: dict, measurement: str):
        self.job_dir, self.job_id = job_dir, bytes.fromhex(job["job_id"])
        self.terms = {k: job[k] for k in ("units", "price_mode", "run_secs", "grace_s")}
        self.spec, self.spec_hash, self.measurement = spec, bytes.fromhex(job["spec_hash"]), measurement
        self.sign_key = bc.new_signing_key()
        self.enc_key = bc.HpkeKeyPair()
        self.sign_pk = bc.public_bytes(self.sign_key)
        self.enc_pk = self.enc_key.public_bytes

    def evidence(self) -> dict:
        """MOCK evidence: self-signed, proves nothing, bound to this job."""
        att = {"tier": "MOCK", "mock": True, "measurement": self.measurement,
               "job_id": self.job_id.hex(), "spec_hash": self.spec_hash.hex(),
               "sign_pk": self.sign_pk.hex(), "enc_pk": self.enc_pk.hex(),
               "report_data": bc.digest("start", self.job_id + self.spec_hash + self.sign_pk
                                        + self.enc_pk).hex(),
               "network": {"isolated": True}}
        sig = self.sign_key.sign(bc.digest("attest", bc.canonical_json(att))).hex()
        return {"type": "MOCK", "attestation": att, "signature": sig}

    def train(self, chain: Chain, sealed_dek: bytes) -> tuple[dict, str, bytes, bytes, bytes]:
        """Returns the report, its signature, the settle signature and the two ciphertexts."""
        dek = self.enc_key.open(sealed_dek, info=b"dek" + self.job_id)
        plain = {}
        for name in ("body", "train", "eval"):
            ct = (self.job_dir / "in" / f"{name}.enc").read_bytes()
            assert bc.sha256(ct).hex() == self.spec["inputs"][name]
            plain[name] = bc.open_blob(dek, ct, self.job_id + name.encode())
        body = json.loads(plain["body"])
        memories, eval_items = json.loads(plain["train"]), json.loads(plain["eval"])

        answers = sorted({m["answer"] for m in memories})
        pairs = [(t.format(topic=m["topic"]), m["answer"]) for m in memories for t in TRAIN_TEMPLATES]
        planned = sum(len(tokens(q)) for q, _ in pairs) * EPOCHS
        model = TinyModel(answers)
        before = model.accuracy(eval_items)
        trained, t0 = 0, time.time()
        for _, (q, a) in itertools.product(range(EPOCHS), pairs):
            if trained + len(tokens(q)) > self.spec["budget"]["max_train_tokens"]:
                break  # never past what was paid
            model.step(q, a)
            trained += len(tokens(q))
        after = model.accuracy(eval_items)
        wall = time.time() - t0
        passed = after - before >= body["acceptance"]["personal_qa_min_gain"]
        outcome, work = outcome_for_run(passed, trained, min(planned, self.spec["budget"]["max_train_tokens"]),
                                        wall, self.spec["budget"]["max_wall_seconds"])

        bundle = tar_bytes({"weights.npy": model.w.tobytes(), "answers.json": bc.canonical_json(answers)})
        adapter_key = bc.new_key()
        adapter_ct = bc.seal_blob(adapter_key, bundle, self.job_id + b"adapter", bc.ADAPTER_BUCKET)
        requester_pk = bytes.fromhex(self.spec["requester_enc_pk"])
        private = {"mode": "recipe", "passed": passed,
                   "personal_qa": {"before": before, "after": after},
                   "training": {"train_tokens": trained, "planned_tokens": planned,
                                "wall_seconds": round(wall, 3)}}
        private_ct = bc.hpke_seal_blob(requester_pk, bc.canonical_json(private),
                                       info=b"private-report" + self.job_id)
        report = {"job_id": self.job_id.hex(), "spec_hash": self.spec_hash.hex(),
                  "measurement": self.measurement, "passed": passed,
                  "outcome": outcome, "work_bps": work,
                  "sealed_dek_hash": bc.sha256(sealed_dek).hex(),
                  "adapter_ct_hash": bc.sha256(adapter_ct).hex(),
                  "adapter_pt_hash": bc.sha256(bundle).hex(),
                  "checkpoint_root": bc.sha256(model.w.tobytes()).hex(),
                  "private_report_ct_hash": bc.sha256(private_ct).hex(),
                  "sealed_adapter_key": bc.hpke_seal(requester_pk, adapter_key, info=b"adapter" + self.job_id).hex()
                  if passed else None, **self.terms}
        report_sig = self.sign_key.sign(bc.digest("report", bc.canonical_json(report))).hex()
        report_hash = bc.sha256(bc.canonical_json(report))
        settle_sig = self.sign_key.sign(chain.settle_message(self.job_id, report, report_hash))
        return report, report_sig, settle_sig, adapter_ct, private_ct


def party(chain: Chain, mint_authority: Keypair, usdc: int = 0) -> Keypair:
    kp = Keypair()
    chain.airdrop(kp.pubkey())
    chain.create_ata(kp, kp.pubkey())
    if usdc:
        chain.mint_test_usdc(mint_authority, kp.pubkey(), usdc)
    return kp


def post(chain: Chain, operator: Keypair, seller: Keypair, idx: int, measurement: bytes, **kw):
    terms = OfferTerms(KIND_TRAINING, TIERS["mock"], measurement, price_a=PRICE_PER_MILLION,
                       tokens_per_s=500, grace_s=RUN_RESERVE_S + 60, slots_total=2,
                       valid_until=chain.chain_time() + 86_400, **kw)
    chain.post_offer(operator, seller.pubkey(), idx, terms)
    return chain.get_offer(chain.offer_pda(seller.pubkey(), idx))


def prepare(chain: Chain, buyer: Keypair, work_dir: Path, offer) -> tuple[Requester, dict]:
    req = Requester(bc.new_key(), wallet_pk=bytes(buyer.pubkey()))
    memories = json.loads((ROOT / "demo/memories.json").read_text())
    eval_items = json.loads((ROOT / "demo/eval_hidden.json").read_text())
    model = {"id": "demo/tiny-softmax", "revision": "0", "manifest_sha256": "0" * 64}
    job = req.prepare_job(work_dir, memories, eval_items, model, {"method": "tiny"}, ACCEPTANCE,
                          fit_budget(UNITS, offer.quote(UNITS)[1]),
                          chain=chain.spec_chain(), open_slot=chain.current_slot(), n_canaries=0)
    return req, job


def inference_session(chain: Chain, seller: Keypair, operator: Keypair, buyer: Keypair,
                      measurement: bytes) -> bool:
    """Prepaid in small steps: the seller serves only while what's been used stays
    below the buyer's latest voucher, so neither side risks more than a step."""
    offer = OfferTerms(KIND_INFERENCE, TIERS["mock"], measurement, price_a=2 * USDC, price_b=6 * USDC,
                       valid_until=chain.chain_time() + 86_400)
    chain.post_offer(operator, seller.pubkey(), 1, offer)
    posted = next(o for o in chain.list_offers(kind=KIND_INFERENCE, tier=TIERS["mock"])
                  if o.seller == chain.seller_pda(seller.pubkey()))
    log("buyer", f"inference offer found: {posted.price_a / USDC:.0f} per million input tokens, "
        f"{posted.price_b / USDC:.0f} per million output tokens")

    session_key = Keypair()  # can sign vouchers for this channel and nothing else
    deposit, step = USDC, USDC // 100
    terms = {"offer": str(posted.address), "rev": posted.rev, "step": step}
    channel = chain.open_channel(buyer, seller.pubkey(), session_key.pubkey(), deposit,
                                 grace_period=60, salt=session_salt(terms))
    start_buyer, start_seller = chain.token_amount(buyer.pubkey()), chain.token_amount(seller.pubkey())
    log("buyer", f"channel opened, {deposit / USDC:.2f} deposited: the most this session can ever cost")

    used_in, used_out, used, covered = 0, 0, 0, 0
    voucher = None
    for _ in range(6):  # a few rounds of streamed tokens
        if used + step // 2 > covered:
            covered = used + step
            voucher = sign_voucher(session_key, channel, covered)
        used_in, used_out = used_in + 400, used_out + 150
        used = posted.usage_cost(used_in, used_out)
        if used > covered:
            log("seller", "stopped: usage passed the latest voucher")
            return False
    voucher = sign_voucher(session_key, channel, used)  # the closing voucher, for exactly what was used
    log("buyer", f"{used_in} tokens in, {used_out} out; cost {used / USDC:.4f} test-USDC; no chain "
        "transaction so far")

    chain.settle_and_seal(seller, chain.get_channel(channel), voucher)
    chain.distribute(seller, chain.get_channel(channel))
    earned = chain.token_amount(seller.pubkey()) - start_seller
    refunded = chain.token_amount(buyer.pubkey()) - start_buyer
    log("chain", f"settle_and_seal and distribute: seller +{earned / USDC:.4f}, "
        f"buyer got back {refunded / USDC:.4f} of the deposit")
    return earned == used and refunded == deposit - used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", default="chain.json")
    ap.add_argument("--mint-authority", default=".keys/deployer.json",
                    help="key allowed to mint the test-USDC")
    args = ap.parse_args()
    chain = Chain.from_file(ROOT / args.chain)
    assert chain.cfg["network"] == "local", "the demo waits out short windows: local chain only"
    bc.configure(bytes(chain.program_id), chain.cfg["cluster"].encode())
    mint_authority = load_keypair(ROOT / args.mint_authority)
    measurement = bc.code_measurement(sandbox_code_files())

    print("--- seller registers, stakes, posts an offer")
    seller, operator = party(chain, mint_authority, 100 * USDC), party(chain, mint_authority)
    chain.register_seller(seller, operator.pubkey())
    chain.stake(seller, STAKE)
    offer = post(chain, operator, seller, 0, bytes.fromhex(measurement))
    log("seller", f"staked {STAKE / USDC:.0f} test-USDC, offer {str(offer.address)[:12]}... "
        f"{offer.price_a / USDC:.0f} test-USDC per million tokens, tier MOCK")

    print("--- buyer finds the offer on chain and takes it")
    buyer = party(chain, mint_authority, 20 * USDC)
    found = chain.list_offers(kind=KIND_TRAINING, tier=TIERS["mock"])
    log("buyer", f"getProgramAccounts by kind and tier: {len(found)} live offers")
    picked = next(o for o in found if o.address == offer.address)
    price, run_secs = picked.quote(UNITS)
    balance0 = chain.token_amount(buyer.pubkey()), chain.token_amount(seller.pubkey())

    with tempfile.TemporaryDirectory() as tmp:
        job_dir = Path(tmp)
        req, job = prepare(chain, buyer, job_dir, picked)
        job_id = bytes.fromhex(job["job_id"])
        chain.open_job(buyer, buyer, picked, job_id, req.nonce, req.open_slot, req.spec_hash, UNITS)
        onchain = chain.get_job(job_id)
        log("buyer", f"open_job: {price / USDC:.2f} test-USDC escrowed, {onchain.bond / USDC:.2f} of "
            f"the seller's stake locked, run budget {run_secs} s")

        print("--- seller checks the spec against the order, then starts: sandbox keys and "
              "evidence hash go on chain")
        spec = json.loads((job_dir / "in" / "spec.json").read_text())
        problem = spec_problem(spec, onchain.terms)
        if problem:
            log("seller", f"would have to decline: {problem}")
            return 1
        sandbox = DemoSandbox(job_dir, {**job, **onchain.terms}, spec, measurement)
        evidence = sandbox.evidence()
        chain.start(operator, onchain, sandbox.sign_pk, sandbox.enc_pk, onchain.measurement,
                    evidence_hash(evidence))
        onchain = chain.get_job(job_id)
        log("seller", f"start: no attester in the loop, verifier = {onchain.verifier}")

        print("--- buyer verifies the evidence locally, then releases the key")
        with tempfile.TemporaryDirectory() as state:
            verifier = EvidenceVerifier(dev_policy(None), Path(state) / "devices.json")
            forged = json.loads(json.dumps(evidence))
            forged["attestation"]["measurement"] = "00" * 32
            try:
                req.check_evidence(verifier, forged, onchain, mainnet=False)
                log("buyer", "FORGED EVIDENCE ACCEPTED")
                return 1
            except AttestationRejected as e:
                log("buyer", f"forged evidence refused: {e}")
            verdict = req.check_evidence(verifier, evidence, onchain, chain.is_mainnet())
        log("buyer", f"{verdict.evidence_type} evidence verified, measurement on the allowlist")
        sealed = req.seal_dek()
        chain.release_key(buyer, job_id, sealed, onchain.evidence_hash)
        log("buyer", "release_key: data key sealed to the sandbox, on chain")

        print("--- the sandbox trains and signs a result")
        onchain = chain.get_job(job_id)
        report, report_sig, settle_sig, adapter_ct, private_ct = sandbox.train(chain, onchain.sealed_dek)
        chain.submit_result(operator, onchain, report, bc.sha256(bc.canonical_json(report)), settle_sig)
        onchain = chain.get_job(job_id)
        log("seller", f"submit_result: outcome {onchain.outcome}, work_bps {onchain.work_bps}")

        print("--- buyer checks the result and settles")
        checks = req.verify_report(report, report_sig, adapter_ct, private_ct, measurement)
        private = req.open_private_report(private_ct)
        qa = private["personal_qa"]
        log("buyer", f"report checks {'all OK' if all(checks.values()) else checks}; private report: "
            f"accuracy {qa['before']:.2f} -> {qa['after']:.2f}, {private['training']['train_tokens']} tokens")
        if not all(checks.values()) or report["outcome"] != 0:
            log("buyer", f"expected a passing job, got outcome {report['outcome']}")
            return 1
        names = req.open_adapter(report, adapter_ct, job_dir)
        log("buyer", f"adapter decrypted: {names}")
        chain.accept(buyer, onchain)
    paid = chain.token_amount(seller.pubkey()) - balance0[1]
    spent = balance0[0] - chain.token_amount(buyer.pubkey())
    closed = chain.get_job(job_id) is None
    log("chain", f"settled: seller +{paid / USDC:.2f}, buyer -{spent / USDC:.2f} test-USDC, "
        f"job account closed: {closed}")
    if not (closed and paid == price and spent == price):
        return 1

    print("--- a seller that never starts is slashed")
    no_show, op2 = party(chain, mint_authority, 100 * USDC), party(chain, mint_authority)
    chain.register_seller(no_show, op2.pubkey())
    chain.stake(no_show, STAKE)
    offer2 = post(chain, op2, no_show, 0, bytes.fromhex(measurement), start_window_s=3)
    with tempfile.TemporaryDirectory() as tmp:
        req2, job2 = prepare(chain, buyer, Path(tmp), offer2)
    job2_id = bytes.fromhex(job2["job_id"])
    before = chain.token_amount(buyer.pubkey())
    chain.open_job(buyer, buyer, offer2, job2_id, req2.nonce, req2.open_slot, req2.spec_hash, UNITS)
    stuck = chain.get_job(job2_id)
    log("buyer", f"took offer {str(offer2.address)[:12]}..., {stuck.price / USDC:.2f} escrowed; "
        "the seller stays silent")
    while chain.chain_time() <= stuck.start_by:
        time.sleep(0.5)
    stranger = party(chain, mint_authority)
    chain.timeout_start(stranger, stuck)
    chain.claim_due(stranger, job2_id)
    gained = chain.token_amount(buyer.pubkey()) - before
    seller2 = chain.get_seller(no_show.pubkey())
    paused = chain.get_offer(offer2.address).status == OFFER_PAUSED
    log("chain", f"anyone can end it: buyer net +{gained / USDC:.2f} test-USDC (refund plus slash), "
        f"seller stake {STAKE / USDC:.2f} -> {seller2.stake_total / USDC:.2f}, "
        f"n_nostart {seller2.n_nostart}, offer paused: {paused}")
    if not (gained > 0 and seller2.stake_total < STAKE and paused and seller2.n_nostart == 1):
        return 1

    if chain.channel_program is None:
        print("\nno channel program in chain.json: skipping the inference session")
    else:
        print("--- an inference session pays per token through a payment channel")
        if not inference_session(chain, seller, operator, buyer, bytes.fromhex(measurement)):
            return 1
    print("\nmarket demo finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
