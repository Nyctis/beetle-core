"""Adversarial tests for the beetle program on a local validator.

Every step sends a transaction that must either succeed or be rejected with a
specific error, or checks what the chain then holds. Sections named "attack N"
are the numbered attacks in section 4.8 of the chain design notes. Run after
scripts/setup_chain.sh local (which runs the --pre-init part itself), from the
beetle directory:

    uv run python scripts/chain_attack_test.py

Every party here is a throwaway key; the deployer in .keys/ is the mint and
freeze authority of the test-USDC.
"""

from __future__ import annotations

import os
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "attester")]

import beetle_crypto as bc  # noqa: E402
from attester import registration_message  # noqa: E402
from beetle_chain import (AWAY, INSTRUCTIONS_SYSVAR, KIND_INFERENCE, KIND_TRAINING,  # noqa: E402
                          NO_VERIFIER, OUTCOME_FAILED,
                          OUTCOME_INCONCLUSIVE, OUTCOME_PASSED, OUTCOME_REQUESTER_FAULT,
                          PAYEE_BUYER, PAYEE_SELLER, PAYEE_SLASH, PRICE_BY_TIME, REACH_ECHO,
                          REACH_PUBLIC_UDP, ROLE_BUYER, ROLE_SELLER, SYSTEM_PROGRAM,
                          TOKEN_PROGRAM, Chain, OfferTerms, ata, chain_ref_for, ed25519_ix,
                          evidence_hash, load_keypair, net_keypair, new_job_id, public_endpoint,
                          rdv_plaintext, session_salt, sign_voucher)
from job_defaults import (LOCAL_CHAIN_PARAMS, TIERS, fit_budget, outcome_for_error,  # noqa: E402
                          outcome_for_run, spec_problem)
from solders.instruction import AccountMeta, Instruction  # noqa: E402
from solders.keypair import Keypair  # noqa: E402
from solders.pubkey import Pubkey  # noqa: E402

P = LOCAL_CHAIN_PARAMS
USDC = 1_000_000
T2 = TIERS["t2"]
MOCK = TIERS["mock"]
UNPROVEN = [TIERS["mock"], TIERS["t1"], TIERS["t3b"], TIERS["t3c"]]
results: list[tuple[str, bool, str]] = []


def expect(name: str, fn, error: str | None = None):
    """Run fn; pass if it succeeds (error=None) or fails with `error` in the message."""
    try:
        out = fn()
        ok, detail = error is None, "succeeded" if error is None else "UNEXPECTEDLY SUCCEEDED"
    except Exception as e:
        msg = str(e)
        ok = error is not None and error in msg
        detail = f"rejected: {error}" if ok else f"WRONG ERROR: {msg[:300]}"
        out = None
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return out


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def summary() -> int:
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


def pre_init(chain: Chain) -> int:
    """Run on a fresh deployment, before setup_chain.sh initializes the config."""
    print("init_config")
    intruder = Keypair()
    chain.airdrop(intruder.pubkey())
    expect("someone other than the upgrade authority initializes", lambda: chain.init_config(
        intruder, P), "NotUpgradeAuthority")
    deployer = load_keypair(ROOT / ".keys/deployer.json")
    expect("a floor above 100%", lambda: chain.init_config(deployer, {**P, "floor_bps": 10_001}),
           "BadParams")
    expect("no tier allowed", lambda: chain.init_config(deployer, {**P, "allowed_tiers": 0}), "BadParams")
    expect("a release window longer than the sandbox waits for its key", lambda: chain.init_config(
        deployer, {**P, "release_window": 601}), "BadParams")
    expect("another network's chain id", lambda: Chain({**chain.cfg, "chain_ref": os.urandom(32).hex()})
           .init_config(deployer, P), "not this network's")
    return summary()


class Sandbox:
    """Stands in for the sealed sandbox: its one-time keys, nothing else."""

    def __init__(self):
        self.sk = bc.new_signing_key()
        self.sign_pk = bc.public_bytes(self.sk)
        self.enc_pk = os.urandom(32)


class Harness:
    def __init__(self, chain: Chain):
        self.chain = chain
        self.deployer = load_keypair(ROOT / ".keys/deployer.json")

    def party(self, usdc: int = 0) -> Keypair:
        kp = Keypair()
        self.chain.airdrop(kp.pubkey())
        self.chain.create_ata(kp, kp.pubkey())
        if usdc:
            self.chain.mint_test_usdc(self.deployer, kp.pubkey(), usdc * USDC)
        return kp

    def terms(self, tier: int = T2, measurement: bytes = bytes(32), **kw) -> OfferTerms:
        base = {"price_a": USDC, "start_window_s": 600, "grace_s": 600, "tokens_per_s": 1000,
                "slots_total": 16, "valid_until": self.chain.chain_time() + 86_400}
        return OfferTerms(KIND_TRAINING, tier, measurement, **{**base, **kw})

    def order(self, buyer: Keypair) -> tuple[bytes, bytes, int, bytes]:
        nonce, spec_hash = os.urandom(32), os.urandom(32)
        slot = self.chain.current_slot()
        return new_job_id(buyer.pubkey(), nonce, slot), nonce, slot, spec_hash

    def wait_past(self, t: int) -> None:
        while self.chain.chain_time() <= t:
            time.sleep(0.4)

    def open(self, buyer: Keypair, offer, units: int = 1_000_000, verifier=NO_VERIFIER,
             payer: Keypair | None = None) -> bytes:
        job_id, nonce, slot, spec_hash = self.order(buyer)
        self.chain.open_job(payer or buyer, buyer, offer, job_id, nonce, slot, spec_hash, units,
                            verifier=verifier)
        return job_id

    def started(self, operator: Keypair, buyer: Keypair, offer, sandbox: Sandbox, **kw) -> bytes:
        job_id = self.open(buyer, offer, **kw)
        self.chain.start(operator, self.chain.get_job(job_id), sandbox.sign_pk, sandbox.enc_pk,
                         offer.measurement, EVIDENCE)
        return job_id

    def running(self, operator: Keypair, buyer: Keypair, offer, sandbox: Sandbox, **kw) -> tuple[bytes, bytes]:
        job_id = self.started(operator, buyer, offer, sandbox, **kw)
        sealed = os.urandom(80)
        self.chain.release_key(buyer, job_id, sealed, EVIDENCE)
        return job_id, sealed

    def report(self, sandbox: Sandbox, job_id: bytes, sealed_dek: bytes, outcome: int,
               work_bps: int = 10_000) -> tuple[dict, bytes, bytes]:
        """A result signed by the sandbox: (report, report_hash, signature)."""
        passed = outcome == OUTCOME_PASSED
        report = {"outcome": outcome, "work_bps": work_bps, "sealed_dek_hash": bc.sha256(sealed_dek).hex(),
                  "adapter_ct_hash": os.urandom(32).hex(),
                  "sealed_adapter_key": os.urandom(80).hex() if passed else None,
                  **self.chain.get_job(job_id).terms}
        report_hash = os.urandom(32)
        return report, report_hash, sandbox.sk.sign(self.chain.settle_message(job_id, report, report_hash))

    def deliver(self, operator: Keypair, sandbox: Sandbox, job_id: bytes, sealed_dek: bytes,
                outcome: int, work_bps: int = 10_000):
        report, report_hash, sig = self.report(sandbox, job_id, sealed_dek, outcome, work_bps)
        self.chain.submit_result(operator, self.chain.get_job(job_id), report, report_hash, sig)
        return self.chain.get_job(job_id)

    def freeze(self, account, thaw: bool = False) -> None:
        self.chain.send([Instruction(TOKEN_PROGRAM, bytes([11 if thaw else 10]), [
            AccountMeta(account, False, True), AccountMeta(self.chain.mint, False, False),
            AccountMeta(self.deployer.pubkey(), True, False)])], [self.deployer])

    def token_account(self, owner: Pubkey, mint: Pubkey) -> Pubkey:
        kp = Keypair()
        rent = self.chain.rpc.call("getMinimumBalanceForRentExemption", 165)
        create = Instruction(SYSTEM_PROGRAM, struct.pack("<IQQ", 0, rent, 165) + bytes(TOKEN_PROGRAM), [
            AccountMeta(self.deployer.pubkey(), True, True), AccountMeta(kp.pubkey(), True, True)])
        init = Instruction(TOKEN_PROGRAM, bytes([18]) + bytes(owner), [
            AccountMeta(kp.pubkey(), False, True), AccountMeta(mint, False, False)])
        self.chain.send([create, init], [self.deployer, kp])
        return kp.pubkey()

    def create_mint(self) -> Keypair:
        """A second token, to prove that only the configured one is accepted."""
        kp = Keypair()
        space, rent = 82, self.chain.rpc.call("getMinimumBalanceForRentExemption", 82)
        create = Instruction(SYSTEM_PROGRAM, struct.pack("<IQQ", 0, rent, space) + bytes(TOKEN_PROGRAM), [
            AccountMeta(self.deployer.pubkey(), True, True), AccountMeta(kp.pubkey(), True, True)])
        init = Instruction(TOKEN_PROGRAM, bytes([20, 6]) + bytes(self.deployer.pubkey()) + bytes([0]),
                           [AccountMeta(kp.pubkey(), False, True)])
        self.chain.send([create, init], [self.deployer, kp])
        return kp


EVIDENCE = evidence_hash({"type": "MOCK", "test": True})


def main() -> int:
    chain = Chain.from_file(ROOT / "chain.json")
    assert chain.cfg["network"] == "local", "run against the local validator only"
    bc.configure(bytes(chain.program_id), chain.cfg["cluster"].encode())
    if "--pre-init" in sys.argv:
        return pre_init(chain)
    h = Harness(chain)
    requester = h.party(100)
    host = h.party(1000)        # the seller's cold authority key
    operator = h.party()        # its hot key on the machine
    intruder = h.party(100)
    measurement = os.urandom(32)
    host_pk = host.pubkey()

    print("init_config")
    expect("initialize a second time", lambda: chain.init_config(h.deployer, P), "already in use")
    config = chain.get_config()
    check("the config holds this network's chain id", config["chain_ref"] == chain.chain_ref
          == chain_ref_for(chain.rpc.call("getGenesisHash")))

    print("sellers and stake")
    expect("post an offer before registering", lambda: chain.post_offer(
        operator, host_pk, 0, h.terms(measurement=measurement)), "AccountNotInitialized")
    expect("register a seller", lambda: chain.register_seller(host, operator.pubkey()))
    expect("register the same seller twice", lambda: chain.register_seller(host), "already in use")
    expect("post with no stake", lambda: chain.post_offer(
        operator, host_pk, 0, h.terms(measurement=measurement)), "InsufficientStake")
    expect("stake from someone else's token account", lambda: chain.send(
        [chain._stake_ix("stake", struct.pack("<Q", USDC), host_pk, ata(intruder.pubkey(), chain.mint))],
        [host]), "ConstraintTokenOwner")
    other_mint = h.create_mint()
    expect("stake in another token", lambda: chain.send(
        [chain._stake_ix("stake", struct.pack("<Q", USDC), host_pk,
                         h.token_account(host_pk, other_mint.pubkey()))],
        [host]), "ConstraintTokenMint")
    expect("stake nothing", lambda: chain.stake(host, 0), "BadParams")
    expect("stake 500 test-USDC", lambda: chain.stake(host, 500 * USDC))
    expect("someone else requests unstake", lambda: chain.send([chain._ix(
        "request_unstake", struct.pack("<Q", USDC),
        [(intruder.pubkey(), True, False), (chain.config_pda, False, False),
         (chain.seller_pda(host_pk), False, True)])], [intruder]), "ConstraintSeeds")
    expect("unstake more than is free", lambda: chain.request_unstake(host, 501 * USDC),
           "InsufficientStake")
    expect("request unstake of 100", lambda: chain.request_unstake(host, 100 * USDC))
    expect("withdraw before the cooldown", lambda: chain.withdraw_stake(host), "TooEarly")
    seller = chain.get_seller(host_pk)
    check("pending stake counts as not free", seller.free_stake == 400 * USDC)
    h.wait_past(seller.unstake_ready_at)
    bal = chain.token_amount(host_pk)
    expect("withdraw after the cooldown", lambda: chain.withdraw_stake(host))
    check("stake returned to the authority", chain.token_amount(host_pk) - bal == 100 * USDC
          and chain.get_seller(host_pk).stake_total == 400 * USDC)
    expect("withdraw again", lambda: chain.withdraw_stake(host), "WrongState")
    expect("cancel with nothing pending", lambda: chain.cancel_unstake(host), "WrongState")
    expect("request unstake of 1", lambda: chain.request_unstake(host, USDC))
    expect("cancel it", lambda: chain.cancel_unstake(host))
    check("stake is free again", chain.get_seller(host_pk).free_stake == 400 * USDC)

    print("offers")
    expect("someone other than the operator posts", lambda: chain.post_offer(
        intruder, host_pk, 0, h.terms(measurement=measurement)), "ConstraintHasOne")
    expect("the cold key is not the operator", lambda: chain.post_offer(
        host, host_pk, 0, h.terms(measurement=measurement)), "ConstraintHasOne")
    expect("bond below the minimum", lambda: chain.post_offer(
        operator, host_pk, 0, h.terms(bond_bps=P["min_bond_bps"] - 1)), "BadParams")
    expect("valid for more than 7 days", lambda: chain.post_offer(
        operator, host_pk, 0, h.terms(valid_until=chain.chain_time() + 8 * 86_400)), "BadParams")
    expect("start window above the cap", lambda: chain.post_offer(
        operator, host_pk, 0, h.terms(start_window_s=P["max_start_window"] + 1)), "BadParams")
    expect("tier outside the allowed set", lambda: chain.post_offer(
        operator, host_pk, 0, h.terms(tier=5)), "TierNotAccepted")
    expect("a by-token offer with no throughput, so no run budget", lambda: chain.post_offer(
        operator, host_pk, 0, h.terms(tokens_per_s=0)), "BadParams")
    expect("an offer with no slots", lambda: chain.post_offer(
        operator, host_pk, 0, h.terms(slots_total=0)), "BadParams")
    expect("post a T2 training offer", lambda: chain.post_offer(
        operator, host_pk, 0, h.terms(measurement=measurement), payer=operator))
    offer = chain.get_offer(chain.offer_pda(host_pk, 0))
    seller = chain.get_seller(host_pk)
    check("the offer locks its stake", seller.stake_locked == P["offer_stake"] and seller.offers == 1)
    expect("post the same index again", lambda: chain.post_offer(
        operator, host_pk, 0, h.terms(measurement=measurement)), "already in use")
    expect("post a MOCK inference offer", lambda: chain.post_offer(operator, host_pk, 1, OfferTerms(
        KIND_INFERENCE, MOCK, measurement, price_a=1, price_b=2,
        valid_until=chain.chain_time() + 86_400)))
    expect("post a T2 by-time training offer", lambda: chain.post_offer(operator, host_pk, 2, h.terms(
        measurement=measurement, price_mode=PRICE_BY_TIME, price_b=3600 * USDC // 100)))
    by_kind = {str(o.address) for o in chain.list_offers(kind=KIND_INFERENCE)}
    by_tier = {str(o.address) for o in chain.list_offers(kind=KIND_TRAINING, tier=T2)}
    check("filter by kind finds the inference offer only",
          str(chain.offer_pda(host_pk, 1)) in by_kind and str(offer.address) not in by_kind)
    check("filter by tier finds the T2 training offers only", str(offer.address) in by_tier
          and str(chain.offer_pda(host_pk, 1)) not in by_tier)
    check("offer layout decodes through fixed offsets", offer.tier == T2 and offer.kind == KIND_TRAINING
          and offer.price_a == USDC and offer.measurement == measurement and offer.status == 1
          and offer.seller == seller.address and offer.offer_stake == P["offer_stake"])

    spammer = h.party(200)
    chain.register_seller(spammer)
    chain.stake(spammer, 100 * USDC)
    expect("post 16 offers", lambda: [chain.post_offer(spammer, spammer.pubkey(), i, h.terms(
        tier=MOCK)) for i in range(16)] and None)
    expect("a 17th offer", lambda: chain.post_offer(spammer, spammer.pubkey(), 16, h.terms(tier=MOCK)),
           "TooManyOffers")
    poor = h.party(30)
    chain.register_seller(poor)
    chain.stake(poor, 30 * USDC)
    expect("T2 needs more stake than MOCK", lambda: chain.post_offer(poor, poor.pubkey(), 0, h.terms()),
           "InsufficientStake")
    expect("someone else updates the offer", lambda: chain.update_offer(intruder, offer, h.terms(
        measurement=measurement)), "ConstraintHasOne")
    expect("update can't change the tier", lambda: chain.update_offer(operator, offer, h.terms(
        tier=MOCK, measurement=measurement)), "BadParams")
    stale = chain.get_offer(offer.address)
    expect("the seller raises the price", lambda: chain.update_offer(operator, offer, h.terms(
        measurement=measurement, price_a=2 * USDC)))
    offer = chain.get_offer(offer.address)
    check("the revision went up", offer.rev == stale.rev + 1 and offer.price_a == 2 * USDC)
    expect("heartbeat", lambda: chain.touch_offer(operator, offer, chain.chain_time() + 3600))
    check("a heartbeat doesn't change the revision", chain.get_offer(offer.address).rev == offer.rev)
    expect("heartbeat beyond 7 days", lambda: chain.touch_offer(
        operator, offer, chain.chain_time() + 8 * 86_400), "BadParams")

    print("open_job")
    buyer = requester
    job_id, nonce, slot, spec_hash = h.order(buyer)
    taken = lambda o, jid=job_id, n=nonce, sl=slot, units=1_000_000, **kw: chain.open_job(
        buyer, buyer, o, jid, n, sl, spec_hash, units, **kw)
    expect("job id not derived from requester+nonce+slot", lambda: taken(offer, os.urandom(32)), "BadJobId")
    expect("a price change between read and take", lambda: taken(stale), "OfferChanged")
    expect("price above the buyer's limit", lambda: taken(offer, max_price=2 * USDC - 1), "PriceTooHigh")
    expect("offer of another kind", lambda: taken(chain.get_offer(chain.offer_pda(host_pk, 1))), "BadParams")
    expect("pause the offer", lambda: chain.pause_offer(operator, offer))
    expect("take a paused offer", lambda: taken(offer), "OfferNotActive")
    expect("reactivate it at the old price", lambda: chain.update_offer(operator, offer, h.terms(
        measurement=measurement, min_order=USDC // 2, max_order=20 * USDC)))
    offer = chain.get_offer(offer.address)
    expect("order below the offer's minimum", lambda: taken(offer, units=1000), "OrderOutOfRange")
    expect("order above the offer's maximum", lambda: taken(offer, units=30_000_000), "OrderOutOfRange")
    expect("post a one-slot offer", lambda: chain.post_offer(operator, host_pk, 3, h.terms(
        measurement=measurement, slots_total=1)))
    single = chain.get_offer(chain.offer_pda(host_pk, 3))
    expect("post an offer that expires in 2 s", lambda: chain.post_offer(operator, host_pk, 4, h.terms(
        measurement=measurement, valid_until=chain.chain_time() + 2)))
    expiring = chain.get_offer(chain.offer_pda(host_pk, 4))
    h.wait_past(expiring.valid_until)
    expect("take an expired offer", lambda: h.open(buyer, expiring), "OfferNotActive")
    check("expired offers are not listed", str(expiring.address) not in
          {str(o.address) for o in chain.list_offers(kind=KIND_TRAINING, tier=T2)})
    expect("offer a seller's stake can't back", lambda: (
        chain.post_offer(poor, poor.pubkey(), 0, h.terms(tier=MOCK)),
        h.open(buyer, chain.get_offer(chain.offer_pda(poor.pubkey(), 0)), units=60_000_000)),
        "InsufficientStake")
    expect("a one-slot offer takes one job", lambda: h.open(buyer, single))
    expect("a second job on the full offer", lambda: h.open(buyer, single), "NoCapacity")
    timed = chain.get_offer(chain.offer_pda(host_pk, 2))
    jid = h.open(buyer, timed, units=600)
    tjob = chain.get_job(jid)
    check("a by-time offer prices seconds and sets the run budget from them",
          (tjob.price, tjob.run_secs) == timed.quote(600) and tjob.price == 6 * USDC
          and tjob.run_secs == 600 + timed.grace_s)
    thin = h.party(20)
    chain.register_seller(thin)
    chain.stake(thin, 15 * USDC)
    expect("post at exactly the MOCK minimum plus the offer stake", lambda: chain.post_offer(
        thin, thin.pubkey(), 0, h.terms(tier=MOCK)))
    expect("start leaving", lambda: chain.request_unstake(thin, 6 * USDC))
    expect("an offer whose stake is on its way out", lambda: h.open(
        buyer, chain.get_offer(chain.offer_pda(thin.pubkey(), 0))), "InsufficientStake")
    expect("open a valid job, no verifier", lambda: taken(offer))
    seller = chain.get_seller(host_pk)
    job = chain.get_job(job_id)
    check("the job locked the bond and took a slot", job.bond == job.price // 2 and
          seller.active_jobs == 3 and seller.stake_locked >= job.bond
          and chain.get_offer(offer.address).slots_used == 1)
    expect("close an offer with a job in progress", lambda: chain.close_offer(operator, offer), "OfferBusy")
    expect("unstake more than the free part", lambda: chain.request_unstake(host, seller.stake_total),
           "InsufficientStake")

    expect("the seller changes the offer mid-job", lambda: chain.update_offer(operator, offer, h.terms(
        measurement=os.urandom(32), price_a=9 * USDC)))
    held = chain.get_job(job_id)
    check("the open job keeps the terms it was taken at", held.price == job.price
          and held.measurement == measurement and held.bond == job.bond)
    expect("change it back", lambda: chain.update_offer(operator, offer, h.terms(
        measurement=measurement, min_order=USDC // 2, max_order=20 * USDC)))
    offer = chain.get_offer(offer.address)

    print("attack 15: old job ids")
    now = chain.current_slot()
    def stale_slot(s):
        n = os.urandom(32)
        return chain.open_job(buyer, buyer, offer, new_job_id(buyer.pubkey(), n, s), n, s, spec_hash, 1_000_000)
    expect("open_slot older than 1,500 slots", lambda: stale_slot(now - 1502), "StaleOpenSlot")
    expect("open_slot in the future", lambda: stale_slot(now + 1000), "StaleOpenSlot")

    print("start, no verifier: the buyer decides off chain")
    sandbox = Sandbox()
    enclave_pk, enc_pk = sandbox.sign_pk, sandbox.enc_pk
    expect("someone other than the operator starts it", lambda: chain.start(
        intruder, job, enclave_pk, enc_pk, measurement, EVIDENCE), "ConstraintHasOne")
    expect("a measurement other than the offer's", lambda: chain.start(
        operator, job, enclave_pk, enc_pk, os.urandom(32), EVIDENCE), "MeasurementMismatch")
    expect("start with no verifier and no signature", lambda: chain.start(
        operator, job, enclave_pk, enc_pk, measurement, EVIDENCE))
    expect("start twice", lambda: chain.start(
        operator, job, enclave_pk, enc_pk, measurement, EVIDENCE), "WrongState")
    job = chain.get_job(job_id)
    check("the job holds the keys and the evidence hash", job.sign_pk == enclave_pk
          and job.enc_pk == enc_pk and job.evidence_hash == EVIDENCE and job.release_by > 0)
    expect("the result before the key is released", lambda: h.deliver(
        operator, sandbox, job_id, bytes(80), OUTCOME_PASSED), "WrongState")
    expect("someone other than the requester rejects", lambda: chain.reject_evidence(intruder, job),
           "NotRequester")

    print("release_key")
    sealed_dek = os.urandom(80)
    expect("someone other than the requester releases the key", lambda: chain.release_key(
        intruder, job_id, sealed_dek, EVIDENCE), "ConstraintHasOne")
    expect("the buyer verified different evidence", lambda: chain.release_key(
        buyer, job_id, sealed_dek, os.urandom(32)), "EvidenceMismatch")
    expect("requester releases the key", lambda: chain.release_key(buyer, job_id, sealed_dek, EVIDENCE))
    job = chain.get_job(job_id)
    check("the run budget starts at release", job.deadline >= job.opened_at + job.run_secs
          and job.state == "Running")
    expect("release the key twice", lambda: chain.release_key(buyer, job_id, sealed_dek, EVIDENCE),
           "WrongState")
    expect("reject after releasing", lambda: chain.reject_evidence(buyer, job), "WrongState")

    print("submit_result signature checks")
    report, report_hash, sig = h.report(sandbox, job_id, sealed_dek, OUTCOME_PASSED)
    msg = chain.settle_message(job_id, report, report_hash)
    good_sig_ix = ed25519_ix(enclave_pk, sig, msg)
    def submit_raw(sig_ix, rep=report, who=operator):
        ix = chain.submit_result_ix(who.pubkey(), job, rep, report_hash)
        return chain.send(([sig_ix] if sig_ix else []) + [ix], [who])
    other_sk = bc.new_signing_key()
    expect("result signed by a key other than the sandbox's", lambda: submit_raw(
        ed25519_ix(bc.public_bytes(other_sk), other_sk.sign(msg), msg)), "WrongSigner")
    fail_rep = {**report, "outcome": OUTCOME_FAILED, "sealed_adapter_key": None}
    fail_msg = chain.settle_message(job_id, fail_rep, report_hash)
    expect("sandbox signed 'failed', host submits 'passed'", lambda: submit_raw(
        ed25519_ix(enclave_pk, sandbox.sk.sign(fail_msg), fail_msg)), "WrongMessage")
    elsewhere = Chain({**chain.cfg, "chain_ref": os.urandom(32).hex()})
    other_msg = elsewhere.settle_message(job_id, report, report_hash)
    expect("a report signed for another chain", lambda: submit_raw(
        ed25519_ix(enclave_pk, sandbox.sk.sign(other_msg), other_msg)), "WrongMessage")
    expect("no signature check instruction", lambda: submit_raw(None), "MissingSigIx")
    bad = bytearray(good_sig_ix.data)
    for off in (4, 8, 14):  # point every offset at instruction 0 instead of "this instruction"
        bad[off:off + 2] = struct.pack("<H", 0)
    expect("signature check pointing at other instruction data", lambda: submit_raw(
        Instruction(good_sig_ix.program_id, bytes(bad), [])), "BadSigIx")
    no_key = {**report, "sealed_adapter_key": None}
    no_key_msg = chain.settle_message(job_id, no_key, report_hash)
    expect("passed without an adapter key", lambda: submit_raw(
        ed25519_ix(enclave_pk, sandbox.sk.sign(no_key_msg), no_key_msg), no_key), "BadParams")
    expect("someone other than the operator submits", lambda: submit_raw(good_sig_ix, who=intruder),
           "ConstraintHasOne")
    expect("the seller rotates its operator key", lambda: chain.set_operator(host, intruder.pubkey()))
    expect("the old operator is out, mid-job", lambda: submit_raw(good_sig_ix), "ConstraintHasOne")
    expect("and rotates it back", lambda: chain.set_operator(host, operator.pubkey()))
    expect("valid sandbox-signed result", lambda: submit_raw(good_sig_ix))

    print("settlement")
    job = chain.get_job(job_id)
    expect("someone other than the requester accepts", lambda: chain.accept(intruder, job), "NotRequester")
    expect("finalize before the settle delay ends", lambda: chain.finalize(intruder, job), "TooEarly")
    expect("claim before settling", lambda: chain.claim(intruder, job, PAYEE_SELLER), "WrongState")
    before = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    rent0 = chain.sol_lamports(buyer.pubkey())
    expect("requester accepts", lambda: chain.accept(buyer, job))
    after = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    check("the host got the full price, no fee", after[1] - before[1] == job.price
          and after[0] == before[0], f"{job.price / USDC} test-USDC")
    seller = chain.get_seller(host_pk)
    check("the bond and the slot are free", seller.active_jobs == 2
          and chain.get_offer(offer.address).slots_used == 0 and seller.n_settled == 1
          and seller.volume == job.price)
    check("the job and vault closed, rent back to the payer", chain.get_job(job_id) is None
          and chain._account(chain.vault_pda(job_id)) is None
          and chain.sol_lamports(buyer.pubkey()) > rent0)
    expect("accept a settled job again", lambda: chain.accept(buyer, job), "AccountNotInitialized")

    print("a job that names a verifier: the old attester rules still hold")
    attester_sk = bc.new_signing_key()
    attester_pk = bc.public_bytes(attester_sk)
    jid = h.open(buyer, offer, verifier=Pubkey.from_bytes(attester_pk))
    vjob = chain.get_job(jid)

    def vstart(who=operator, signer=attester_sk, tier=T2, expiry=None, meas=measurement,
               signed_spec=None, signed_host=None):
        expiry = expiry or int(chain.chain_time()) + 600
        msg = registration_message(jid, signed_spec or vjob.spec_hash, enclave_pk, enc_pk, meas, tier,
                                   expiry, bytes(signed_host or host_pk))
        return chain.start(who, vjob, enclave_pk, enc_pk, measurement, EVIDENCE,
                           (bc.public_bytes(signer), signer.sign(msg), msg, expiry))

    expect("someone other than the operator starts it", lambda: vstart(who=intruder), "ConstraintHasOne")
    expect("no registration at all", lambda: chain.start(
        operator, vjob, enclave_pk, enc_pk, measurement, EVIDENCE), "MissingSigIx")
    expect("registration signed by someone other than the verifier",
           lambda: vstart(signer=bc.new_signing_key()), "WrongSigner")
    expect("verifier registered MOCK, the offer is T2", lambda: vstart(tier=MOCK), "WrongMessage")
    expect("verifier saw a different measurement", lambda: vstart(meas=os.urandom(32)), "WrongMessage")
    expect("registration made for another spec", lambda: vstart(signed_spec=os.urandom(32)), "WrongMessage")
    expect("registration made for another host", lambda: vstart(signed_host=intruder.pubkey()), "WrongMessage")
    expect("expired registration", lambda: vstart(expiry=chain.chain_time() - 5), "RegistrationExpired")
    expect("valid T2 registration", lambda: vstart())
    vjob = chain.get_job(jid)
    check("the sandbox keys are on chain", vjob.tier == T2 and vjob.measurement == measurement
          and vjob.sign_pk == enclave_pk)

    print("attack 5: the buyer starts jobs and never releases")
    h_job = vjob
    expect("claim before the release window ends", lambda: chain.claim_no_release(intruder, h_job), "TooEarly")
    h.wait_past(h_job.release_by + P["clock_margin"])
    expect("release after the window", lambda: chain.release_key(buyer, jid, sealed_dek, EVIDENCE), "TooLate")
    expect("reject after the window, to pay the lower fee", lambda: chain.reject_evidence(buyer, h_job), "TooLate")
    before = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    expect("anyone claims the no-release fee", lambda: chain.claim_no_release(intruder, h_job))
    expect("anyone pays both sides", lambda: chain.claim_due(intruder, jid))
    fee = min(h_job.price * P["no_release_fee_bps"] // 10_000, P["no_release_fee_cap"])
    after = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    check("the seller is paid for the held capacity", after[1] - before[1] == fee
          and after[0] - before[0] == h_job.price - fee, f"fee {fee / USDC}")
    seller = chain.get_seller(host_pk)
    check("bond and slot freed, and the buyer's no-show is on record", seller.n_norelease == 1
          and seller.active_jobs == 2 and chain.get_job(jid) is None)

    print("reject_evidence")
    jid = h.started(operator, buyer, offer, sandbox)
    rjob = chain.get_job(jid)
    before = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    expect("the buyer rejects the evidence", lambda: chain.reject_evidence(buyer, rjob))
    expect("both sides paid", lambda: chain.claim_due(intruder, jid))
    fee = min(rjob.price * P["reject_fee_bps"] // 10_000, P["reject_fee_cap"])
    after = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    check("the refund is less a small capped fee", after[1] - before[1] == fee
          and after[0] - before[0] == rjob.price - fee and fee <= P["reject_fee_cap"])
    check("counted as rejected", chain.get_seller(host_pk).n_rejected == 1)

    print("attack 1: throttle the sandbox to collect the floor")
    jid, dek = h.running(operator, buyer, offer, sandbox)
    rep, rh, _ = h.report(sandbox, jid, dek, OUTCOME_FAILED, work_bps=1000)
    msg = chain.settle_message(jid, rep, rh)
    forged = {**rep, "work_bps": 10_000}
    fjob = chain.get_job(jid)
    def submit(j, r, sig_ix):
        return chain.send([sig_ix, chain.submit_result_ix(operator.pubkey(), j, r, rh)], [operator])
    expect("host raises work_bps above what the sandbox signed", lambda: submit(
        fjob, forged, ed25519_ix(enclave_pk, sandbox.sk.sign(msg), msg)), "WrongMessage")
    over = {**rep, "work_bps": 10_001}
    over_msg = chain.settle_message(jid, over, rh)
    expect("work_bps above 100%", lambda: submit(
        fjob, over, ed25519_ix(enclave_pk, sandbox.sk.sign(over_msg), over_msg)), "BadParams")
    before = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    expect("failed after 10% of the planned tokens", lambda: submit(
        fjob, rep, ed25519_ix(enclave_pk, sandbox.sk.sign(msg), msg)))
    fjob = chain.get_job(jid)
    expect("accept", lambda: chain.accept(buyer, fjob))
    after = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    check("the floor scales with the work done: 2% of the price, not 20%",
          after[1] - before[1] == fjob.price * 200 // 10_000
          and after[0] - before[0] == fjob.price * 9800 // 10_000)
    check("counted as failed acceptance", chain.get_seller(host_pk).n_failed_accept == 1)
    outcome, done = outcome_for_run(False, 100_000, 1_000_000, 1900, 1800)
    check("hitting the wall clock is inconclusive, not a failure",
          outcome == OUTCOME_INCONCLUSIVE and done == 1000)
    jid, dek = h.running(operator, buyer, offer, sandbox)
    before = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    expect("an inconclusive report", lambda: h.deliver(operator, sandbox, jid, dek, outcome, done))
    expect("accept it", lambda: chain.accept(buyer, chain.get_job(jid)))
    after = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    check("inconclusive: full refund, nothing to the host", after[1] == before[1]
          and after[0] - before[0] == fjob.price)

    print("attack 2: frame the buyer for the floor")
    jid, dek = h.running(operator, buyer, offer, sandbox)
    swapped = outcome_for_error("train_ciphertext_hash_mismatch")
    check("a swapped input file is inconclusive, not the buyer's fault", swapped == OUTCOME_INCONCLUSIVE)
    check("dek_timeout is not reported at all", outcome_for_error("dek_timeout") is None)
    fake_dek = os.urandom(80)
    expect("host feeds the sandbox a fake sealed key and reports a bad key", lambda: h.deliver(
        operator, sandbox, jid, fake_dek, OUTCOME_REQUESTER_FAULT, 0), "SealedDekMismatch")
    before = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    expect("a real bad key: the sandbox opened what the chain holds", lambda: h.deliver(
        operator, sandbox, jid, dek, outcome_for_error("dek_open_failed"), 0))
    expect("accept it", lambda: chain.accept(buyer, chain.get_job(jid)))
    after = chain.token_amount(buyer.pubkey()), chain.token_amount(host_pk)
    check("the buyer's fault pays 5%, not the 20% floor", after[1] - before[1] == fjob.price * 500 // 10_000
          and after[0] - before[0] == fjob.price * 9500 // 10_000)

    print("attack 15: replay into a reopened job")
    nonce2 = os.urandom(32)
    slot2 = chain.current_slot()
    jid = new_job_id(buyer.pubkey(), nonce2, slot2)
    spec2 = os.urandom(32)
    chain.open_job(buyer, buyer, offer, jid, nonce2, slot2, spec2, 1_000_000)
    chain.start(operator, chain.get_job(jid), enclave_pk, enc_pk, measurement, EVIDENCE)
    dek = os.urandom(80)
    chain.release_key(buyer, jid, dek, EVIDENCE)
    rep, rh, sig = h.report(sandbox, jid, dek, OUTCOME_PASSED)
    old_msg = chain.settle_message(jid, rep, rh)
    chain.submit_result(operator, chain.get_job(jid), rep, rh, sig)
    chain.accept(buyer, chain.get_job(jid))
    check("the first job closed", chain.get_job(jid) is None)
    expect("the same id reopens inside the slot window", lambda: chain.open_job(
        buyer, buyer, offer, jid, nonce2, slot2, spec2, 1_000_000))
    chain.start(operator, chain.get_job(jid), enclave_pk, enc_pk, measurement, EVIDENCE)
    new_dek = os.urandom(80)
    chain.release_key(buyer, jid, new_dek, EVIDENCE)
    rjob = chain.get_job(jid)
    replay = lambda r, m: chain.send([ed25519_ix(enclave_pk, sig, m), chain.submit_result_ix(
        operator.pubkey(), rjob, r, rh)], [operator])
    expect("replay the old report", lambda: replay(rep, old_msg), "SealedDekMismatch")
    forged = {**rep, "sealed_dek_hash": bc.sha256(new_dek).hex()}
    expect("replay it with the new key's hash", lambda: replay(forged, old_msg), "WrongMessage")
    expect("the sandbox signs a fresh result", lambda: h.deliver(
        operator, sandbox, jid, new_dek, OUTCOME_PASSED))
    expect("accept it", lambda: chain.accept(buyer, chain.get_job(jid)))

    print("timeouts and slashing")
    expect("post a 2 s start window offer", lambda: chain.post_offer(operator, host_pk, 5, h.terms(
        measurement=measurement, start_window_s=2)))
    short = chain.get_offer(chain.offer_pda(host_pk, 5))
    jid = h.open(buyer, short)
    njob = chain.get_job(jid)
    expect("refund before the start window ends", lambda: chain.timeout_start(intruder, njob), "TooEarly")
    expect("timeout_run on a job that never started", lambda: chain.timeout_run(intruder, njob), "WrongState")
    h.wait_past(njob.start_by + P["clock_margin"])
    expect("start after the window", lambda: chain.start(
        operator, njob, enclave_pk, enc_pk, measurement, EVIDENCE), "TooLate")
    stake0 = chain.get_seller(host_pk)
    before = chain.token_amount(buyer.pubkey())
    expect("anyone ends it after the seller never started", lambda: chain.timeout_start(intruder, njob))
    expect("anyone pays the refund and the slash", lambda: chain.claim_due(intruder, jid))
    slash = max(njob.bond * P["nostart_slash_bps"] // 10_000, P["nostart_slash_min"])
    check("refund plus the no-start slash", chain.token_amount(buyer.pubkey()) - before == njob.price + slash,
          f"slash {slash / USDC}")
    seller = chain.get_seller(host_pk)
    check("the stake went down by the slash", stake0.stake_total - seller.stake_total == slash
          and seller.n_nostart == 1 and seller.stake_locked == stake0.stake_locked - njob.bond)
    check("the offer paused itself", chain.get_offer(short.address).status == 0)
    expect("it can't be taken again until the seller is back", lambda: h.open(buyer, short), "OfferNotActive")
    short = short_offer(chain, h, operator, host_pk, measurement)

    expect("post an offer with a 2 s run budget", lambda: chain.post_offer(operator, host_pk, 6, h.terms(
        measurement=measurement, grace_s=1, tokens_per_s=1_000_000)))
    quick = chain.get_offer(chain.offer_pda(host_pk, 6))
    jid, dek = h.running(operator, buyer, quick, sandbox)
    tjob = chain.get_job(jid)
    expect("timeout before the deadline", lambda: chain.timeout_run(intruder, tjob), "TooEarly")
    h.wait_past(tjob.deadline + P["clock_margin"])
    expect("deliver after the deadline", lambda: h.deliver(
        operator, sandbox, jid, dek, OUTCOME_PASSED), "TooLate")
    before = chain.token_amount(buyer.pubkey())
    expect("anyone ends it after the deadline", lambda: chain.timeout_run(intruder, tjob))
    expect("anyone pays the refund and the slash", lambda: chain.claim_due(intruder, jid))
    slash = tjob.bond * P["timeout_slash_bps"] // 10_000
    check("refund plus half the bond", chain.token_amount(buyer.pubkey()) - before == tjob.price + slash)
    check("counted as a timeout", chain.get_seller(host_pk).n_timeout == 1)

    print("settlement can't be blocked by one side")
    jid, dek = h.running(operator, buyer, offer, sandbox)
    h.deliver(operator, sandbox, jid, dek, OUTCOME_FAILED, 10_000)
    h.freeze(ata(host_pk, chain.mint))
    before = chain.token_amount(buyer.pubkey())
    fjob = chain.get_job(jid)
    expect("accept with the seller's account frozen", lambda: chain.accept(buyer, fjob))
    check("the buyer got paid anyway", chain.token_amount(buyer.pubkey()) - before == fjob.price * 8000 // 10_000)
    pending = chain.get_job(jid)
    expect("the seller's claim fails while frozen", lambda: chain.claim(intruder, pending, PAYEE_SELLER), "0x11")
    expect("the buyer can't claim twice", lambda: chain.claim(intruder, pending, PAYEE_BUYER), "NothingDue")
    h.freeze(ata(host_pk, chain.mint), thaw=True)
    expect("a claim can't be redirected to another wallet", lambda: chain.send(
        [chain.claim_ix(pending, PAYEE_SELLER, ata(intruder.pubkey(), chain.mint))], [intruder]),
        "WrongPayee")
    # Tokens sent to the vault must not keep it from closing.
    chain.send([Instruction(TOKEN_PROGRAM, bytes([3]) + struct.pack("<Q", 7), [
        AccountMeta(ata(intruder.pubkey(), chain.mint), False, True),
        AccountMeta(chain.vault_pda(jid), False, True), AccountMeta(intruder.pubkey(), True, False)])],
        [intruder])
    before = chain.token_amount(host_pk)
    expect("after the thaw the seller collects", lambda: chain.claim_due(intruder, jid))
    check("stray tokens went to the last payee, and the job closed",
          chain.token_amount(host_pk) - before == fjob.price * 2000 // 10_000 + 7
          and chain.get_job(jid) is None)

    jid, dek = h.running(operator, buyer, offer, sandbox)
    h.deliver(operator, sandbox, jid, dek, OUTCOME_FAILED, 10_000)
    h.freeze(ata(buyer.pubkey(), chain.mint))
    before = chain.token_amount(host_pk)
    fjob = chain.get_job(jid)
    expect("finalize with the buyer's account frozen", lambda: (
        h.wait_past(fjob.challenge_end), chain.finalize(intruder, fjob)))
    check("the seller got paid anyway", chain.token_amount(host_pk) - before == fjob.price * 2000 // 10_000)
    h.freeze(ata(buyer.pubkey(), chain.mint), thaw=True)
    expect("after the thaw the buyer collects", lambda: chain.claim_due(intruder, jid))

    jid = h.open(buyer, short_offer(chain, h, operator, host_pk, measurement))
    njob = chain.get_job(jid)
    h.wait_past(njob.start_by)
    vault = chain.seller_vault_pda(host_pk)
    h.freeze(vault)
    before = chain.token_amount(buyer.pubkey())
    expect("a no-start ends even with the seller vault frozen", lambda: chain.timeout_start(intruder, njob))
    pending = chain.get_job(jid)
    expect("the refund is paid", lambda: chain.claim(intruder, pending, PAYEE_BUYER))
    check("the refund arrived", chain.token_amount(buyer.pubkey()) - before == njob.price)
    expect("the slash waits for the thaw", lambda: chain.claim(intruder, chain.get_job(jid), PAYEE_SLASH), "0x11")
    h.freeze(vault, thaw=True)
    expect("the slash is paid after the thaw", lambda: chain.claim_due(intruder, jid))
    check("the last claim closed the job", chain.get_job(jid) is None)
    short_offer(chain, h, operator, host_pk, measurement)

    print("third-party payer")
    sponsor = h.party()
    jid, dek = h.running(operator, buyer, offer, sandbox, payer=sponsor)
    check("the buyer signs, the sponsor pays rent", chain.get_job(jid).payer == sponsor.pubkey())
    h.deliver(operator, sandbox, jid, dek, OUTCOME_PASSED)
    rent0 = chain.sol_lamports(sponsor.pubkey())
    chain.accept(buyer, chain.get_job(jid))
    check("the rent went back to the sponsor", chain.sol_lamports(sponsor.pubkey()) > rent0)

    griefing_section(chain, h, config, buyer, operator, intruder, offer, sandbox)
    abort_section(chain, h, config, buyer, operator, intruder, offer, sandbox)
    rendezvous_section(chain, h, buyer, intruder)
    order_section(chain, h, buyer, operator, intruder, offer, sandbox)
    unproven_section(chain, h, buyer, intruder)
    outage_section(chain, h, buyer, intruder)
    slot_section(chain, h, buyer, intruder)
    if chain.channel_program:
        channel_section(chain, h, buyer, host, intruder)

    print("closing an offer")
    seller = chain.get_seller(host_pk)
    expect("close the offer once it's idle", lambda: chain.close_offer(operator, offer))
    after = chain.get_seller(host_pk)
    check("its stake is unlocked and the account gone", after.stake_locked == seller.stake_locked - P["offer_stake"]
          and after.offers == seller.offers - 1 and chain.get_offer(offer.address) is None)
    return summary()


def griefing_section(chain: Chain, h: Harness, config: dict, buyer: Keypair, operator: Keypair,
                     intruder: Keypair, offer, sandbox: Sandbox) -> None:
    """A buyer who sends nothing must not be able to collect the no-start slash
    from a seller who is there."""
    print("griefing: a buyer takes the order and sends nothing")
    stake = lambda: chain.get_seller(offer_host)
    jid = h.open(buyer, offer)
    job = chain.get_job(jid)
    offer_host = job.host
    expect("someone else declines it", lambda: chain.decline_job(intruder, job), "NotOperator")
    expect("abort_unstarted with no rendezvous ever opened", lambda: chain.abort_unstarted(operator, job),
           "NoRendezvous")
    before, stake0 = chain.token_amount(buyer.pubkey()), stake()
    expect("the seller declines inside the window", lambda: chain.decline_job(operator, job))
    expect("anyone pays the refund", lambda: chain.claim_due(intruder, jid))
    s = stake()
    check("the buyer got its money back and nothing more",
          chain.token_amount(buyer.pubkey()) - before == job.price)
    check("no slash: stake unchanged, bond and slot free, on record as declined",
          s.stake_total == stake0.stake_total and s.stake_locked == stake0.stake_locked - job.bond
          and s.n_declined == stake0.n_declined + 1 and s.n_nostart == stake0.n_nostart)
    check("the job is gone, so no timeout can slash it later", chain.get_job(jid) is None)

    jid = h.open(buyer, offer)
    job = chain.get_job(jid)
    expect("a declined job can't then be started", lambda: (
        chain.decline_job(operator, job),
        chain.start(operator, job, sandbox.sign_pk, sandbox.enc_pk, offer.measurement, EVIDENCE)), "WrongState")
    expect("a declined job can't be timed out", lambda: chain.timeout_start(intruder, job), "WrongState")
    chain.claim_due(intruder, jid)

    jid = h.open(buyer, offer)
    job = chain.get_job(jid)
    chain.start(operator, job, sandbox.sign_pk, sandbox.enc_pk, offer.measurement, EVIDENCE)
    expect("decline after starting", lambda: chain.decline_job(operator, job), "WrongState")
    chain.reject_evidence(buyer, job)
    chain.claim_due(buyer, jid)

    print("griefing: the buyer opens a rendezvous so that decline runs out, then stalls")
    net = net_keypair(os.urandom(32))
    chain.rotate_net_pk(operator, offer_host, net.public_bytes)
    blob = lambda: os.urandom(224)
    jid = h.open(buyer, offer)
    job = chain.get_job(jid)
    chain.rdv_open(buyer, jid, blob())
    job = chain.get_job(jid)
    check("the job remembers a rendezvous was opened", job.rdv_opened)
    expect("abort_unstarted before the delay", lambda: chain.abort_unstarted(operator, job), "TooEarly")
    # The griefer tries to hold the seller here until the slash.
    rdv = chain.get_rendezvous(jid)
    expect("the griefer closes the rendezvous", lambda: chain.rdv_close(buyer, rdv))
    h.wait_past(job.opened_at + config["rdv_abort_after"])
    expect("a stranger aborts it", lambda: chain.abort_unstarted(intruder, job), "NotParticipant")
    before, stake0 = chain.token_amount(buyer.pubkey()), stake()
    expect("the seller still gets out, with no slash", lambda: chain.abort_unstarted(operator, job))
    expect("anyone pays the refund", lambda: chain.claim_due(intruder, jid))
    s = stake()
    check("no slash: stake unchanged, counted as a failed connection",
          s.stake_total == stake0.stake_total and s.n_connect_fail == stake0.n_connect_fail + 1
          and s.n_nostart == stake0.n_nostart)
    check("the griefer got its money back and nothing more",
          chain.token_amount(buyer.pubkey()) - before == job.price)

    print("griefing: the buyer shows up, then stalls past the old decline window")
    jid = h.open(buyer, offer)
    chain.rdv_open(buyer, jid, blob())
    job = chain.get_job(jid)
    h.wait_past(job.opened_at + config["decline_window"])
    before, stake0 = chain.token_amount(buyer.pubkey()), stake()
    expect("the seller can still decline", lambda: chain.decline_job(operator, job))
    chain.claim_due(intruder, jid)
    s = stake()
    check("the stalling buyer got its money back and nothing more",
          chain.token_amount(buyer.pubkey()) - before == job.price and s.stake_total == stake0.stake_total)
    check("the offer stays on sale", chain.get_offer(offer.address).status == 1)
    chain.rdv_close(buyer, chain.get_rendezvous(jid))

    print("griefing: a verifier that will never sign")
    jid = h.open(buyer, offer, verifier=Pubkey.from_bytes(os.urandom(32)))
    job = chain.get_job(jid)
    expect("start without the verifier's signature", lambda: chain.start(
        operator, job, sandbox.sign_pk, sandbox.enc_pk, offer.measurement, EVIDENCE), "MissingSigIx")
    h.wait_past(job.opened_at + config["decline_window"])
    expect("the seller declines it instead of waiting for the slash", lambda: chain.decline_job(operator, job))
    chain.claim_due(intruder, jid)

    jid = h.open(buyer, offer)
    chain.rdv_open(buyer, jid, blob())
    job = chain.get_job(jid)
    h.wait_past(job.opened_at + config["rdv_abort_after"])
    chain.rdv_open(buyer, jid, blob())  # reopening doesn't push the seller's exit back
    expect("reopening the rendezvous doesn't restart the clock", lambda: chain.abort_unstarted(operator, job))
    chain.claim_due(intruder, jid)
    chain.rdv_close(buyer, chain.get_rendezvous(jid))

    jid = h.open(buyer, offer)
    chain.rdv_open(buyer, jid, blob())
    job = chain.get_job(jid)
    h.wait_past(job.opened_at + config["rdv_abort_after"])
    expect("the buyer can leave too, when the seller never answers", lambda: chain.abort_unstarted(buyer, job))
    chain.claim_due(buyer, jid)
    chain.rdv_close(buyer, chain.get_rendezvous(jid))

    jid = h.started(operator, buyer, offer, sandbox)
    chain.rdv_open(buyer, jid, blob())
    job = chain.get_job(jid)
    h.wait_past(job.opened_at + config["rdv_abort_after"])
    expect("abort_unstarted after the job started", lambda: chain.abort_unstarted(operator, job), "WrongState")
    chain.reject_evidence(buyer, job)
    chain.claim_due(buyer, jid)
    chain.rdv_close(buyer, chain.get_rendezvous(jid))


def abort_section(chain: Chain, h: Harness, config: dict, buyer: Keypair, operator: Keypair,
                  intruder: Keypair, offer, sandbox: Sandbox) -> None:
    print("abort_job: the seller gives up on a running job")
    jid = h.started(operator, buyer, offer, sandbox)
    job = chain.get_job(jid)
    expect("abort before the key is released", lambda: chain.abort_job(operator, job), "WrongState")
    chain.release_key(buyer, jid, os.urandom(80), EVIDENCE)
    expect("someone else aborts it", lambda: chain.abort_job(intruder, job), "NotOperator")
    stake0 = chain.get_seller(job.host)
    before = chain.token_amount(buyer.pubkey())
    expect("the seller aborts", lambda: chain.abort_job(operator, job))
    expect("anyone pays the refund and the slash", lambda: chain.claim_due(intruder, jid))
    slash = job.bond * config["abort_slash_bps"] // 10_000
    s = chain.get_seller(job.host)
    check("refund plus 10% of the bond, less than a timeout costs",
          chain.token_amount(buyer.pubkey()) - before == job.price + slash
          and slash < job.bond * config["timeout_slash_bps"] // 10_000, f"slash {slash / USDC}")
    check("the stake went down by the slash, and it's on record",
          stake0.stake_total - s.stake_total == slash and s.n_abort == stake0.n_abort + 1)


def rendezvous_section(chain: Chain, h: Harness, buyer: Keypair, intruder: Keypair) -> None:
    print("rendezvous: network identity on chain")
    net = net_keypair(os.urandom(32))
    seller, op = h.party(200), h.party()
    chain.register_seller(seller, op.pubkey(), net.public_bytes)
    chain.stake(seller, 100 * USDC)
    pk = seller.pubkey()
    check("the seller's net key is on chain", chain.get_seller(pk).net_pk == net.public_bytes)
    expect("reach bits outside the defined three", lambda: chain.post_offer(
        op, pk, 0, h.terms(tier=MOCK, reach=8)), "BadParams")
    entry = os.urandom(32)
    expect("post an offer with reach and a public address", lambda: chain.post_offer(op, pk, 0, h.terms(
        tier=MOCK, reach=REACH_PUBLIC_UDP | REACH_ECHO, endpoint=public_endpoint(entry, "203.0.113.7:4433"))))
    offer = chain.get_offer(chain.offer_pda(pk, 0))
    check("reach and the public contact read back", offer.reach == REACH_PUBLIC_UDP | REACH_ECHO
          and offer.public_contact == {"id": entry.hex(), "addrs": ["203.0.113.7:4433"]})
    other = net_keypair(os.urandom(32))
    expect("someone else rotates the net key", lambda: chain.rotate_net_pk(intruder, pk, other.public_bytes),
           "ConstraintHasOne")
    expect("the operator rotates it", lambda: chain.rotate_net_pk(op, pk, other.public_bytes))
    s = chain.get_seller(pk)
    check("the rotation is counted", s.net_pk == other.public_bytes and s.net_pk_epoch == 1)
    net = other

    print("rendezvous: only the job's two sides can write")
    jid = h.open(buyer, offer)
    eid = os.urandom(32)
    addr = "203.0.113.9:40123"
    reply = bc.HpkeKeyPair()
    not_after = chain.chain_time() + 600
    def offer_blob(round_=1, j=jid):
        return chain.rdv_seal(net.public_bytes, j, ROLE_BUYER, round_, rdv_plaintext(
            ROLE_BUYER, round_, eid, not_after, [addr], reply.public_bytes))
    def answer_blob(round_=1):
        return chain.rdv_seal(reply.public_bytes, jid, ROLE_SELLER, round_, rdv_plaintext(
            ROLE_SELLER, round_, os.urandom(32), not_after, ["203.0.113.10:5000"]))
    blob = offer_blob()
    check("a blob is 224 bytes", len(blob) == 224)
    expect("someone else opens it", lambda: chain.send([chain._ix(
        "rdv_open", struct.pack("<Bq", 0, 600) + blob,
        [(intruder.pubkey(), True, True), (intruder.pubkey(), True, False), (chain.job_pda(jid), False, True),
         (chain.rdv_pda(jid), False, True), (SYSTEM_PROGRAM, False, False)])], [intruder]), "ConstraintHasOne")
    expect("an unknown policy", lambda: chain.rdv_open(buyer, jid, blob, policy=3), "BadParams")
    expect("no time to live", lambda: chain.rdv_open(buyer, jid, blob, ttl_secs=0), "BadParams")
    expect("more than a day to live", lambda: chain.rdv_open(buyer, jid, blob, ttl_secs=86_401), "BadParams")
    balance = chain.sol_lamports(buyer.pubkey())
    expect("the buyer opens it", lambda: chain.rdv_open(buyer, jid, blob))
    rdv = chain.get_rendezvous(jid)
    check("round 1 is offered, and the buyer paid the deposit", rdv.round == 1 and rdv.state == "Offered"
          and rdv.offer == blob and rdv.payer == buyer.pubkey()
          and chain.sol_lamports(buyer.pubkey()) < balance)
    raw = chain._account(rdv.address)
    check("no address or id in the clear on chain", eid not in raw
          and bytes([203, 0, 113, 9]) not in raw and struct.pack("<H", 40123) not in blob[32:])
    check("the seller can read it", chain.rdv_unseal(net, jid, ROLE_BUYER, 1, blob)["addrs"] == [addr])
    def refuses(name, fn):
        try:
            fn()
            check(name, False, "IT OPENED")
        except Exception:
            check(name, True)
    refuses("a stranger can't", lambda: chain.rdv_unseal(bc.HpkeKeyPair(), jid, ROLE_BUYER, 1, blob))
    refuses("it won't open as another job's", lambda: chain.rdv_unseal(net, os.urandom(32), ROLE_BUYER, 1, blob))
    refuses("it won't open as another round's", lambda: chain.rdv_unseal(net, jid, ROLE_BUYER, 2, blob))
    refuses("it won't open on another chain", lambda: Chain({
        **chain.cfg, "chain_ref": os.urandom(32).hex()}).rdv_unseal(net, jid, ROLE_BUYER, 1, blob))

    expect("someone else answers", lambda: chain.rdv_answer(intruder, rdv, 1, answer_blob()), "ConstraintHasOne")
    expect("the cold key is not the operator", lambda: chain.rdv_answer(seller, rdv, 1, answer_blob()),
           "ConstraintHasOne")
    expect("answer the wrong round", lambda: chain.rdv_answer(op, rdv, 2, answer_blob()), "BadRound")
    expect("someone else starts a new round", lambda: chain.rdv_reoffer(intruder, rdv, 2, offer_blob(2)),
           "ConstraintHasOne")
    expect("skip a round", lambda: chain.rdv_reoffer(buyer, rdv, 3, offer_blob(3)), "BadRound")
    expect("the seller answers", lambda: chain.rdv_answer(op, rdv, 1, answer_blob()))
    rdv = chain.get_rendezvous(jid)
    check("the buyer can read the answer", rdv.state == "Answered"
          and chain.rdv_unseal(reply, jid, ROLE_SELLER, 1, rdv.answer)["addrs"] == ["203.0.113.10:5000"])
    expect("answer twice", lambda: chain.rdv_answer(op, rdv, 1, answer_blob()), "WrongState")
    expect("close it early as a stranger", lambda: chain.rdv_close(intruder, rdv), "TooEarly")
    expect("a second round after a failed dial", lambda: chain.rdv_reoffer(buyer, rdv, 2, offer_blob(2)))
    rdv = chain.get_rendezvous(jid)
    check("round 2 is offered, the old answer wiped", rdv.round == 2 and rdv.state == "Offered"
          and rdv.answer == bytes(224))
    expect("a third round", lambda: chain.rdv_reoffer(buyer, rdv, 3, offer_blob(3)), "BadRound")
    expect("a new connection starts over at round 1", lambda: chain.rdv_open(buyer, jid, offer_blob()))
    check("reopening keeps the one account and its deposit", chain.get_rendezvous(jid).round == 1
          and chain.get_rendezvous(jid).address == rdv.address)

    print("rendezvous: failing and closing")
    expect("a stranger fails it", lambda: chain.rdv_fail(intruder, rdv), "NotParticipant")
    expect("the seller gives up", lambda: chain.rdv_fail(op, rdv, 1))
    rdv = chain.get_rendezvous(jid)
    expect("fail twice", lambda: chain.rdv_fail(buyer, rdv), "WrongState")
    expect("answer a failed rendezvous", lambda: chain.rdv_answer(op, rdv, 1, answer_blob()), "WrongState")
    expect("reoffer a failed rendezvous", lambda: chain.rdv_reoffer(buyer, rdv, 2, offer_blob(2)), "WrongState")
    expect("send the deposit to someone else", lambda: chain.send([chain._ix("rdv_close", b"", [
        (intruder.pubkey(), True, False), (rdv.address, False, True),
        (intruder.pubkey(), False, True)])], [intruder]), "ConstraintAddress")
    balance = chain.sol_lamports(buyer.pubkey())
    expect("anyone closes a failed one", lambda: chain.rdv_close(intruder, rdv))
    check("the account is gone and the deposit went back", chain.get_rendezvous(jid) is None
          and chain.sol_lamports(buyer.pubkey()) - balance >= 4_000_000)
    expect("a rendezvous can't outlive into a closed job", lambda: chain.rdv_open(
        buyer, os.urandom(32), offer_blob()), "AccountNotInitialized")

    sponsor = h.party()
    expect("someone else can pay the deposit", lambda: chain.rdv_open(
        buyer, jid, offer_blob(), ttl_secs=1, payer=sponsor))
    rdv = chain.get_rendezvous(jid)
    h.wait_past(rdv.expires_at)
    expect("answer after it expired", lambda: chain.rdv_answer(op, rdv, 1, answer_blob()), "TooLate")
    balance = chain.sol_lamports(sponsor.pubkey())
    expect("anyone closes an expired one", lambda: chain.rdv_close(intruder, rdv))
    check("the deposit went back to the one who paid", chain.sol_lamports(sponsor.pubkey()) > balance)

    print("rendezvous: reconnecting during a job")
    sandbox = Sandbox()
    chain.start(op, chain.get_job(jid), sandbox.sign_pk, sandbox.enc_pk, offer.measurement, EVIDENCE)
    expect("open it on a started job", lambda: chain.rdv_open(buyer, jid, offer_blob()))
    dek = os.urandom(80)
    chain.release_key(buyer, jid, dek, EVIDENCE)
    expect("open it on a running job", lambda: chain.rdv_open(buyer, jid, offer_blob()))
    h.deliver(op, sandbox, jid, dek, OUTCOME_PASSED)
    expect("open it on a delivered job", lambda: chain.rdv_open(buyer, jid, offer_blob()))
    chain.accept(buyer, chain.get_job(jid))
    expect("the buyer closes it when done, job gone", lambda: chain.rdv_close(buyer, chain.get_rendezvous(jid)))

    jid = h.open(buyer, offer)
    job = chain.get_job(jid)
    chain.decline_job(op, job)
    expect("open it on a finished job", lambda: chain.rdv_open(buyer, jid, offer_blob(j=jid)), "WrongState")
    chain.claim_due(buyer, jid)


def order_section(chain: Chain, h: Harness, buyer: Keypair, operator: Keypair, intruder: Keypair,
                  offer, sandbox: Sandbox) -> None:
    """What was paid for is what runs (review finding F1)."""
    print("an order for one token and a spec for two million")
    job = chain.get_job(h.open(buyer, offer))
    check("the job keeps what was paid for", job.units == 1_000_000 and job.price == offer.quote(1_000_000)[0]
          and job.grace_s == offer.grace_s and job.price_mode == offer.price_mode)
    fits = {"budget": fit_budget(job.units, job.run_secs), "inputs": dict.fromkeys(("body", "train", "eval", "program"))}
    check("a spec within the order is fine", spec_problem(fits, job.terms) is None)
    greedy = {**fits, "budget": {**fits["budget"], "max_train_tokens": job.units + 1}}
    check("a seller sees a spec that asks for more tokens", spec_problem(greedy, job.terms) == "budget_not_paid_for")
    late = {**fits, "budget": {**fits["budget"], "max_wall_seconds": fits["budget"]["max_wall_seconds"] + 1}}
    check("and one that couldn't report in time", spec_problem(late, job.terms) == "budget_not_paid_for")
    before, stake0 = chain.token_amount(buyer.pubkey()), chain.get_seller(job.host)
    expect("so it declines, at no cost", lambda: chain.decline_job(operator, job))
    chain.claim_due(intruder, job.job_id)
    check("no slash for refusing to work for free",
          chain.get_seller(job.host).stake_total == stake0.stake_total
          and chain.token_amount(buyer.pubkey()) - before == job.price)

    jid, dek = h.running(operator, buyer, offer, sandbox)
    job = chain.get_job(jid)
    for field, value in (("units", job.units * 2), ("price_mode", 1), ("run_secs", job.run_secs + 9),
                         ("grace_s", job.grace_s + 9)):
        rep, rh, _ = h.report(sandbox, jid, dek, OUTCOME_PASSED)
        told = {**rep, field: value}
        msg = chain.settle_message(jid, told, rh)
        expect(f"a sandbox that was told another {field} can't settle this job", lambda: chain.send(
            [ed25519_ix(sandbox.sign_pk, sandbox.sk.sign(msg), msg),
             chain.submit_result_ix(operator.pubkey(), job, told, rh)], [operator]), "WrongMessage")
    h.deliver(operator, sandbox, jid, dek, OUTCOME_PASSED)
    chain.accept(buyer, chain.get_job(jid))


def unproven_section(chain: Chain, h: Harness, buyer: Keypair, intruder: Keypair) -> None:
    """Off T2 the seller holds the signing key, so its Passed is a claim (F2)."""
    print("no hardware proof: the seller signs its own result")
    seller, op = h.party(500), h.party()
    chain.register_seller(seller, op.pubkey())
    chain.stake(seller, 300 * USDC)
    pk = seller.pubkey()
    sandbox = Sandbox()
    terms = lambda tier: h.terms(tier=tier, price_a=USDC // 10)
    for tier in UNPROVEN + [T2]:
        chain.post_offer(op, pk, tier, terms(tier))
    offers = {tier: chain.get_offer(chain.offer_pda(pk, tier)) for tier in UNPROVEN + [T2]}
    earned = lambda: chain.token_amount(pk)

    for tier in UNPROVEN:
        jid, dek = h.running(op, buyer, offers[tier], sandbox)
        job = h.deliver(op, sandbox, jid, dek, OUTCOME_PASSED)
        expect("finalize before the delay ends", lambda: chain.finalize(intruder, job), "TooEarly")
        h.wait_past(job.challenge_end)
        before, paid0 = earned(), chain.get_seller(pk).n_settled
        expect(f"tier {tier}: the buyer stays silent, anyone finalizes", lambda: chain.finalize(intruder, job))
        floor = job.price * P["floor_bps"] // 10_000
        check(f"tier {tier}: a self-signed Passed pays the floor, not the price",
              earned() - before == floor and floor < job.price and chain.get_job(jid) is None)
        check(f"tier {tier}: and doesn't count as a settled job", chain.get_seller(pk).n_settled == paid0)

    offer = offers[TIERS["t3c"]]
    jid, dek = h.running(op, buyer, offer, sandbox)
    job = h.deliver(op, sandbox, jid, dek, OUTCOME_PASSED)
    before = earned()
    expect("someone else can't accept for the buyer", lambda: chain.accept(intruder, job), "NotRequester")
    expect("the buyer tried it and accepts", lambda: chain.accept(buyer, job))
    check("accepting pays the full price at once", earned() - before == job.price)

    jid, dek = h.running(op, buyer, offer, sandbox)
    job = h.deliver(op, sandbox, jid, dek, OUTCOME_PASSED, 5_000)
    before, fails0 = (chain.token_amount(buyer.pubkey()), earned()), chain.get_seller(pk).n_failed_accept
    expect("someone else can't reject for the buyer", lambda: chain.reject(intruder, job), "NotRequester")
    expect("its hidden questions failed, so the buyer rejects", lambda: chain.reject(buyer, job))
    after = chain.token_amount(buyer.pubkey()), earned()
    pay = job.price * P["floor_bps"] * 5_000 // 10_000**2
    check("rejecting pays the floor scaled by the work, and refunds the rest",
          after[1] - before[1] == pay and after[0] - before[0] == job.price - pay, f"{pay / USDC}")
    check("it's on the seller's record", chain.get_seller(pk).n_failed_accept == fails0 + 1)
    expect("reject a settled job", lambda: chain.reject(buyer, job), "AccountNotInitialized")

    jid = h.started(op, buyer, offer, sandbox)
    job = chain.get_job(jid)
    expect("reject before there's a delivery", lambda: chain.reject(buyer, job), "WrongState")
    chain.reject_evidence(buyer, job)
    chain.claim_due(buyer, jid)

    jid, dek = h.running(op, buyer, offer, sandbox)
    job = h.deliver(op, sandbox, jid, dek, OUTCOME_REQUESTER_FAULT, 10_000)
    before = earned()
    h.wait_past(job.challenge_end)
    chain.finalize(intruder, job)
    check("a seller can't get there by blaming the buyer: 5%, whatever work it signs",
          earned() - before == job.price * P["fault_bps"] // 10_000)

    print("a proven sandbox: its signature is enough")
    jid, dek = h.running(op, buyer, offers[T2], sandbox)
    job = h.deliver(op, sandbox, jid, dek, OUTCOME_PASSED)
    expect("a buyer can't refuse a T2 result", lambda: chain.reject(buyer, job), "ProvenResult")
    before = earned()
    h.wait_past(job.challenge_end)
    expect("anyone finalizes it", lambda: chain.finalize(intruder, job))
    check("T2 pays the full price with no accept", earned() - before == job.price)

    jid, dek = h.running(op, buyer, offers[T2], sandbox)
    job = h.deliver(op, sandbox, jid, dek, OUTCOME_REQUESTER_FAULT, 3_000)
    before = earned()
    chain.accept(buyer, job)
    check("a hung program pays for the work it used up, not just the 5%",
          earned() - before == job.price * 3_000 // 10_000)


def outage_section(chain: Chain, h: Harness, buyer: Keypair, intruder: Keypair) -> None:
    """One outage costs one slash (M3), and a connection only counts against a
    seller that wasn't there (M2)."""
    print("one outage, many offers")
    seller, op = h.party(500), h.party()
    chain.register_seller(seller, op.pubkey())
    chain.stake(seller, 300 * USDC)
    pk = seller.pubkey()
    terms = lambda: h.terms(tier=MOCK, price_a=USDC, start_window_s=3, slots_total=8)
    for i in range(3):
        chain.post_offer(op, pk, i, terms())
    offers = [chain.get_offer(chain.offer_pda(pk, i)) for i in range(3)]
    jobs = [chain.get_job(h.open(buyer, offers[i % 3])) for i in range(6)]
    h.wait_past(max(j.start_by for j in jobs))
    stake0 = chain.get_seller(pk).stake_total
    expect("the first no-start", lambda: chain.timeout_start(buyer, jobs[0]))
    first = stake0 - chain.get_seller(pk).stake_total
    check("it slashes", first > 0)
    check("the seller is out until its operator returns", chain.get_seller(pk).away)
    expect("no offer of this seller takes orders", lambda: h.open(buyer, offers[1]), "OfferNotActive")
    for job in jobs[1:4]:
        expect("the same outage again, through other jobs", lambda: chain.timeout_start(buyer, job))
    check("it costs nothing more", stake0 - chain.get_seller(pk).stake_total == first)
    chain.update_offer(op, offers[0], terms())
    check("the operator is back", not chain.get_seller(pk).away and chain.get_seller(pk).outage_until < AWAY)
    expect("the buyer times out a job it held back from the outage", lambda: chain.timeout_start(buyer, jobs[4]))
    check("still the same outage: no slash", stake0 - chain.get_seller(pk).stake_total == first)
    for job in jobs:
        chain.claim_due(buyer, job.job_id) if chain.get_job(job.job_id) else None
    chain.update_offer(op, chain.get_offer(offers[0].address), terms())
    chain.timeout_start(buyer, jobs[5])
    chain.claim_due(buyer, jobs[5].job_id)
    chain.update_offer(op, chain.get_offer(offers[0].address), terms())
    fresh = h.open(buyer, chain.get_offer(offers[0].address))
    job = chain.get_job(fresh)
    h.wait_past(job.start_by)
    stake1 = chain.get_seller(pk).stake_total
    chain.timeout_start(buyer, job)
    check("a new outage is a new slash", chain.get_seller(pk).stake_total < stake1)
    chain.claim_due(buyer, fresh)
    chain.update_offer(op, chain.get_offer(offers[0].address), terms())

    print("abort_unstarted: who was there")
    net = net_keypair(os.urandom(32))
    chain.rotate_net_pk(op, pk, net.public_bytes)
    offer = chain.get_offer(chain.offer_pda(pk, 0))
    reply = bc.HpkeKeyPair()
    jid = h.open(buyer, offer)
    chain.rdv_open(buyer, jid, os.urandom(224))
    answer = chain.rdv_seal(reply.public_bytes, jid, ROLE_SELLER, 1, rdv_plaintext(
        ROLE_SELLER, 1, os.urandom(32), chain.chain_time() + 600, ["203.0.113.10:5000"]))
    chain.rdv_answer(op, chain.get_rendezvous(jid), 1, answer)
    job = chain.get_job(jid)
    check("the job remembers the seller answered", job.rdv_answered)
    h.wait_past(job.opened_at + P["rdv_abort_after"])
    before, fails = (chain.token_amount(buyer.pubkey()), chain.token_amount(pk)), chain.get_seller(pk).n_connect_fail
    expect("the buyer read the address and walks away", lambda: chain.abort_unstarted(buyer, job))
    chain.claim_due(intruder, jid)
    fee = P["reject_fee_cap"]
    check("that has a price now, and the seller's record stays clean",
          chain.token_amount(buyer.pubkey()) - before[0] == job.price - fee
          and chain.token_amount(pk) - before[1] == fee and chain.get_seller(pk).n_connect_fail == fails)
    chain.rdv_close(buyer, chain.get_rendezvous(jid))

    jid = h.open(buyer, offer)
    chain.rdv_open(buyer, jid, os.urandom(224))
    job = chain.get_job(jid)
    h.wait_past(job.opened_at + P["rdv_abort_after"])
    before = chain.token_amount(buyer.pubkey())
    expect("a seller that never answered", lambda: chain.abort_unstarted(buyer, job))
    chain.claim_due(intruder, jid)
    check("the buyer leaves for free and it's on the seller's record",
          chain.token_amount(buyer.pubkey()) - before == job.price
          and chain.get_seller(pk).n_connect_fail == fails + 1)
    chain.rdv_close(buyer, chain.get_rendezvous(jid))

    print("fees can't be an income")
    check("the reject and no-release fees only cover transaction costs",
          P["reject_fee_cap"] <= USDC // 100 and P["no_release_fee_cap"] <= USDC // 100)


def slot_section(chain: Chain, h: Harness, buyer: Keypair, intruder: Keypair) -> None:
    """A delivered job holds no machine, so it holds no slot. Its bond stays locked
    until it settles, and that, not the slot, backs the orders it makes room for."""
    print("a delivered job gives its slot back")
    seller, op = h.party(500), h.party()
    chain.register_seller(seller, op.pubkey())
    chain.stake(seller, 100 * USDC)
    pk, tier, sandbox = seller.pubkey(), TIERS["t3c"], Sandbox()
    terms = lambda **kw: h.terms(tier=tier, price_a=USDC // 10, slots_total=1, **kw)
    chain.post_offer(op, pk, 0, terms())
    offer = chain.get_offer(chain.offer_pda(pk, 0))
    slots = lambda: chain.get_offer(offer.address).slots_used
    held = lambda: chain.get_seller(pk).stake_locked

    first, dek = h.running(op, buyer, offer, sandbox)
    expect("one slot, one running job", lambda: h.open(buyer, offer), "NoCapacity")
    locked = held()
    job = h.deliver(op, sandbox, first, dek, OUTCOME_PASSED)
    check("delivered: the slot is free, with the sandbox gone", slots() == 0)
    check("the bond is not: it stays locked until the job settles",
          held() == locked and chain.get_seller(pk).active_jobs == 1)
    second = expect("a buyer who says nothing can't hold the seller up", lambda: h.running(
        op, buyer, offer, sandbox))
    if second is None:
        return
    check("the slot is on the new job, both bonds locked", slots() == 1 and held() == locked + job.bond)

    print("settling doesn't give the slot back again")
    expect("accept the first", lambda: chain.accept(buyer, job))
    check("accept: the second job keeps its slot, the first bond is free",
          slots() == 1 and held() == locked)
    job = h.deliver(op, sandbox, *second, OUTCOME_PASSED)
    third = h.running(op, buyer, offer, sandbox)
    expect("reject the second", lambda: chain.reject(buyer, job))
    check("reject: the third job keeps its slot", slots() == 1 and held() == locked)
    job = h.deliver(op, sandbox, *third, OUTCOME_PASSED)
    fourth = h.running(op, buyer, offer, sandbox)
    h.wait_past(job.challenge_end)
    expect("the third times out, anyone finalizes", lambda: chain.finalize(intruder, job))
    check("finalize: the fourth job keeps its slot", slots() == 1 and held() == locked)
    job = h.deliver(op, sandbox, *fourth, OUTCOME_FAILED)
    h.freeze(ata(pk, chain.mint))
    expect("settle a delivered job with the seller's account frozen", lambda: chain.accept(buyer, job))
    check("nothing is left to free", slots() == 0 and held() == locked - job.bond)
    h.freeze(ata(pk, chain.mint), thaw=True)
    expect("claim_due touches no slot", lambda: chain.claim_due(intruder, job.job_id))
    check("still free", slots() == 0 and chain.get_seller(pk).active_jobs == 0)

    print("a job that never delivered gives its slot back when it ends")
    def ends(name, jid, fn):
        expect(name, fn)
        check(f"{name}: the slot is free", slots() == 0)
        chain.claim_due(intruder, jid)

    jid = h.open(buyer, offer)
    ends("decline_job", jid, lambda: chain.decline_job(op, chain.get_job(jid)))
    jid = h.open(buyer, offer)
    chain.rdv_open(buyer, jid, os.urandom(224))
    job = chain.get_job(jid)
    h.wait_past(job.opened_at + P["rdv_abort_after"])
    ends("abort_unstarted", jid, lambda: chain.abort_unstarted(buyer, job))
    jid = h.started(op, buyer, offer, sandbox)
    ends("reject_evidence", jid, lambda: chain.reject_evidence(buyer, chain.get_job(jid)))
    jid = h.started(op, buyer, offer, sandbox)
    job = chain.get_job(jid)
    h.wait_past(job.release_by + P["clock_margin"])
    ends("claim_no_release", jid, lambda: chain.claim_no_release(intruder, job))
    jid, dek = h.running(op, buyer, offer, sandbox)
    ends("abort_job", jid, lambda: chain.abort_job(op, chain.get_job(jid)))
    chain.update_offer(op, offer, terms(grace_s=1, tokens_per_s=1_000_000))
    offer = chain.get_offer(offer.address)
    jid, dek = h.running(op, buyer, offer, sandbox)
    job = chain.get_job(jid)
    h.wait_past(job.deadline + P["clock_margin"])
    ends("timeout_run", jid, lambda: chain.timeout_run(intruder, job))
    chain.update_offer(op, offer, terms(start_window_s=2))
    offer = chain.get_offer(offer.address)
    jid = h.open(buyer, offer)
    job = chain.get_job(jid)
    h.wait_past(job.start_by + P["clock_margin"])
    ends("timeout_start", jid, lambda: chain.timeout_start(intruder, job))
    chain.update_offer(op, offer, terms())
    offer = chain.get_offer(offer.address)

    print("the offer account a delivered job still needs")
    chain.post_offer(op, pk, 1, terms())
    other = chain.get_offer(chain.offer_pda(pk, 1))
    jid, dek = h.running(op, buyer, offer, sandbox)
    job = chain.get_job(jid)
    rep, rh, sig = h.report(sandbox, jid, dek, OUTCOME_PASSED)
    msg = chain.settle_message(jid, rep, rh)
    sig_ix = ed25519_ix(sandbox.sign_pk, sig, msg)
    ix = chain.submit_result_ix(op.pubkey(), job, rep, rh)
    expect("a slot on another offer can't be freed instead", lambda: chain.send([sig_ix, Instruction(
        ix.program_id, ix.data, [*ix.accounts[:-1], AccountMeta(other.address, False, True)])], [op]),
        "BadParams")
    expect("deliver", lambda: chain.send([sig_ix, ix], [op]))
    job = chain.get_job(jid)
    check("no slot held, one job still to settle", slots() == 0 and job.state == "Delivered")
    expect("the offer can't close under it: the job needs it to settle", lambda: chain.close_offer(
        op, offer), "OfferBusy")
    expect("so it settles", lambda: chain.accept(buyer, job))

    print("a client that doesn't pass the offer")
    jid, dek = h.running(op, buyer, offer, sandbox)
    job = chain.get_job(jid)
    rep, rh, sig = h.report(sandbox, jid, dek, OUTCOME_PASSED)
    msg = chain.settle_message(jid, rep, rh)
    ix = chain.submit_result_ix(op.pubkey(), job, rep, rh)
    expect("still submits", lambda: chain.send([ed25519_ix(sandbox.sign_pk, sig, msg), Instruction(
        ix.program_id, ix.data, ix.accounts[:-1])], [op]))
    check("the slot then waits for settling, as it used to", slots() == 1)
    expect("settle it", lambda: chain.accept(buyer, chain.get_job(jid)))
    check("and goes back once", slots() == 0)
    expect("an idle offer closes", lambda: chain.close_offer(op, offer))

    print("a free slot is not free stake")
    rich = h.party(100)
    thin, thin_op = h.party(50), h.party()
    chain.register_seller(thin, thin_op.pubkey())
    bond = P["min_stake"][tier]  # all of the stake that isn't the offer's own
    chain.stake(thin, bond + P["offer_stake"])
    chain.post_offer(thin_op, thin.pubkey(), 0, h.terms(
        tier=tier, price_a=2 * bond, slots_total=4))
    single = chain.get_offer(chain.offer_pda(thin.pubkey(), 0))
    jid, dek = h.running(thin_op, rich, single, sandbox)
    job = h.deliver(thin_op, sandbox, jid, dek, OUTCOME_PASSED)
    check("its one bond takes all the free stake", job.bond == bond
          and chain.get_seller(thin.pubkey()).free_stake == 0)
    expect("the slot is free, but no new order fits without a bond", lambda: h.open(rich, single),
           "InsufficientStake")
    expect("nor can the bond be unstaked", lambda: chain.request_unstake(thin, 1), "InsufficientStake")
    chain.accept(rich, job)
    expect("once it settles the stake is free", lambda: h.open(rich, single))

    print("a slash on one job leaves the bond of a delivered one")
    jid, dek = h.running(op, buyer, chain.get_offer(chain.offer_pda(pk, 1)), sandbox)
    delivered = h.deliver(op, sandbox, jid, dek, OUTCOME_PASSED)
    jid, dek = h.running(op, buyer, chain.get_offer(chain.offer_pda(pk, 1)), sandbox)
    chain.abort_job(op, chain.get_job(jid))
    chain.claim_due(intruder, jid)
    s = chain.get_seller(pk)
    check("the stake still covers everything locked", s.stake_total >= s.stake_locked >= delivered.bond)
    chain.accept(buyer, delivered)


def channel_section(chain: Chain, h: Harness, buyer: Keypair, seller: Keypair, intruder: Keypair) -> None:
    """Inference money runs through the Foundation's program, not ours. These
    check that the client drives it correctly and that the rules we rely on hold.
    Errors are that program's codes (docs/003-program-instructions.md in its repo)."""
    print("inference: a session in a payment channel")
    session = Keypair()
    payee = seller.pubkey()
    salt = session_salt({"offer": "test", "nonce": os.urandom(8).hex()})
    deposit = USDC
    addr = expect("open a session with a 1.00 ceiling", lambda: chain.open_channel(
        buyer, payee, session.pubkey(), deposit, 2, salt))
    ch = chain.get_channel(addr)
    check("the channel reads back", ch.status == "Open" and ch.deposit == deposit and ch.settled == 0
          and ch.payee == payee and ch.authorized_signer == session.pubkey() and ch.salt == salt)
    check("the deposit left the buyer and sits in escrow",
          chain.rpc.call("getTokenAccountBalance", str(ata(addr, chain.mint)),
                         {"commitment": "confirmed"})["value"]["amount"] == str(deposit))
    expect("anyone settles a voucher", lambda: chain.settle_channel(
        intruder, ch, sign_voucher(session, addr, USDC // 4)))
    check("the watermark moved", chain.get_channel(addr).settled == USDC // 4)
    expect("an older voucher", lambda: chain.settle_channel(
        intruder, ch, sign_voucher(session, addr, USDC // 8)), "0xea")
    expect("a voucher above the deposit", lambda: chain.settle_channel(
        intruder, ch, sign_voucher(session, addr, deposit + 1)), "0xeb")
    stranger = Keypair()
    msg, sig = sign_voucher(stranger, addr, USDC // 2)
    expect("a voucher signed correctly by another key", lambda: chain.send(
        [ed25519_ix(bytes(stranger.pubkey()), sig, msg), chain._channel_ix(
            2, b"", [(addr, False, True), (INSTRUCTIONS_SYSVAR, False, False)])], [intruder]), "0xed")
    expect("a voucher with a bad signature", lambda: chain.settle_channel(
        intruder, ch, (msg, sig)), "custom program error: 0x2")
    expect("an expired voucher", lambda: chain.settle_channel(
        intruder, ch, sign_voucher(session, addr, USDC // 2, chain.chain_time() - 5)), "0xe9")
    other = Keypair().pubkey()
    expect("a voucher for another channel", lambda: chain.settle_channel(
        intruder, ch, sign_voucher(session, other, USDC // 2)), "0xe8")
    expect("top up the session", lambda: chain.top_up_channel(buyer, ch, USDC // 2))
    check("the ceiling went up", chain.get_channel(addr).deposit == deposit + USDC // 2)
    expect("someone other than the seller seals", lambda: chain.settle_and_seal(intruder, ch), "custom program error")
    before = chain.token_amount(buyer.pubkey()), chain.token_amount(payee)
    expect("the seller seals with its last voucher", lambda: chain.settle_and_seal(
        seller, ch, sign_voucher(session, addr, USDC // 2)))
    expect("no vouchers once sealed", lambda: chain.settle_channel(
        intruder, ch, sign_voucher(session, addr, USDC)), "custom program error")
    expect("anyone distributes", lambda: chain.distribute(intruder, chain.get_channel(addr)))
    after = chain.token_amount(buyer.pubkey()), chain.token_amount(payee)
    check("the seller got what was settled and the buyer the balance",
          after[1] - before[1] == USDC // 2 and after[0] - before[0] == deposit)
    ended = chain.get_channel(addr)
    check("the channel holds its address until the slot window passes", ended.status == "Distributed")
    expect("reclaim its rent too early", lambda: chain.reclaim_channel(intruder, ended), "custom program error")

    print("inference: the seller vanishes")
    session = Keypair()
    addr = chain.open_channel(buyer, payee, session.pubkey(), USDC, 2, session_salt({"n": os.urandom(8).hex()}))
    ch = chain.get_channel(addr)
    chain.settle_channel(intruder, ch, sign_voucher(session, addr, USDC // 10))
    expect("seal before asking to close", lambda: chain.seal_channel(intruder, ch), "custom program error")
    expect("the buyer asks to close", lambda: chain.request_close(buyer, ch))
    expect("seal inside the grace period", lambda: chain.seal_channel(intruder, ch), "custom program error")
    h.wait_past(chain.get_channel(addr).closure_started_at + 2)
    expect("anyone seals after the grace period", lambda: chain.seal_channel(intruder, ch))
    before = chain.token_amount(buyer.pubkey())
    expect("anyone distributes", lambda: chain.distribute(intruder, chain.get_channel(addr)))
    check("the buyer got back everything not settled",
          chain.token_amount(buyer.pubkey()) - before == USDC - USDC // 10)


def short_offer(chain: Chain, h: Harness, operator: Keypair, host_pk, measurement: bytes):
    """The 2 s start window offer again, reactivated after its slash paused it."""
    offer = chain.get_offer(chain.offer_pda(host_pk, 5))
    chain.update_offer(operator, offer, h.terms(measurement=measurement, start_window_s=2))
    return chain.get_offer(offer.address)


if __name__ == "__main__":
    sys.exit(main())
