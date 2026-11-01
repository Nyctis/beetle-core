"""Market rules that don't need a chain: the byte layouts every client depends
on, the error classification, and the buyer's local evidence check. What the
program itself enforces is in scripts/chain_attack_test.py.

Run from the beetle directory:  uv run --with pytest pytest -q tests/
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "requester"), str(ROOT / "attester")]

import beetle_crypto as bc  # noqa: E402
import beetle_chain as bch  # noqa: E402
import job_defaults as jd  # noqa: E402
from beetle_chain import (Chain, Channel, Job, Offer, OfferTerms, Seller, evidence_hash,  # noqa: E402
                          new_job_id, session_salt, sign_voucher, voucher_message)
from requester import AttestationRejected, Requester  # noqa: E402
from server import dev_policy  # noqa: E402
from solders.keypair import Keypair  # noqa: E402
from verifier import EvidenceVerifier, Rejected  # noqa: E402

FIX = ROOT / "tests" / "fixtures"


def chain() -> Chain:
    return Chain({"network": "local", "rpc_url": "http://127.0.0.1:1", "chain_ref": os.urandom(32).hex(),
                  "program_id": str(Keypair().pubkey()), "mint": str(Keypair().pubkey())})


def offer_bytes(**over) -> bytes:
    """An Offer account, field by field from the table in the design notes."""
    f = {"schema": 1, "status": 1, "kind": 0, "tier": 2, "reach": 0, "price_mode": 0, "hw_class": 7,
         "mem_gb": 80, "slots_total": 4, "slots_used": 1, "region": 0, "price_a": 11, "price_b": 22,
         "min_order": 33, "max_order": 44, "bond_bps": 5000, "start_window_s": 600, "grace_s": 30,
         "tokens_per_s": 1000, "valid_until": 1000, "updated_at": 900, "rev": 3, "offer_stake": 5,
         "idx": 9, "bump": 254, **over}
    return (bytes(8) + bytes([f["schema"], f["status"], f["kind"], f["tier"], f["reach"], f["price_mode"]])
            + struct.pack("<5H", f["hw_class"], f["mem_gb"], f["slots_total"], f["slots_used"], f["region"])
            + bytes(range(32)) + b"\xaa" * 32
            + struct.pack("<4Q", f["price_a"], f["price_b"], f["min_order"], f["max_order"])
            + struct.pack("<HIII", f["bond_bps"], f["start_window_s"], f["grace_s"], f["tokens_per_s"])
            + b"\xbb" * 32 + b"\xcc" * 64
            + struct.pack("<qqIQHB", f["valid_until"], f["updated_at"], f["rev"], f["offer_stake"],
                          f["idx"], f["bump"]) + bytes(11))


# --- Layouts -----------------------------------------------------------------------


def test_offer_filters_use_the_published_offsets():
    data = offer_bytes(status=1, kind=2, tier=3)
    assert len(data) == bch.OFFER_SIZE == bch.OFFER_LAYOUT.size == 272
    assert (data[bch.OFFER_STATUS_OFFSET], data[bch.OFFER_KIND_OFFSET], data[bch.OFFER_TIER_OFFSET]) == (1, 2, 3)
    assert data[24:56] == bytes(range(32))  # seller
    assert struct.unpack_from("<I", data, 246)[0] == 3  # rev


def test_offer_decodes_every_field():
    o = Offer.decode(Keypair().pubkey(), offer_bytes())
    assert (o.tier, o.hw_class, o.mem_gb, o.slots_total, o.slots_used) == (2, 7, 80, 4, 1)
    assert (o.price_a, o.price_b, o.min_order, o.max_order) == (11, 22, 33, 44)
    assert (o.bond_bps, o.start_window_s, o.grace_s, o.tokens_per_s) == (5000, 600, 30, 1000)
    assert (o.valid_until, o.updated_at, o.rev, o.offer_stake, o.idx, o.bump) == (1000, 900, 3, 5, 9, 254)
    assert o.measurement == b"\xaa" * 32 and o.model_root == b"\xbb" * 32 and o.endpoint == b"\xcc" * 64


def test_offer_terms_encode_to_the_program_struct():
    t = OfferTerms(0, 2, bytes(32), endpoint=b"node")
    # 4 u8 + 4 u16 + 32 + 4 u64 + u16 + 3 u32 + 32 + 64 + i64
    assert len(t.pack()) == 4 + 8 + 32 + 32 + 2 + 12 + 32 + 64 + 8
    assert t.pack()[-72:-8].startswith(b"node\0")


def test_seller_and_job_layouts():
    assert Seller.LAYOUT.size + 1 + 12 == 240  # + bump and padding
    job = Job.decode(Keypair().pubkey(), bytes(703))
    assert job.state == "Opened" and job.price == 0 and job.slash_due == 0
    tail = struct.pack("<?QIB?", True, 7, 9, 1, True) + bytes([254])
    job = Job.decode(Keypair().pubkey(), bytes(703 - len(tail)) + tail)
    assert job.terms == {"units": 7, "price_mode": 1, "run_secs": 0, "grace_s": 9}
    assert job.rdv_opened and job.rdv_answered


def test_quote_rounds_up_for_the_seller():
    o = Offer.decode(Keypair().pubkey(), offer_bytes(price_a=1_000_000, tokens_per_s=1000, grace_s=30))
    assert o.quote(2_000_000) == (2_000_000, 2030)
    assert o.quote(1) == (1, 31)
    by_time = Offer.decode(Keypair().pubkey(), offer_bytes(price_mode=1, price_b=3_600_000, grace_s=30))
    assert by_time.quote(1800) == (1_800_000, 1830)


@pytest.mark.parametrize("outcome,work_bps,seller", [
    (bch.OUTCOME_PASSED, 10_000, 1_000_000),
    (bch.OUTCOME_FAILED, 10_000, 200_000),
    (bch.OUTCOME_FAILED, 1_000, 20_000),     # attack 1: floor scales with the work
    (bch.OUTCOME_FAILED, 0, 0),
    (bch.OUTCOME_REQUESTER_FAULT, 0, 50_000),
    (bch.OUTCOME_INCONCLUSIVE, 10_000, 0),
])
def test_expected_dues_match_the_program(outcome, work_bps, seller):
    job = delivered(jd.TIERS["t2"], outcome, work_bps)
    assert job.expected_dues() == (seller, 1_000_000 - seller)


def delivered(tier: int, outcome: int, work_bps: int) -> Job:
    job = Job.decode(Keypair().pubkey(), bytes(703))
    job.state, job.price, job.floor_bps, job.fault_bps = "Delivered", 1_000_000, 2000, 500
    job.tier, job.outcome, job.work_bps = tier, outcome, work_bps
    return job


@pytest.mark.parametrize("tier", [jd.TIERS["mock"], jd.TIERS["t1"], jd.TIERS["t3b"], jd.TIERS["t3c"]])
def test_only_a_proven_sandbox_or_the_buyer_pays_the_full_price(tier):
    assert delivered(tier, bch.OUTCOME_PASSED, 10_000).expected_dues() == (200_000, 800_000)
    assert delivered(tier, bch.OUTCOME_PASSED, 10_000).expected_dues(accepted=True) == (1_000_000, 0)
    assert delivered(tier, bch.OUTCOME_FAILED, 10_000).expected_dues(accepted=True) == (200_000, 800_000)
    assert delivered(tier, bch.OUTCOME_REQUESTER_FAULT, 10_000).expected_dues() == (50_000, 950_000)


def test_a_buyers_fault_on_t2_pays_for_the_work_it_used_up():
    t2 = jd.TIERS["t2"]
    assert delivered(t2, bch.OUTCOME_REQUESTER_FAULT, 3_000).expected_dues() == (300_000, 700_000)
    assert delivered(t2, bch.OUTCOME_REQUESTER_FAULT, 100).expected_dues() == (50_000, 950_000)


def test_job_id_binds_requester_nonce_and_slot():
    a, b = Keypair().pubkey(), Keypair().pubkey()
    nonce = os.urandom(32)
    assert new_job_id(a, nonce, 5) == hashlib.sha256(bytes(a) + nonce + struct.pack("<Q", 5)).digest()
    assert new_job_id(a, nonce, 5) != new_job_id(a, nonce, 6)
    assert new_job_id(a, nonce, 5) != new_job_id(b, nonce, 5)


# --- Inference sessions ----------------------------------------------------------------


def test_channel_layout_is_the_foundation_programs():
    assert Channel.LAYOUT.size == 256
    keys = [Keypair().pubkey() for _ in range(5)]
    data = Channel.LAYOUT.pack(1, 1, 255, 1, 7, 1000, 400, 300, 5, 6, 60, bytes(32),
                               *[bytes(k) for k in keys], 99)
    ch = Channel.decode(Keypair().pubkey(), data)
    assert (ch.status, ch.salt, ch.deposit, ch.settled, ch.grace_period) == ("Sealed", 7, 1000, 400, 60)
    assert (ch.payer, ch.payee, ch.authorized_signer, ch.mint, ch.rent_payer) == tuple(keys)
    assert ch.open_slot == 99


def test_voucher_format():
    channel = Keypair().pubkey()
    msg = voucher_message(channel, 5, 7)
    assert len(msg) == 50 and msg[:2] == b"\x56\x01" and msg[2:34] == bytes(channel)
    assert struct.unpack("<Qq", msg[34:]) == (5, 7)
    key = Keypair()
    got, sig = sign_voucher(key, channel, 5, 7)
    assert got == msg and bc.verify(bytes(key.pubkey()), sig, msg)


def test_session_salt_commits_to_the_terms():
    assert session_salt({"p": 1}) == session_salt({"p": 1})
    assert session_salt({"p": 1}) != session_salt({"p": 2})
    assert 0 <= session_salt({"p": 1}) < 2**64


def test_usage_cost_rounds_up():
    o = Offer.decode(Keypair().pubkey(), offer_bytes(price_a=2_000_000, price_b=6_000_000))
    assert o.usage_cost(1_000_000, 1_000_000) == 8_000_000
    assert o.usage_cost(1, 0) == 2


# --- What the sandbox signs -----------------------------------------------------------


def report(**over) -> dict:
    return {"outcome": 0, "work_bps": 10_000, "sealed_dek_hash": "11" * 32, "adapter_ct_hash": "22" * 32,
            "sealed_adapter_key": "33" * 80, "units": 500, "price_mode": 1, "run_secs": 700, "grace_s": 30,
            **over}


def test_settle_message_layout():
    c = chain()
    job_id, rh = b"\x01" * 32, b"\x02" * 32
    want = hashlib.sha256(b"beetle/v0/settle" + bytes(c.program_id) + c.chain_ref.hex().encode() + job_id
                          + b"\x00" + struct.pack("<H", 10_000) + b"\x11" * 32 + b"\x22" * 32
                          + b"\x01" + b"\x33" * 80 + rh + struct.pack("<QBqI", 500, 1, 700, 30)).digest()
    assert c.settle_message(job_id, report(), rh) == want


@pytest.mark.parametrize("change", [
    {"outcome": 1}, {"work_bps": 9999}, {"sealed_dek_hash": "44" * 32}, {"adapter_ct_hash": "44" * 32},
    {"sealed_adapter_key": None}, {"sealed_adapter_key": "44" * 80}, {"units": 501},
    {"price_mode": 0}, {"run_secs": 701}, {"grace_s": 31}])
def test_settle_message_binds_every_field(change):
    c = chain()
    assert c.settle_message(b"\0" * 32, report(), bytes(32)) != c.settle_message(b"\0" * 32, report(**change), bytes(32))


def test_settle_message_binds_the_deployment():
    a, b = chain(), chain()
    assert a.settle_message(b"\0" * 32, report(), bytes(32)) != b.settle_message(b"\0" * 32, report(), bytes(32))


def test_the_sandbox_and_the_client_digest_the_same_chain_tag():
    c = chain()
    bc.configure_from_spec({"chain": c.spec_chain()})
    r, rh = report(), b"\x07" * 32
    assert bc.settle_message(b"\0" * 32, r, rh) == c.settle_message(b"\0" * 32, r, rh)


def test_chain_ref_is_the_caip2_hash():
    genesis = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d"
    assert bch.chain_ref_for(genesis) == hashlib.sha256(b"solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp").digest()


# --- Who is at fault (design notes, 4.7) ------------------------------------------------


@pytest.mark.parametrize("reason", ["dek_open_failed", "data_decrypt_failed", "malformed_inputs",
                                    "program_hash_mismatch", "unknown_mode", "threshold_above_cap",
                                    "baseline_already_passes"])
def test_faults_the_host_cannot_fake_are_the_requesters(reason):
    assert jd.outcome_for_error(reason) == jd.OUTCOME_REQUESTER_FAULT


@pytest.mark.parametrize("reason", ["body_ciphertext_hash_mismatch", "train_ciphertext_hash_mismatch",
                                    "eval_ciphertext_hash_mismatch", "program_ciphertext_hash_mismatch",
                                    "base_model_hash_mismatch", "internal_error", "run_budget_exhausted",
                                    "anything_new"])
def test_anything_else_is_inconclusive_and_pays_the_host_nothing(reason):
    assert jd.outcome_for_error(reason) == jd.OUTCOME_INCONCLUSIVE


@pytest.mark.parametrize("reason", ["dek_timeout", "job_terms_missing", "spec_malformed",
                                    "budget_not_paid_for"])
def test_what_stops_a_sandbox_before_the_key_is_not_reported(reason):
    assert jd.outcome_for_error(reason) is None


def test_work_bps():
    assert jd.work_bps(0, 100) == 0
    assert jd.work_bps(50, 100) == 5000
    assert jd.work_bps(150, 100) == 10_000
    assert jd.work_bps(5, 0) == 10_000


def test_wall_clock_overrun_is_inconclusive():
    assert jd.outcome_for_run(False, 10, 100, 99, 100) == (jd.OUTCOME_FAILED, 1000)
    assert jd.outcome_for_run(False, 10, 100, 101, 100) == (jd.OUTCOME_INCONCLUSIVE, 1000)
    assert jd.outcome_for_run(True, 100, 100, 101, 100) == (jd.OUTCOME_PASSED, 10_000)


def test_a_program_that_stopped_itself_pays_for_what_it_used():
    assert jd.outcome_for_run(False, 30, 100, 101, 100, True) == (jd.OUTCOME_REQUESTER_FAULT, 3000)
    assert jd.outcome_for_run(True, 30, 100, 101, 100, True) == (jd.OUTCOME_PASSED, 3000)


def test_network_params():
    assert jd.NETWORK_PARAMS["mainnet"]["allowed_tiers"] & 1 == 0, "MOCK must be off on mainnet"
    assert jd.NETWORK_PARAMS["devnet"]["clock_margin"] >= 60
    assert jd.NETWORK_PARAMS["local"]["release_window"] < jd.NETWORK_PARAMS["devnet"]["release_window"]
    assert jd.NETWORK_PARAMS["mainnet"]["allowed_tiers"] >> jd.TIERS["t3c"] & 1, "plain GPUs have a tier"


def test_the_sandbox_outlasts_any_release_window():
    """A buyer that releases on the window's last second must find a sandbox."""
    program = (ROOT / "programs/beetle/src/lib.rs").read_text()
    assert f"MAX_RELEASE_WINDOW: i64 = {jd.MAX_RELEASE_WINDOW};" in program
    assert jd.DEK_WAIT_SECONDS >= jd.MAX_RELEASE_WINDOW + jd.START_LATENCY_S > jd.MAX_RELEASE_WINDOW
    assert all(p["release_window"] <= jd.MAX_RELEASE_WINDOW for p in jd.NETWORK_PARAMS.values())


def test_a_seller_keeps_its_fees_too_small_to_farm():
    for p in jd.NETWORK_PARAMS.values():
        assert p["reject_fee_cap"] <= 10_000 and p["no_release_fee_cap"] <= 10_000


def test_a_budget_is_checked_against_the_order():
    terms = {"units": 1000, "price_mode": jd.PRICE_BY_TOKEN, "run_secs": 900, "grace_s": 600}
    inputs = dict.fromkeys(("body", "train", "eval", "program"), "00")
    spec = lambda **b: {"inputs": inputs, "budget": {**jd.fit_budget(1000, 900), **b}}
    assert jd.spec_problem(spec(), terms) is None
    assert jd.spec_problem(spec(max_train_tokens=1001), terms) == "budget_not_paid_for"
    assert jd.spec_problem(spec(max_wall_seconds=900), terms) == "budget_not_paid_for"
    assert jd.spec_problem(spec(max_train_tokens=2 * 10**12), {**terms, "price_mode": jd.PRICE_BY_TIME}) is None
    for bad in (0, -1, True, "9", 1.5, None):
        assert jd.spec_problem(spec(max_train_tokens=bad), terms) == "spec_malformed"
        assert jd.spec_problem(spec(max_wall_seconds=bad), terms) == "spec_malformed"
    assert jd.spec_problem({"inputs": {}, "budget": {}}, terms) == "spec_malformed"


# --- The buyer's local check --------------------------------------------------------------


MEASUREMENT = dev_policy(None)["mock"]["allowed_measurements"][0]


def setup_job(measurement=MEASUREMENT, tier=0):
    """A buyer with a prepared job, and what a sandbox and the chain would show it."""
    req = Requester(os.urandom(32), os.urandom(32))
    req.job_id, req.spec_hash = os.urandom(32), os.urandom(32)
    sk = bc.new_signing_key()
    sign_pk, enc_pk = bc.public_bytes(sk), os.urandom(32)
    att = {"tier": "MOCK", "mock": True, "measurement": measurement, "job_id": req.job_id.hex(),
           "spec_hash": req.spec_hash.hex(), "sign_pk": sign_pk.hex(), "enc_pk": enc_pk.hex(),
           "report_data": bc.digest("start", req.job_id + req.spec_hash + sign_pk + enc_pk).hex(),
           "network": {"isolated": True}}
    evidence = {"type": "MOCK", "attestation": att,
                "signature": sk.sign(bc.digest("attest", bc.canonical_json(att))).hex()}
    job = SimpleNamespace(sign_pk=sign_pk, enc_pk=enc_pk, measurement=bytes.fromhex(measurement),
                          tier=tier, evidence_hash=evidence_hash(evidence))
    return req, evidence, job


@pytest.fixture
def dev(tmp_path):
    return EvidenceVerifier(dev_policy(None), tmp_path / "state.json")


def test_good_mock_evidence_lets_the_key_go(dev):
    req, evidence, job = setup_job()
    verdict = req.check_evidence(dev, evidence, job, mainnet=False)
    assert verdict.evidence_type == "MOCK" and req.attested_enc_pk == job.enc_pk
    assert len(req.seal_dek()) == 80


def test_no_key_without_a_passing_check(dev):
    req, evidence, job = setup_job()
    job.evidence_hash = os.urandom(32)
    with pytest.raises(AttestationRejected):
        req.check_evidence(dev, evidence, job, mainnet=False)
    with pytest.raises(AttributeError):
        req.seal_dek()


def test_mock_is_never_valid_on_mainnet(dev):
    req, evidence, job = setup_job()
    with pytest.raises(AttestationRejected, match="mainnet"):
        req.check_evidence(dev, evidence, job, mainnet=True)


@pytest.mark.parametrize("policy", [
    {}, {"mock": {}}, {"mock": {"allow": False, "allowed_measurements": [MEASUREMENT]}},
    {"mock": {"allow": True}}, {"mock": {"allow": True, "allowed_measurements": []}}])
def test_policies_reject_by_default(policy, tmp_path):
    req, evidence, job = setup_job()
    with pytest.raises(AttestationRejected):
        req.check_evidence(EvidenceVerifier(policy, tmp_path / "s.json"), evidence, job, mainnet=False)


def test_measurement_outside_the_allowlist(dev):
    req, evidence, job = setup_job(measurement=os.urandom(32).hex())
    with pytest.raises(AttestationRejected, match="allowlist"):
        req.check_evidence(dev, evidence, job, mainnet=False)


def test_evidence_for_another_job(dev):
    req, evidence, job = setup_job()
    req.job_id = os.urandom(32)
    with pytest.raises(AttestationRejected, match="bind"):
        req.check_evidence(dev, evidence, job, mainnet=False)


def test_keys_on_chain_differ_from_the_evidence(dev):
    req, evidence, job = setup_job()
    job.enc_pk = os.urandom(32)
    with pytest.raises(AttestationRejected, match="bind"):
        req.check_evidence(dev, evidence, job, mainnet=False)


def test_tampered_evidence(dev):
    req, evidence, job = setup_job()
    forged = json.loads(json.dumps(evidence))
    forged["attestation"]["network"]["isolated"] = False
    job.evidence_hash = evidence_hash(forged)
    with pytest.raises(AttestationRejected, match="bad mock signature"):
        req.check_evidence(dev, forged, job, mainnet=False)


def test_the_offers_tier_claim_must_match_the_evidence(dev):
    req, evidence, job = setup_job(tier=2)  # a T2 offer backed by MOCK evidence
    with pytest.raises(AttestationRejected, match="tier"):
        req.check_evidence(dev, evidence, job, mainnet=False)


def test_job_measurement_must_match_the_evidence(dev):
    req, evidence, job = setup_job()
    job.measurement = os.urandom(32)
    with pytest.raises(AttestationRejected, match="measurement"):
        req.check_evidence(dev, evidence, job, mainnet=False)


def test_unenrolled_tpm_is_rejected(dev):
    req, _, job = setup_job(tier=3)
    evidence = {"type": "TPM2", "enrollment_id": "nobody", "quote": "", "signature": "", "pcrs": {}}
    job.evidence_hash = evidence_hash(evidence)
    with pytest.raises(AttestationRejected, match="not enrolled"):
        req.check_evidence(dev, evidence, job, mainnet=False)


def test_real_tdx_quote_not_bound_to_this_job_is_rejected(tmp_path):
    req, _, job = setup_job(tier=2)
    import base64

    evidence = {"type": "TDX", "quote": base64.b64encode((FIX / "tdx" / "tdx_quote").read_bytes()).decode(),
                "collateral": (FIX / "tdx" / "tdx_quote_collateral.json").read_text()}
    job.evidence_hash = evidence_hash(evidence)
    policy = {"tdx": {"allowed_tcb_status": ["UpToDate"], "allowed_measurements": [MEASUREMENT]}}
    with pytest.raises(AttestationRejected):
        req.check_evidence(EvidenceVerifier(policy, tmp_path / "s.json"), evidence, job, mainnet=False)


@pytest.mark.parametrize("evidence", [{}, {"type": "SGX"}, {"type": "TDX"}, {"type": "TPM2"}])
def test_malformed_evidence_is_rejected(evidence, dev):
    with pytest.raises(Rejected):
        dev.verify(evidence, bytes(32))
