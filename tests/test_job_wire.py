"""One job over a real loopback link: what a seller will take and what it hands back."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import node  # noqa: E402,F401

import beetle_crypto as bc  # noqa: E402
import pytest  # noqa: E402
from net import transfer  # noqa: E402
from net.peer import Peer  # noqa: E402
from node import wire  # noqa: E402
from node.daemon import JobBox  # noqa: E402
from requester import AttestationRejected, Requester  # noqa: E402
from solders.keypair import Keypair  # noqa: E402


def prepared(tmp_path: Path):
    wallet = Keypair()
    req = Requester(bc.new_key(), bytes(wallet.pubkey()))
    root = tmp_path / "buyer"
    job = req.prepare_job(root, [{"topic": "a", "answer": "b"}], [{"question": "c", "answer": "d"}],
                          {"id": "x", "revision": "r", "manifest_sha256": "0" * 64}, {"r": 1}, {"a": 1},
                          {"max_train_tokens": 10, "max_wall_seconds": 1})
    spec = json.loads((root / "in" / "spec.json").read_text())
    onchain = SimpleNamespace(job_id=bytes.fromhex(job["job_id"]), spec_hash=bytes.fromhex(job["spec_hash"]),
                              requester=wallet.pubkey(), terms={"units": 10, "price_mode": 0})
    return wallet, root, spec, onchain


async def linked():
    a, b = await Peer.bind(), await Peer.bind()
    buyer, seller = await asyncio.gather(a.link(b.id, b.addrs, True), b.link(a.id, a.addrs, False))
    return a, b, buyer, seller


def test_all_inputs_share_one_size_tier_and_a_named_root(tmp_path):
    _, root, spec, _ = prepared(tmp_path)
    sizes = {(root / "in" / f"{n}.enc").stat().st_size for n in spec["inputs"]}
    assert len(sizes) == 1
    for name, value in spec["input_roots"].items():
        m = transfer.manifest(root / "in" / f"{name}.enc")
        assert transfer.root_of(m).hex() == value and m["sha256"] == spec["inputs"][name]


def test_secrets_survive_a_restart(tmp_path):
    wallet, root, spec, _ = prepared(tmp_path)
    req = Requester(bc.new_key(), bytes(wallet.pubkey()))
    key = req.storage_key
    req2 = Requester(key, bytes(wallet.pubkey()))
    req.job_id, req.spec_hash, req.nonce, req.open_slot = b"j" * 32, b"s" * 32, b"n" * 32, 7
    restored = Requester.restore(key, bytes(wallet.pubkey()), req.secrets())
    sealed = bc.hpke_seal(req.enc_key.public_bytes, b"k" * 32, info=b"x")
    assert restored.enc_key.open(sealed, info=b"x") == b"k" * 32
    assert restored.dek == req.dek and restored.dek != req2.dek


async def _inputs_then_results(tmp_path: Path) -> dict:
    wallet, root, spec, onchain = prepared(tmp_path)
    box = JobBox(onchain, tmp_path / "seller")
    box.prepare()
    a, b, buyer, seller = await linked()
    server = asyncio.create_task(box.handle(seller))
    got = {}

    async def take(kind: str, out: Path) -> dict:
        await wire.send_msg(buyer, {"t": kind})
        answer = await wire.recv_msg(buyer)
        if answer["t"] == kind:
            for name, file_root in answer["files"]:  # the seller's order, whatever it is
                await transfer.receive(buyer, bytes.fromhex(file_root), out / name)
        return answer

    try:
        await wire.send_msg(buyer, {"t": "spec", "spec": {**spec, "nonce": "00"}})
        got["refused"] = (await wire.recv_msg(buyer))["t"]

        await wire.send_msg(buyer, {"t": "spec", "spec": spec})
        need = await wire.recv_msg(buyer)
        for name in need["names"]:
            await transfer.send(buyer, root / "in" / f"{name}.enc")
        await asyncio.wait_for(box.inputs.wait(), 10)
        got["same"] = all(bc.file_sha256(box.inbox / f"{n}.enc") == spec["inputs"][n] for n in spec["inputs"])

        out = tmp_path / "got"
        out.mkdir()
        got["waiting"] = (await take("results", out))["t"]

        adapter = b"cipher" * 5000
        report = {"adapter_ct_hash": bc.sha256(adapter).hex(), "sealed_adapter_key": "ab" * 80}
        (box.outbox / "report.json").write_text(json.dumps(report))
        (box.outbox / "report.sig").write_text("sig")
        (box.outbox / "adapter.enc").write_bytes(adapter)
        box.sealed = [box.outbox / "adapter.enc"]

        answer = await take("results", out)
        got["first_files"] = [name for name, _ in answer["files"]]
        got["early_report"] = (await take("report", out))["t"]
        got["key_on_disk"] = (out / "report.json").exists()

        bad = bytes(Keypair().sign_message(b"nope"))
        await wire.send_msg(buyer, {"t": "ack", "sig": bad.hex()})
        await wire.recv_msg(buyer)
        got["forged"] = box.acked.is_set()
        digest = wire.ack_digest(onchain.job_id, report["adapter_ct_hash"])
        await wire.send_msg(buyer, {"t": "ack", "sig": bytes(wallet.sign_message(digest)).hex()})
        await wire.recv_msg(buyer)
        got["acked"] = box.acked.is_set()

        box.submitted = True  # submit_result landed
        answer = await take("report", out)
        got["later_files"] = sorted(name for name, _ in answer["files"])
        got["key"] = json.loads((out / "report.json").read_text())["sealed_adapter_key"]
        got["copied"] = (out / "adapter.enc").read_bytes() == adapter

        await wire.send_msg(buyer, {"t": "done", "sig": bad.hex()})
        await wire.recv_msg(buyer)
        got["forged_done"] = box.taken
        sig = bytes(wallet.sign_message(wire.taken_digest(onchain.job_id)))
        await wire.send_msg(buyer, {"t": "done", "sig": sig.hex()})
        await wire.recv_msg(buyer)
        got["taken"] = box.taken
        return got
    finally:
        server.cancel()
        await a.close()
        await b.close()


def test_the_adapter_key_stays_back_until_the_result_is_on_chain(tmp_path):
    got = asyncio.run(_inputs_then_results(tmp_path))
    assert got == {"refused": "refused", "same": True, "waiting": "wait", "first_files": ["adapter.enc"],
                   "early_report": "wait", "key_on_disk": False, "forged": False, "acked": True,
                   "later_files": ["report.json", "report.sig"], "key": "ab" * 80, "copied": True,
                   "forged_done": False, "taken": True}


async def _vetted(tmp_path: Path) -> tuple[str, str]:
    _, root, spec, onchain = prepared(tmp_path)

    box = JobBox(onchain, tmp_path / "seller", vet=lambda spec: "budget_not_paid_for")
    box.prepare()
    a, b, buyer, seller = await linked()
    server = asyncio.create_task(box.handle(seller))
    try:
        await wire.send_msg(buyer, {"t": "spec", "spec": spec})
        answer = await wire.recv_msg(buyer)
        await asyncio.wait_for(box.inputs.wait(), 5)
        return answer["t"], box.refused
    finally:
        server.cancel()
        await a.close()
        await b.close()


def test_a_budget_the_escrow_does_not_pay_for_is_refused_before_any_work(tmp_path):
    kind, reason = asyncio.run(_vetted(tmp_path))
    assert kind == "refused" and reason == "budget_not_paid_for"


# --- a tier with nothing to prove ------------------------------------------------------


def no_evidence(tmp_path: Path, tamper=lambda att: None):
    """A T3-C job as a seller would start it, with the sandbox's own claim."""
    wallet = Keypair()
    req = Requester(bc.new_key(), bytes(wallet.pubkey()))
    req.prepare_job(tmp_path / "b", [{"topic": "a", "answer": "b"}], [{"question": "c", "answer": "d"}],
                    {"id": "x", "revision": "r", "manifest_sha256": "0" * 64}, {"r": 1}, {"a": 1},
                    {"max_train_tokens": 10, "max_wall_seconds": 1})
    sign, enc = bc.new_signing_key(), bc.HpkeKeyPair()
    measurement = "ab" * 32
    report_data = bc.digest("start", req.job_id + req.spec_hash + bc.public_bytes(sign) + enc.public_bytes)
    att = {"tier": "MOCK", "mock": True, "measurement": measurement, "job_id": req.job_id.hex(),
           "spec_hash": req.spec_hash.hex(), "sign_pk": bc.public_bytes(sign).hex(), "enc_pk": enc.public_bytes.hex(),
           "report_data": report_data.hex(), "network": {"isolated": True}}
    tamper(att)
    sig = sign.sign(bc.digest("attest", bc.canonical_json(att))).hex()
    evidence = {"type": "NONE", "attestation": att, "signature": sig}
    job = SimpleNamespace(tier=4, sign_pk=bc.public_bytes(sign), enc_pk=enc.public_bytes,
                          measurement=bytes.fromhex(measurement),
                          evidence_hash=bc.sha256(bc.canonical_json(evidence)))
    return req, evidence, job


def test_t3c_needs_no_policy_and_says_it_verified_nothing(tmp_path):
    req, evidence, job = no_evidence(tmp_path)
    verdict = req.check_evidence(None, evidence, job, mainnet=True)
    assert verdict.evidence_type == "NONE" and verdict.tier == 4


def test_t3c_still_checks_that_the_claim_is_about_this_job(tmp_path):
    req, evidence, job = no_evidence(tmp_path, tamper=lambda att: att.update(report_data="00" * 32))
    with pytest.raises(AttestationRejected):
        req.check_evidence(None, evidence, job, mainnet=False)


def test_a_t3c_job_takes_no_other_kind_of_evidence(tmp_path):
    req, evidence, job = no_evidence(tmp_path)
    evidence = {**evidence, "type": "MOCK"}
    job.evidence_hash = bc.sha256(bc.canonical_json(evidence))
    with pytest.raises(AttestationRejected):
        req.check_evidence(None, evidence, job, mainnet=False)
