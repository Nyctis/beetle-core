"""What a seller keeps, for how long, and which jobs it turns down. No chain: a stub
answers what the daemon reads."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import node  # noqa: E402,F401

from beetle_chain import NO_VERIFIER  # noqa: E402
from node import daemon  # noqa: E402
from node.daemon import JobBox, Node  # noqa: E402
from solders.keypair import Keypair  # noqa: E402


class Stub:
    """A chain with one job on it, or none."""

    def __init__(self, job=None):
        self.job, self.calls = job, []

    def get_job(self, job_id):
        return self.job

    def chain_time(self):
        return int(time.time())

    def token_balance(self, owner):
        return 0.0

    def job_pda(self, job_id):
        return Keypair().pubkey()

    def decline_job(self, kp, job):
        self.calls.append("decline")



class Runtime:
    def prepare(self):
        pass


def seller(tmp_path: Path, chain=None, **host) -> Node:
    cfg = {"rendezvous": {"kind": "file", "dir": str(tmp_path / "rdv")},
           "host": {"models": [], "evidence": "mock", **host}}
    return Node(tmp_path, cfg, chain or Stub(), Keypair(), Runtime())


def job(**fields) -> SimpleNamespace:
    now = int(time.time())
    return SimpleNamespace(**{"job_id": bytes(range(32)), "spec_hash": bytes(32), "requester": Keypair().pubkey(),
                              "state": "Settled", "opened_at": now, "start_by": now + 900, "price": 1000,
                              "run_secs": 900, "offer": None, "verifier": NO_VERIFIER, "host": Keypair().pubkey(),
                              "bond": 500, **fields})


def delivered(n: Node, onchain, submitted: bool = True) -> JobBox:
    box = JobBox(onchain, n.where / "host_jobs" / onchain.job_id.hex()[:16])
    box.prepare()
    (box.inbox / "body.enc").write_bytes(b"sealed input")
    (box.inbox / "sealed_dek.bin").write_bytes(b"sealed key")
    (box.outbox / "settle.sig").write_text("sig")
    for name in ("adapter.enc", "report.json", "report.sig", "private_report.enc"):
        (box.outbox / name).write_bytes(b"{}")
    box.sealed, box.submitted, box.finished = [box.outbox / "adapter.enc"], submitted, True
    n.boxes[onchain.job_id] = box
    return box


def run(coro_fn):
    async def main():
        return await coro_fn()
    return asyncio.run(main())


def test_a_settled_job_keeps_its_result_and_loses_its_inputs(tmp_path):
    async def go():
        n, onchain = seller(tmp_path, retain_s=60), job()
        box = delivered(n, onchain)
        n.sweep([])
        assert onchain.job_id in n.boxes and box.keep_until > time.time()
        assert not box.inbox.exists() and not (box.outbox / "settle.sig").exists()
        assert (box.outbox / "adapter.enc").exists() and (box.root / "retained.json").exists()
        assert box.listener is not None
        box.listener.cancel()

    run(go)


def test_a_buyer_can_still_ask_for_it_after_the_account_closed(tmp_path):
    async def go():
        from net import transfer
        from net.peer import Peer
        from node import wire

        n, onchain = seller(tmp_path, retain_s=60), job()
        box = delivered(n, onchain)
        n.sweep([])
        box.listener.cancel()
        a, b = await Peer.bind(), await Peer.bind()
        buyer, there = await asyncio.gather(a.link(b.id, b.addrs, True), b.link(a.id, a.addrs, False))
        server = asyncio.create_task(box.handle(there))
        await wire.send_msg(buyer, {"t": "report"})
        answer = await wire.recv_msg(buyer)
        for name, root in answer["files"]:
            await transfer.receive(buyer, bytes.fromhex(root), tmp_path / name)
        server.cancel()
        await a.close()
        await b.close()
        return sorted(name for name, _ in answer["files"])

    assert run(go) == ["private_report.enc", "report.json", "report.sig"]


def test_a_buyers_signature_ends_the_wait(tmp_path):
    async def go():
        n, onchain = seller(tmp_path, retain_s=60), job()
        box = delivered(n, onchain)
        box.taken = True
        n.sweep([])
        assert onchain.job_id not in n.boxes and not box.root.exists()

    run(go)


def test_the_window_ends(tmp_path):
    async def go():
        n, onchain = seller(tmp_path, retain_s=60), job()
        box = delivered(n, onchain)
        n.sweep([])
        box.keep_until = time.time() - 1
        n.sweep([])
        assert not box.root.exists()

    run(go)


def test_a_job_that_never_submitted_keeps_nothing(tmp_path):
    async def go():
        n, onchain = seller(tmp_path, retain_s=60), job()
        box = delivered(n, onchain, submitted=False)
        n.sweep([])
        assert not box.root.exists()

    run(go)


def test_a_running_job_is_not_touched(tmp_path):
    async def go():
        n, onchain = seller(tmp_path, retain_s=60), job(state="Delivered")
        box = delivered(n, onchain)
        n.sweep([onchain])
        assert box.keep_until is None and box.inbox.exists()

    run(go)


def test_retained_results_survive_a_restart(tmp_path):
    async def go():
        n, onchain = seller(tmp_path, retain_s=60), job()
        box = delivered(n, onchain)
        n.sweep([])
        box.listener.cancel()
        again = seller(tmp_path, retain_s=60)
        again.restore_retained()
        restored = again.boxes[onchain.job_id]
        assert restored.submitted and restored.sealed[0].name == "adapter.enc"
        assert restored.keep_until == box.keep_until and restored.job.requester == onchain.requester
        restored.listener.cancel()

    run(go)


def test_a_silent_buyer_is_declined_soon(tmp_path):
    async def go():
        n = seller(tmp_path, Stub())
        box = JobBox(job(opened_at=int(time.time()) - daemon.NO_SHOW_S), tmp_path / "b")
        assert await n._inputs_arrive(box) is False

    run(go)


def test_a_buyer_who_showed_up_and_went_quiet_is_declined_by_the_start_window(tmp_path):
    async def go():
        n = seller(tmp_path, Stub())
        onchain = job(opened_at=int(time.time()) - 60, start_by=int(time.time()) + 1)
        box = JobBox(onchain, tmp_path / "b")
        box.contacted = True
        assert await n._inputs_arrive(box) is False

    run(go)


def test_an_unknown_verifier_is_declined_at_once(tmp_path):
    async def go():
        chain = Stub()
        n = seller(tmp_path, chain)
        onchain = job(state="Opened", verifier=Keypair().pubkey())
        box = JobBox(onchain, tmp_path / "host_jobs" / "x")
        box.prepare()
        n.boxes[onchain.job_id] = box
        await n.run_job(box)
        assert chain.calls == ["decline"]
        box.listener.cancel()

    run(go)


def test_a_budget_the_chain_did_not_charge_for_is_refused(tmp_path):
    from job_defaults import RUN_RESERVE_S, spec_problem

    terms = {"units": 1000, "price_mode": 0, "run_secs": RUN_RESERVE_S + 600, "grace_s": 60}
    spec = {"inputs": dict.fromkeys(("body", "train", "eval", "program"), "x")}
    fine = {**spec, "budget": {"max_train_tokens": 1000, "max_wall_seconds": 600}}
    assert spec_problem(fine, terms) is None
    assert spec_problem({**spec, "budget": {"max_train_tokens": 10**9, "max_wall_seconds": 600}}, terms)
    assert spec_problem({**spec, "budget": {"max_train_tokens": 1000, "max_wall_seconds": 10**6}}, terms)


def test_off_t2_a_buyer_gets_time_to_accept_before_finalize(tmp_path):
    n = seller(tmp_path, accept_wait_s=1234)
    assert n._accept_wait(job(tier=0, outcome=daemon.OUTCOME_PASSED)) == 1234
    assert n._accept_wait(job(tier=daemon.TIER_T2, outcome=daemon.OUTCOME_PASSED)) == 0


def test_a_job_that_did_not_pass_does_not_wait_for_an_accept(tmp_path):
    n = seller(tmp_path, accept_wait_s=1234)
    assert n._accept_wait(job(tier=0, outcome=daemon.OUTCOME_PASSED + 3)) == 0


def test_the_daemon_never_loads_a_wallet_into_the_environment():
    source = (ROOT / "node/environment.py").read_text()
    assert "load_wallet" not in source and "wallet.json" not in source.split('"""')[2]
    assert "wallet" not in (ROOT / "node/metered.py").read_text().split("class MeteredEnvironment")[1] \
        .split("class Settler")[0]


def test_retained_marker_is_plain_public_data(tmp_path):
    async def go():
        n, onchain = seller(tmp_path, retain_s=60), job()
        box = delivered(n, onchain)
        n.sweep([])
        box.listener.cancel()
        saved = json.loads((box.root / "retained.json").read_text())
        assert set(saved) == {"job_id", "requester", "adapter", "keep_until"}

    run(go)


def test_a_sandbox_that_outlives_its_deadline_is_stopped(monkeypatch):
    import subprocess

    from node import runtime

    calls = []

    def run(cmd, **kw):
        calls.append(cmd[1])
        if cmd[1] == "wait":
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime.DockerRuntime(False).wait("job", 5) is False
    assert calls == ["wait", "kill"]


def test_an_offer_follows_the_sandbox_it_will_run(tmp_path, monkeypatch):
    from beetle_chain import OFFER_ACTIVE, OFFER_PAUSED

    stale = SimpleNamespace(status=OFFER_ACTIVE, measurement=b"old", address="a")
    paused = SimpleNamespace(status=OFFER_PAUSED, measurement=b"old", address="b")
    current = SimpleNamespace(status=OFFER_ACTIVE, measurement=b"new", address="c")
    updated = []
    chain = Stub()
    chain.update_offer = lambda wallet, offer, terms: updated.append(offer.address)
    monkeypatch.setattr(daemon, "offers_of", lambda chain, authority: [stale, paused, current])
    monkeypatch.setattr(daemon, "measurement", lambda host: b"new")
    monkeypatch.setattr(daemon, "terms_of", lambda offer, **changes: changes)
    seller(tmp_path, chain).refresh_offers()
    assert updated == ["a"]


def test_a_sandbox_that_will_not_start_is_declined_not_left_to_the_slash(tmp_path):
    class Silent(Runtime):
        stopped = False

        def launch(self, name, model, inbox, outbox):
            pass

        def attested(self, name, outbox, timeout):
            return None

        def stop(self, name):
            return ""

    async def go():
        chain = Stub()
        n = seller(tmp_path, chain)
        n.runtime = Silent()
        n._model_for = lambda spec: Path(".")
        onchain = job(state="Opened")
        box = JobBox(onchain, tmp_path / "host_jobs" / "y")
        box.prepare()
        box.spec = {}
        box.inputs.set()
        await n.run_job(box)
        assert chain.calls == ["decline"]
        box.listener.cancel()

    run(go)


def test_a_model_on_someone_elses_gpu_needs_none_here():
    import model_profiles as mp
    import pytest
    from node.cli import check_memory

    qwen = mp.get("qwen3.5-2b")
    check_memory([qwen], 0, True, ["qwen3.5-2b"])
    with pytest.raises(SystemExit):
        check_memory([qwen], 0, True, [])
    with pytest.raises(SystemExit):
        check_memory([qwen], 0, False, ["qwen3.5-2b"])  # training does run here


def test_vllm_serve_starts_a_server_that_takes_adapters(tmp_path, monkeypatch):
    import model_profiles as mp
    from node import cli

    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda argv, **options: calls.append(argv))
    monkeypatch.setattr(cli, "default_scratch", lambda: tmp_path / "scratch")
    host = {"vllm": {"port": 8012, "gpu_fraction": 0.2}}
    cli.start_vllm(tmp_path, host, mp.get("qwen3.5-2b"))
    command = calls[-1]
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True" in command and "--enable-lora" in command
    assert f"{tmp_path}:{tmp_path}:ro" in command  # the parent, so a reboot leaves the scratch ours
    assert host["backend"] == "http://127.0.0.1:8012"
