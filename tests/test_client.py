"""The buyer's rules, without a chain: floors, budgets, proof wording, the local API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import client  # noqa: E402,F401
from client import doctor, policy, proof, verify  # noqa: E402
from client.datasets import Datasets, hidden_questions, train_tokens  # noqa: E402
from client.errors import BeetleError  # noqa: E402
from client.policy import Approvals, Ledger, Policy, accepts, strictest  # noqa: E402
import model_profiles as mp  # noqa: E402
from node import wire  # noqa: E402

LIMITS = {"per_job_usd": 5.0, "per_day_usd": 8.0, "auto_approve_under_usd": 1.0}


def ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "ledger.json")


def approvals(tmp_path) -> Approvals:
    return Approvals(tmp_path / "approvals.json")


# --- floors --------------------------------------------------------------------------


def test_only_public_data_can_go_to_a_machine_without_protection():
    assert accepts("public", "T3-C")
    for floor in ("T3-B", "T1", "T2", "T2+", "T0"):
        assert not accepts(floor, "T3-C")


def test_t0_accepts_nothing_remote_and_t2_plus_waits_for_its_flags():
    for tier in ("T3-C", "T3-B", "T1", "T2"):
        assert not accepts("T0", tier) and not accepts("T2+", tier)


def test_the_strictest_of_the_three_floors_wins():
    assert strictest("public", "T1", None) == "T1"
    assert strictest("T2", "public", "public") == "T2"
    assert strictest("T2", "T0") == "T0"


def test_a_label_sets_a_floor_a_request_cannot_lower():
    assert strictest("public", policy.LABEL_FLOOR["personal"], "public") == "T1"
    assert strictest("public", policy.LABEL_FLOOR["local_only"], "public") == "T0"


def test_tiers_without_evidence_only_ever_meet_the_public_floor():
    assert policy.CHAIN_TIER[0] == "MOCK" and policy.CHAIN_TIER[4] == "T3-C"
    for tier in ("MOCK", "T3-C", "T3"):
        assert [f for f in policy.FLOORS if accepts(f, tier)] == ["public"]


# --- spending -------------------------------------------------------------------------


def test_over_the_job_limit_stops(tmp_path):
    with pytest.raises(BeetleError) as e:
        ledger(tmp_path).reserve("a", 6.0, "train", LIMITS, approved=True)
    assert e.value.status == 402


def test_a_request_can_only_lower_the_limit(tmp_path):
    with pytest.raises(BeetleError):
        ledger(tmp_path).reserve("a", 2.0, "train", LIMITS, cap=1.5, approved=True)


def test_reserved_and_settled_count_against_the_day_until_refunded(tmp_path):
    book = ledger(tmp_path)
    book.reserve("a", 4.0, "train", LIMITS, approved=True)
    book.update("a", "settled", 4.0)
    book.reserve("b", 4.0, "train", LIMITS, approved=True)
    with pytest.raises(BeetleError) as e:
        book.reserve("c", 0.5, "train", LIMITS, approved=True)
    assert e.value.code == "budget_exceeded"
    book.release("b")
    book.reserve("c", 0.5, "train", LIMITS, approved=True)
    assert Ledger(tmp_path / "ledger.json").summary()["settled"] == 4.0


def test_over_the_automatic_amount_waits_for_a_person(tmp_path):
    book, gate = ledger(tmp_path), approvals(tmp_path)
    request = {"dataset": "d", "usd": 2.0}
    with pytest.raises(BeetleError) as e:
        book.reserve("a", 2.0, "train", LIMITS, approvals=gate, reason=request)
    assert e.value.status == 403 and e.value.code == "approval_required"
    ticket = e.value.extra["approval"]
    what = {**request, "why": "over the automatic amount"}
    assert not gate.granted(what)
    gate.approve(ticket)
    assert gate.granted(what)


def test_an_approval_is_used_once(tmp_path):
    gate, what = approvals(tmp_path), {"dataset": "d", "why": "repeat"}
    ticket = gate.request(what)
    assert gate.take(what) == []  # not approved yet
    gate.approve(ticket)
    assert gate.take(what) == [ticket]
    assert not gate.granted(what) and gate.take(what) == []
    with pytest.raises(BeetleError):
        gate.approve(ticket)
    assert gate.request(what) != ticket


def test_an_approval_given_back_still_stands(tmp_path):
    gate, what = approvals(tmp_path), {"dataset": "d", "why": "repeat"}
    gate.approve(gate.request(what))
    gate.give_back(gate.take(what))
    assert gate.granted(what)


def test_policy_needs_both_settings(tmp_path):
    p = Policy(tmp_path / "policy.json")
    with pytest.raises(BeetleError):
        p.ready()
    p.save(per_job_usd=1, per_day_usd=2, min_tier="T2")
    assert p.ready()["min_tier"] == "T2"


# --- proof cards ----------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["T3-C", "T3-B", "MOCK", "rejected", "nonsense"])
def test_no_comfort_words_without_protection(tmp_path, tier):
    text = json.dumps(proof.card(tier)).lower()
    for word in ("protected", "private", "secure", "confidential server", "safe"):
        assert word not in text.replace("no protection", "").replace("no confidential protection", "")


def test_unknown_states_show_the_worse_card():
    assert proof.card("nonsense")["on_their_machine"] == "the other side can read it"


def test_the_two_lamps_stay_apart_on_an_ordinary_gpu():
    card = proof.card("T3-C", path={"direct": True, "rtt_ms": 30})
    assert card["on_the_way"] == "encrypted" and card["on_their_machine"] == "the other side can read it"
    assert "can see your IP" in card["path"]


def test_a_seller_supplies_no_wording():
    assert "owner" in proof.card("T2", peer={"seller": "x"})["detail"]


# --- the adapter must do something ---------------------------------------------------


def test_identical_answers_are_an_error():
    items = [{"question": "q", "answer": "Pixel"}] * 2
    result = verify.compare(items, ["a", "b"], ["a", "b"])
    with pytest.raises(BeetleError) as e:
        verify.require_effect(result, "the seller's server")
    assert e.value.code == "adapter_no_effect"


def test_changed_answers_are_counted_and_scored():
    items = [{"question": "q", "answer": "Pixel"}, {"question": "r", "answer": "Ines"}]
    result = verify.compare(items, ["I don't know", "Ines?"], ["It is Pixel", "Ines?"])
    assert result == {"asked": 2, "changed": 1, "base_correct": 1, "adapter_correct": 2}
    assert verify.require_effect(result, "here") is result


def test_probes_stand_in_when_there_are_no_hidden_questions():
    assert verify.questions(None) == verify.FIXED_PROBES
    assert len(verify.questions([{"question": "q", "answer": "a"}] * 20)) == verify.MAX_ITEMS


# --- datasets ------------------------------------------------------------------------


def write_memories(path: Path) -> Path:
    path.write_text(json.dumps([{"topic": "cat's name", "answer": "Pixel"}]))
    return path


def test_a_dataset_is_used_by_name_and_must_not_change(tmp_path):
    registry = Datasets(tmp_path / "datasets.json")
    source = write_memories(tmp_path / "m.json")
    registry.add(source, "ds", "personal")
    memories, hidden = registry.load("ds")
    assert memories[0]["answer"] == "Pixel" and hidden == hidden_questions(memories)
    source.write_text("[]")
    with pytest.raises(BeetleError):
        registry.load("ds")


def test_an_unregistered_name_is_refused(tmp_path):
    with pytest.raises(BeetleError) as e:
        Datasets(tmp_path / "datasets.json").get("~/.ssh")
    assert e.value.status == 404


def test_labels_and_shape_are_checked(tmp_path):
    registry = Datasets(tmp_path / "datasets.json")
    with pytest.raises(BeetleError):
        registry.add(write_memories(tmp_path / "m.json"), "ds", "whatever")
    (tmp_path / "bad.json").write_text('{"a": 1}')
    with pytest.raises(BeetleError):
        registry.add(tmp_path / "bad.json", "ds", "personal")


def test_token_estimate_grows_with_data_and_epochs():
    one = [{"topic": "x", "answer": "y"}]
    assert train_tokens(one * 2, 4) > train_tokens(one, 4) > 0
    assert train_tokens(one, 8) == pytest.approx(2 * train_tokens(one, 4), abs=2)


# --- doctor --------------------------------------------------------------------------


def caps(budget: float, engine="mlx", power=True, disk=20) -> dict:
    return {"budget_gb": budget, "local_engine": engine, "on_power": power, "disk_gb": disk}


def test_paths_follow_memory():
    p = SimpleNamespace(mlx={"infer_gb": 1.5, "train_gb": 6.0})
    assert doctor.path_for(caps(8.0), p)["path"] == "local"
    assert doctor.path_for(caps(4.0), p)["path"] == "hybrid"
    assert doctor.path_for(caps(1.0), p)["path"] == "remote"
    assert doctor.path_for(caps(8.0, power=False), p)["path"] == "hybrid"
    assert doctor.path_for(caps(8.0, engine=None), p)["path"] == "remote"


# --- the local API -------------------------------------------------------------------


@pytest.fixture
def api(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from client.core import Core
    from client.daemon import add_token, build, owner_token
    from node import wire

    monkeypatch.setenv("BEETLE_HOME", str(tmp_path))
    wire.write_json(tmp_path / "config.json", {"chain": None, "rendezvous": {"kind": "file", "dir": "x"}})
    core = Core(tmp_path)
    core.policy.save(per_job_usd=1, per_day_usd=2, min_tier="T2", auto_approve_under_usd=1)
    client = TestClient(build(core), base_url="http://127.0.0.1:8741")
    return SimpleNamespace(client=client, core=core, owner=owner_token(tmp_path), agent=add_token(tmp_path, "bot"))


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_no_token_no_answer(api):
    assert api.client.get("/beetle/health").status_code == 401
    assert api.client.get("/beetle/health", headers=bearer("bk_local_wrong")).status_code == 401
    assert api.client.get("/beetle/health", headers=bearer(api.agent)).status_code == 200


def test_health_names_the_api_so_a_client_can_tell_an_old_daemon(api):
    health = api.client.get("/beetle/health", headers=bearer(api.owner)).json()
    assert health["api"] >= 1 and health["chain"] is None  # the app restarts a daemon that predates `beetle init`


def test_an_mlx_too_old_for_our_models_is_no_engine(monkeypatch):
    old = SimpleNamespace()  # no device_info, like the last MLX for macOS 13
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=old))
    monkeypatch.setitem(sys.modules, "mlx.core", old)
    assert doctor._mac()["local_engine"] is None


def test_another_host_header_is_refused(api):
    r = api.client.get("/beetle/health", headers={**bearer(api.owner), "host": "evil.example"})
    assert r.status_code == 403


def test_an_agent_can_ask_for_a_higher_limit_but_not_set_one(api):
    r = api.client.post("/beetle/limits", json={"per_job_usd": 1000}, headers=bearer(api.agent))
    assert r.status_code == 202 and "needs_approval" in r.json()
    assert api.client.get("/beetle/limits", headers=bearer(api.agent)).json()["per_job_usd"] == 1
    r = api.client.post(f"/beetle/approvals/{r.json()['needs_approval']}", headers=bearer(api.agent))
    assert r.status_code == 403


def test_bad_input_is_the_callers_error(api):
    owner = bearer(api.owner)
    r = api.client.post("/beetle/approvals/no_such_ticket", headers=owner)
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = api.client.post("/beetle/quote", json={"base": "qwen3.5-2b"}, headers=owner)
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
    r = api.client.get("/beetle/market?kind=nope", headers=owner)
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"


def test_idle_connections_outlast_the_apps_polling():
    from client.daemon import listening
    assert listening(None, 8741).timeout_keep_alive > 30  # the page polls jobs every 15 and the wallet every 30


def test_a_job_needs_both_policies_and_a_key(api):
    r = api.client.post("/beetle/jobs", json={"dataset": "d", "base": "qwen3.5-2b"}, headers=bearer(api.agent))
    assert r.status_code == 400


def test_a_t0_floor_never_goes_remote(api):
    api.core.policy.save(min_tier="T0")
    r = api.client.post("/v1/chat/completions", headers=bearer(api.agent), json={
        "model": "bolt-e4b", "messages": [{"role": "user", "content": "hi"}], "beetle": {"route": "remote"}})
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_route_meets_floor"


def test_models_list_aliases_and_bases(api):
    api.core.models.add("me", {"job": "j", "base": "qwen3.5-2b", "label": "synthetic", "dataset": "d",
                               "adapter_sha256": "ab", "at": 0})
    ids = [m["id"] for m in api.client.get("/v1/models", headers=bearer(api.agent)).json()["data"]]
    assert ids[0] == "me" and "qwen3.5-2b" in ids


def test_rollback_keeps_the_old_version(api):
    for job in ("old", "new"):
        api.core.models.add("me", {"job": job, "base": "qwen3.5-2b", "label": "x", "dataset": "d",
                                   "adapter_sha256": job, "at": 0})
    assert api.core.models.current("me")["job"] == "new"
    api.client.post("/beetle/models/me/rollback", headers=bearer(api.owner))
    assert api.core.models.current("me")["job"] == "old"


# --- routes ---------------------------------------------------------------------------


def stripped(pieces: list[str], end: str | None) -> str:
    import asyncio

    from client.core import without_thinking

    async def source():
        for p in pieces:
            yield p

    async def run():
        return "".join([p async for p in without_thinking(source(), end)])

    return asyncio.run(run())


def test_reasoning_is_dropped_up_to_its_marker():
    assert stripped(["some ", "thought</th", "ink>\n", "Pixel", "."], "</think>") == "Pixel."


def test_a_model_without_a_marker_passes_through():
    assert stripped(["<think>", "a"], None) == "<think>a"
    assert stripped(["no marker here"], "</think>") == "no marker here"


def test_a_second_home_gets_its_own_port():
    import socket

    from client.daemon import DEFAULT_PORT, usable_port

    assert usable_port(9) == 9                  # asked for: not ours to change
    with socket.socket() as held:
        try:
            held.bind(("127.0.0.1", DEFAULT_PORT))
        except OSError:
            pass                                # a real daemon holds it
        assert usable_port(None) != DEFAULT_PORT


@pytest.mark.parametrize("asked, answered", [("2025-06-18", "2025-06-18"), ("2024-11-05", "2024-11-05"),
                                             ("2026-07-28", "2025-11-25"), (None, "2025-11-25")])
def test_mcp_answers_with_a_version_that_has_a_handshake(asked, answered):
    from client.mcp_server import handle

    msg = {"id": 1, "method": "initialize", "params": {"protocolVersion": asked}}
    assert handle(None, msg)["result"]["protocolVersion"] == answered


@pytest.mark.parametrize("tool, args", [("proof", {"id": "../wallet"}), ("proof", {"id": ".."}), ("proof", {"id": "a?b"}),
                                        ("job_status", {"job_id": "../wallet"}), ("fetch_adapter", {"job_id": "../wallet"}),
                                        ("cancel_job", {"job_id": "x/../../wallet"})])
def test_an_mcp_id_cannot_walk_to_another_route(tool, args):
    from client.mcp_server import handle

    asked = []
    api = SimpleNamespace(get=lambda *a: asked.append(a), post=lambda *a: asked.append(a))
    reply = handle(api, {"id": 1, "method": "tools/call", "params": {"name": tool, "arguments": args}})
    assert reply["error"]["code"] == -32602 and not asked


def test_mcp_ids_reach_their_routes():
    from client.mcp_server import run_tool

    api = SimpleNamespace(get=lambda path: path, post=lambda path, *a: path)
    assert run_tool(api, "proof", {"id": "prf_0a1b"}) == "/beetle/proof/prf_0a1b"
    assert run_tool(api, "job_status", {"job_id": "job_0001"}) == "/beetle/jobs/job_0001"
    assert run_tool(api, "cancel_job", {"job_id": "0001"}) == "/beetle/jobs/0001/cancel"


def test_job_status_answers_when_the_stage_changes(monkeypatch):
    from client import mcp_server

    stages = iter(["escrowed", "escrowed", "uploading", "evidence"])
    api = SimpleNamespace(get=lambda path: {"stage": next(stages)})
    monkeypatch.setattr(mcp_server.time, "sleep", lambda s: None)
    assert mcp_server.run_tool(api, "job_status", {"job_id": "0001", "wait_s": 30})["stage"] == "uploading"


def test_job_status_does_not_wait_on_a_finished_job(monkeypatch):
    from client import mcp_server

    def never(seconds):
        raise AssertionError("slept on a finished job")

    monkeypatch.setattr(mcp_server.time, "sleep", never)
    api = SimpleNamespace(get=lambda path: {"stage": "delivered"})
    assert mcp_server.run_tool(api, "job_status", {"job_id": "0001", "wait_s": 30})["stage"] == "delivered"


def test_job_status_wants_a_number_of_seconds():
    from client.mcp_server import handle

    api = SimpleNamespace(get=lambda path: {"stage": "escrowed"})
    params = {"name": "job_status", "arguments": {"job_id": "0001", "wait_s": "soon"}}
    assert handle(api, {"id": 1, "method": "tools/call", "params": params})["error"]["code"] == -32602


@pytest.mark.parametrize("reported, shown", [("NVIDIA A100-SXM4-80GB", "NVIDIA A100"), ("NVIDIA L4", "NVIDIA L4"),
                                             ("Tesla T4", "NVIDIA T4"), ("NVIDIA A10-24Q", "NVIDIA A10"),
                                             ("Quadro RTX 8000", "NVIDIA")])
def test_a_gpu_name_survives_the_trip_through_an_offer(reported, shown):
    assert wire.gpu_name(wire.hardware_class({"name": reported})) == shown


def test_hardware_with_no_gpu_name_shows_none():
    assert wire.gpu_name(wire.hardware_class(None)) is None
    assert wire.gpu_name(wire.HW_NVIDIA) == "NVIDIA"  # listed before names went on chain


def test_weaknesses_fit_the_tier():
    plain = proof.card("T3-C")["known_weaknesses"]
    assert "read all of it" in plain[0] and not any("GPU evidence" in w for w in plain)
    assert any("GPU evidence" in w for w in proof.card("T2")["known_weaknesses"])
    assert proof.card("T0")["known_weaknesses"] == proof.card("rejected")["known_weaknesses"] == []
    assert proof.card("T3")["known_weaknesses"] == plain  # a tier we don't know gets the worse answer


def test_local_use_is_not_a_failed_route(api):
    r = api.client.get("/beetle/wallet", headers=bearer(api.owner))
    assert r.status_code == 409 and r.json()["error"]["code"] == "local_only"


@pytest.mark.parametrize("sol, usdc, said", [(0, 5.0, "no SOL"), (1_000_000_000, 0.5, r"holds \$0.50"),
                                             (1_000_000_000, RuntimeError("no account"), r"holds \$0.00")])
def test_an_empty_wallet_says_so_before_any_transaction(api, sol, usdc, said):
    def balance(me):
        if isinstance(usdc, Exception):
            raise usdc
        return usdc

    api.core.wallet = SimpleNamespace(pubkey=lambda: "me")
    api.core.chain = SimpleNamespace(sol_lamports=lambda me: sol, token_balance=balance)
    with pytest.raises(BeetleError, match=said) as e:
        api.core.require_funds(1.0)
    assert e.value.code == "no_funds"


def test_a_funded_wallet_passes(api):
    api.core.wallet = SimpleNamespace(pubkey=lambda: "me")
    api.core.chain = SimpleNamespace(sol_lamports=lambda me: 1_000_000_000, token_balance=lambda me: 1.0)
    api.core.require_funds(1.0)


def test_a_receipt_says_what_a_job_was(api):
    rec = {"base": "qwen3.5-2b", "stage": "settled", "dataset_sha256": "d" * 64, "job_id": "a" * 64, "usd": 1.2,
           "gpu": {"name": "NVIDIA A100", "memory_gb": 80}, "tx": {"escrow": "sig1", "settlement": "sig2"},
           "timeline": [{"stage": "quoted", "at": 100}, {"stage": "settled", "at": 247}],
           "result": {"passed": True, "paid_usd": 1.0, "scores": {"before": 0.0, "after": 1.0},
                      "effect": {"asked": 6, "base_correct": 0, "adapter_correct": 6}}}
    out = api.core.receipt(SimpleNamespace(rec=rec))
    assert out["gpu"]["name"] == "NVIDIA A100" and out["duration_s"] == 147 and out["cost_usd"] == 1.0
    assert out["evaluation"] == {"passed": True, "before": 0.0, "after": 1.0, "asked": 6,
                                 "base_correct": 0, "adapter_correct": 6}
    assert out["settlement"] == "settled" and out["transactions"]["escrow"] == "sig1"
    assert out["model_revision"] == mp.get("qwen3.5-2b").revision


def test_the_full_job_id_finds_a_job(api):
    job = SimpleNamespace(rec={"job_id": "a" * 64})
    api.core.jobs["a" * 16] = job
    assert api.core.job("a" * 64) is job and api.core.job("a" * 16) is job and api.core.job("job_aaaa") is job
    with pytest.raises(BeetleError):
        api.core.job("a" * 63 + "b")


def test_a_running_job_has_no_evidence_yet(api):
    rec = {"base": "qwen3.5-2b", "stage": "escrowed", "dataset_sha256": "d" * 64, "job_id": "a" * 64, "usd": 1.0,
           "timeline": []}
    api.core.jobs["a" * 16] = SimpleNamespace(rec=rec)
    found = api.core.proof("a" * 16)
    assert found["tier"] == "no_evidence" and found["headline"] == "No evidence yet" and found["color"] != "red"
    assert found["on_their_machine"] == "ciphertext only" and found["known_weaknesses"] == []
    assert found["receipt"]["settlement"] == "in escrow"


def test_a_job_from_before_receipts_shows_what_it_has(api):
    rec = {"base": "qwen3.5-2b", "stage": "training", "dataset_sha256": "d", "job_id": "a", "usd": 1.2, "timeline": []}
    out = api.core.receipt(SimpleNamespace(rec=rec))
    assert out["gpu"] is None and out["transactions"] == {} and out["settlement"] == "in escrow"
    assert out["cost_usd"] is None and out["held_usd"] == 1.2


def test_a_job_that_ended_holds_nothing(api):
    rec = {"base": "qwen3.5-2b", "stage": "failed", "ended": "refunded", "dataset_sha256": "d", "job_id": "a",
           "usd": 1.2, "timeline": []}
    out = api.core.receipt(SimpleNamespace(rec=rec))
    assert out["held_usd"] is None and out["cost_usd"] is None and out["settlement"] == "refunded"


def test_open_jobs_show_addresses_and_amounts_only(api):
    job = SimpleNamespace(job_id=bytes(range(32)), requester="buyer", host="seller", price=1_500_000,
                          state="Running", opened_at=7, sealed_dek=b"never shown")
    api.core.chain = SimpleNamespace(open_jobs=lambda: [job])
    assert api.client.get("/beetle/chain/jobs", headers=bearer(api.agent)).json() == [
        {"job": "0001020304050607", "buyer": "buyer", "seller": "seller", "usd": 1.5, "state": "Running", "opened_at": 7}]


def refused_by(tmp_path, monkeypatch, pid: int) -> list:
    """Point an Api at a port that answers 401, with `pid` recorded as this home's daemon."""
    import httpx

    from client.api import Api

    (tmp_path / "serve.json").write_text(json.dumps({"port": 1, "pid": pid}))
    monkeypatch.setattr("client.api.wire.home", lambda: tmp_path)
    monkeypatch.setattr(Api, "_client", lambda self, port: httpx.Client(
        base_url="http://daemon", transport=httpx.MockTransport(lambda request: httpx.Response(401))))
    started = []
    monkeypatch.setattr(Api, "_start", lambda self: started.append(True))
    Api("bk_local_wrong")
    return started


def test_a_refused_token_never_starts_a_second_daemon(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="refused this token"):
        refused_by(tmp_path, monkeypatch, os.getpid())


def test_a_stale_record_starts_our_daemon_when_another_holds_the_port(tmp_path, monkeypatch):
    gone = subprocess.Popen([sys.executable, "-c", ""])
    gone.wait()
    assert refused_by(tmp_path, monkeypatch, gone.pid) == [True]
