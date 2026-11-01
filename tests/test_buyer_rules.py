"""The buyer's side of the review's findings: nothing private leaves before the
floor passed, a result can be recovered, and a job can move to the next seller."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import client  # noqa: E402,F401

import model_profiles as mp  # noqa: E402
import client.remote as remote  # noqa: E402
from beetle_chain import KIND_INFERENCE, REACH_ECHO, chain_ref_for  # noqa: E402
from client.core import Core, Target  # noqa: E402
from client.datasets import paid_units  # noqa: E402
from client.errors import BeetleError  # noqa: E402
from client.jobs import JobEnded, TrainJob, training_note  # noqa: E402
from client.policy import accepts  # noqa: E402
from client.tokens import MESSAGE_OVERHEAD, TokenCounter  # noqa: E402
from node import wire  # noqa: E402
from solders.keypair import Keypair  # noqa: E402

BASE = mp.get("qwen3.5-2b")


class Refuses:
    """A chain that fails the test if anything moves."""

    chain_ref = bytes(32)

    def list_offers(self, **kw):
        return [SimpleNamespace(kind=KIND_INFERENCE, tier=2, model_root=BASE.root, price_a=1, price_b=1,
                                reach=0, public_contact=None)]

    def __getattr__(self, name):
        raise AssertionError(f"chain.{name} must not be reached")


@pytest.fixture
def core(tmp_path, monkeypatch):
    monkeypatch.setenv("BEETLE_HOME", str(tmp_path))
    wire.write_json(tmp_path / "config.json", {"chain": None, "rendezvous": {"kind": "file", "dir": "x"}})
    c = Core(tmp_path)
    c.policy.save(per_job_usd=1, per_day_usd=2, min_tier="public", auto_approve_under_usd=1)
    return c


def test_a_claimed_tier_buys_an_inference_offer_nothing(core):
    core.chain = Refuses()
    assert core._matching(KIND_INFERENCE, BASE, "T2") == []
    assert len(core._matching(KIND_INFERENCE, BASE, "public")) == 1


def test_a_floor_above_t3c_stops_before_any_channel_or_upload(core):
    core.chain = Refuses()
    target = Target("me", BASE, "job", label="personal", dataset="d")

    async def go():
        with pytest.raises(BeetleError) as e:
            await core.remote_session(target, "T1", None)
        assert e.value.code == "no_route_meets_floor"

    asyncio.run(go())
    assert core.ledger.entries == {}


def test_a_session_below_the_floor_never_gets_the_adapter(core, monkeypatch):
    """The second gate: the session itself says what it is, before anything is sent."""
    sent = []

    class Faked:
        tier, info, closing, contact = "T3", {"models": []}, False, {"id": "a", "enc_pk": "00"}

        def upload_adapter_files(self, files):
            sent.append(files)

    async def opened(*args, **kw):
        return Faked()

    class Fake:
        async def close(self):
            pass

    async def bind(*args, **options):
        return Fake()

    async def candidates(*args):
        return []

    chain = SimpleNamespace(open_channel=lambda *a: Keypair().pubkey(), mint=None)
    monkeypatch.setattr(remote, "TokenCounter", lambda p=None: TokenCounter())
    monkeypatch.setattr(remote, "seller_of", lambda c, o: SimpleNamespace(net_pk=bytes(32), authority=Keypair().pubkey()))
    monkeypatch.setattr(remote.Session, "open", opened)
    monkeypatch.setattr(remote.Peer, "bind", bind)
    monkeypatch.setattr(wire, "candidates", candidates)
    monkeypatch.setattr(wire, "buyer_session", lambda *a: None)
    core.chain = chain
    core.wallet = Keypair()
    core.echo_nodes = lambda: []
    core.ledger.entries["k"] = {}
    session = remote.RemoteSession(core, SimpleNamespace(address=Keypair().pubkey(), rev=0), BASE)
    session.ledger_key = "k"

    async def go():
        with pytest.raises(BeetleError) as e:
            await session.open(1000, "T3-B")
        assert e.value.code == "no_route_meets_floor"

    asyncio.run(go())
    assert sent == []


def test_t3_counts_as_unprotected():
    assert accepts("public", "T3") and not accepts("T3-B", "T3")


def test_a_chat_with_a_private_floor_is_refused_before_a_route_opens(core):
    core.chain = Refuses()
    core.policy.save(min_tier="T2")

    async def go():
        with pytest.raises(BeetleError) as e:
            await core.open_chat("bolt-e4b", [{"role": "user", "content": "x"}], {}, {"route": "remote"})
        assert e.value.status == 409

    asyncio.run(go())


def test_the_market_shows_what_a_session_is_not_what_it_claims(core):
    offer = SimpleNamespace(kind=KIND_INFERENCE, tier=2, model_root=BASE.root, price_a=1, price_b=2, reach=0,
                            seller=None, hw_class=1, mem_gb=0, slots_total=1, slots_used=0, updated_at=0,
                            address=Keypair().pubkey())
    seller = SimpleNamespace(authority=Keypair().pubkey(), stake_total=0, n_settled=0, n_nostart=0, n_timeout=0,
                             n_rejected=0, n_norelease=0)
    import client.core as module

    module.Seller.decode, keep = staticmethod(lambda a, d: seller), module.Seller.decode
    core.chain = SimpleNamespace(_account=lambda a: b"")
    try:
        row = core._row(offer)
    finally:
        module.Seller.decode = keep
    assert row["tier"] == "T3-C" and row["tier_claimed"] == "T2"


def test_only_well_staked_sellers_get_asked_for_this_ip(core):
    import client.core as module

    rich = SimpleNamespace(reach=REACH_ECHO, public_contact={"id": "a", "addrs": ["203.0.113.1:1"]}, seller="rich")
    poor = SimpleNamespace(reach=REACH_ECHO, public_contact={"id": "b", "addrs": ["203.0.113.2:1"]}, seller="poor")
    stakes = {"rich": module.ECHO_MIN_STAKE, "poor": 1}
    keep = module.Seller.decode
    module.Seller.decode = staticmethod(lambda a, d: SimpleNamespace(stake_total=stakes[a]))
    core.chain = SimpleNamespace(_account=lambda a: b"", list_offers=lambda **kw: [rich, poor])
    try:
        assert [n["id"] for n in core.echo_nodes()] == ["a"]
    finally:
        module.Seller.decode = keep


def test_the_network_comes_from_chain_json_not_the_rpc_node(tmp_path, monkeypatch):
    monkeypatch.setenv("BEETLE_HOME", str(tmp_path))
    wire.write_json(tmp_path / "config.json", {"chain": "chain.json", "rendezvous": {"kind": "file", "dir": "x"}})
    chain = SimpleNamespace(rpc=SimpleNamespace(call=lambda m: "someothergenesis"),
                            chain_ref=chain_ref_for("thegenesisinchainjson"))
    monkeypatch.setattr(wire, "load_chain", lambda cfg, where: chain)
    monkeypatch.setattr(wire, "load_wallet", lambda where: Keypair())
    with pytest.raises(SystemExit):
        Core(tmp_path)


def test_units_say_little_about_the_data():
    assert [paid_units(n) for n in (1, 2, 3, 6, 11, 1999, 2001, 700_000)] == [1, 2, 5, 10, 20, 2000, 5000, 1_000_000]


# --- addresses ------------------------------------------------------------------------


def test_nothing_goes_to_a_stun_server_unless_it_was_configured():
    assert wire.stun_servers({}) == [] and wire.public_ip({}) is None
    assert "not probed" in wire.nat_type({})["nat"]


def fake_peer():
    return SimpleNamespace(port=5000, addrs=["192.168.1.2:5000", "10.0.0.3:5000"])


def test_local_addresses_stay_home_when_a_public_one_exists(monkeypatch):
    monkeypatch.setattr(wire, "public_ip", lambda cfg: "203.0.113.9")
    assert asyncio.run(wire.candidates(fake_peer(), {})) == ["203.0.113.9:5000"]
    assert len(asyncio.run(wire.candidates(fake_peer(), {"lan": True}))) == 3


def test_with_nothing_else_local_addresses_are_all_there_is(monkeypatch):
    monkeypatch.setattr(wire, "public_ip", lambda cfg: None)
    assert asyncio.run(wire.candidates(fake_peer(), {})) == fake_peer().addrs


def test_a_closed_job_is_reached_through_memos(tmp_path):
    from net.rendezvous import ChannelRendezvous, FileRendezvous

    pda = Keypair().pubkey()
    chain = SimpleNamespace(job_pda=lambda job_id: pda)
    made = wire.closed_job_rendezvous({"rendezvous": {"kind": "chain"}}, b"j" * 32, "buyer", chain, Keypair(),
                                      Keypair().pubkey(), net_pk=bytes(32))
    assert isinstance(made, ChannelRendezvous) and made.channel == pda
    filed = wire.closed_job_rendezvous({"rendezvous": {"kind": "file", "dir": str(tmp_path)}}, b"j" * 32, "buyer",
                                       chain, Keypair(), Keypair().pubkey())
    assert isinstance(filed, FileRendezvous)


# --- a job after trouble ---------------------------------------------------------------


class Keeper:
    """What a job reads from the chain."""

    def __init__(self, job=None):
        self.job, self.calls = job, []

    def get_job(self, job_id):
        return self.job

    def chain_time(self):
        return int(time.time())

    def get_config(self):
        return {"rdv_abort_after": 60, "reject_fee_cap": 10_000}

    def claim_due(self, caller, job_id):
        self.calls.append("claim_due")

    def reject_evidence(self, caller, job):
        self.calls.append("reject_evidence")

    def abort_unstarted(self, caller, job):
        self.calls.append("abort_unstarted")


def train_job(tmp_path, chain, **rec) -> TrainJob:
    fields = {"job_id": "ab" * 32, "stage": "training", "timeline": [], "ledger": "k", **rec}
    ledger = SimpleNamespace(release=lambda key: None, update=lambda key, state, usd: fields.update(spent=usd))
    core = SimpleNamespace(chain=chain, wallet=Keypair(), ledger=ledger)
    root = tmp_path / "job"
    root.mkdir(exist_ok=True)
    return TrainJob(core, root, fields)


def no_link(tmp_path, answered: bool) -> TrainJob:
    onchain = SimpleNamespace(state="Opened", rdv_opened=True, rdv_answered=answered, opened_at=0, start_by=10**12,
                              price=1_000_000)
    job = train_job(tmp_path, Keeper(onchain))
    with pytest.raises(JobEnded):
        asyncio.run(job.check_alive(unreachable=True))
    return job


def test_no_link_to_a_seller_that_never_answered_costs_nothing(tmp_path):
    job = no_link(tmp_path, answered=False)
    assert job.rec["ended"] == "no link to the seller: refunded, no fault" and "spent" not in job.rec


def test_leaving_a_seller_that_answered_says_what_it_cost_and_counts_it(tmp_path):
    job = no_link(tmp_path, answered=True)
    assert "$0.01 fee" in job.rec["ended"] and job.rec["spent"] == 0.01


def test_a_closed_account_does_not_end_a_released_job(tmp_path):
    job = train_job(tmp_path, Keeper(None), released=1)
    asyncio.run(job.check_alive())
    assert job.closed and job.rec["stage"] == "training"


def test_a_settled_job_is_recovered_too(tmp_path):
    job = train_job(tmp_path, Keeper(SimpleNamespace(state="Settled")), released=1)
    asyncio.run(job.check_alive())
    assert job.closed


def test_a_seller_that_never_answers_ends_the_search(tmp_path):
    from client.jobs import RECOVER_TRIES

    job = train_job(tmp_path, Keeper(None), released=1, usd=3.0)

    async def go():
        for _ in range(RECOVER_TRIES - 1):
            await job.check_alive(unreachable=True)
        with pytest.raises(JobEnded):
            await job.check_alive(unreachable=True)

    asyncio.run(go())
    assert job.rec["stage"] == "failed" and "gone" in job.rec["ended"]
    assert job.rec["spent"] == 3.0   # unknown counts as spent


def test_a_refund_is_not_mistaken_for_a_result(tmp_path):
    job = train_job(tmp_path, Keeper(SimpleNamespace(state="TimedOut")), released=1)

    async def go():
        with pytest.raises(JobEnded):
            await job.check_alive()

    asyncio.run(go())
    assert job.rec["stage"] == "failed" and not job.rec["retry"]


@pytest.mark.parametrize("state", ["Declined", "Unstarted", "NoStart"])
def test_a_job_that_never_got_going_can_move_on(tmp_path, state):
    job = train_job(tmp_path, Keeper(SimpleNamespace(state=state)))

    async def go():
        with pytest.raises(JobEnded):
            await job.check_alive()

    asyncio.run(go())
    assert job.rec["retry"]


def test_no_evidence_is_refused_while_that_is_cheap(tmp_path):
    now = int(time.time())
    chain = Keeper(SimpleNamespace(state="Started", release_by=now + 1))
    job = train_job(tmp_path, chain)

    async def go():
        with pytest.raises(JobEnded):
            await job._give_up_on_evidence()

    asyncio.run(go())
    assert chain.calls[0] == "reject_evidence" and job.rec["proof"]["headline"].startswith("Evidence failed")


def test_evidence_can_still_be_waited_for_early_on(tmp_path):
    chain = Keeper(SimpleNamespace(state="Started", release_by=int(time.time()) + 300))
    asyncio.run(train_job(tmp_path, chain)._give_up_on_evidence())
    assert chain.calls == []


def test_the_next_seller_is_tried_a_few_times_only(core):
    called = []

    async def train(*args, **kw):
        called.append(kw["tried"])
        return {"job": "next"}

    core.train = train
    rec = {"dataset": "d", "base": "b", "min_tier": "public", "max_usd": 1.0, "idempotency_key": "k",
           "alias": None, "tried": ["o1"], "usd": 0.5}
    job = SimpleNamespace(rec=rec, save=lambda: None)
    asyncio.run(core.try_next_seller(job))
    assert rec["next_job"] == "next" and called == [("o1",)]
    rec.update(tried=["o1", "o2", "o3"], next_job=None)
    asyncio.run(core.try_next_seller(job))
    assert rec["next_job"] is None and len(called) == 1


def test_a_refused_next_seller_is_recorded_not_raised(core):
    async def train(*args, **kw):
        raise BeetleError(409, "no_route_meets_floor", "none")

    core.train = train
    rec = {"dataset": "d", "base": "b", "min_tier": "public", "max_usd": 1.0, "idempotency_key": "k",
           "alias": None, "tried": ["o1"], "usd": 0.5}
    asyncio.run(core.try_next_seller(SimpleNamespace(rec=rec, save=lambda: None)))
    assert rec["next_error"] == "no_route_meets_floor"


def test_a_seller_the_buyer_named_is_never_swapped(core):
    async def train(*args, **kw):
        raise AssertionError("went to another seller")

    core.train = train
    rec = {"tried": ["o1"], "pinned": True, "usd": 0.5}
    asyncio.run(core.try_next_seller(SimpleNamespace(rec=rec, save=lambda: None)))
    assert rec["next_error"] == "seller_pinned" and "next_job" not in rec


def test_the_next_seller_gets_only_the_price_already_approved(core):
    seen = {}

    async def train(*args, **kw):
        seen.update(kw)
        return {"job": "next"}

    core.train = train
    rec = {"dataset": "d", "base": "b", "min_tier": "public", "max_usd": 9.0, "idempotency_key": "k",
           "alias": None, "tried": ["o1"], "usd": 1.2}
    asyncio.run(core.try_next_seller(SimpleNamespace(rec=rec, save=lambda: None)))
    assert seen["within"] == 1.2 and not seen.get("human")


@pytest.mark.parametrize("price, opens", [(1.5, True), (2.5, False)])
def test_a_retry_past_the_approved_price_waits_for_a_person(core, price, opens):
    core.policy.save(per_job_usd=5, per_day_usd=8, min_tier="public", auto_approve_under_usd=1)
    core.quote = lambda *a: {"_offer": None, "base": BASE.name, "offer": "o2", "usd": price}
    core.datasets = SimpleNamespace(get=lambda name: {"sha256": "x"})
    core._repeat_of = lambda *a: None
    opened = []

    async def open_job(*a):
        opened.append(a)
        raise RuntimeError("stop here")

    core._open_job = open_job

    async def go():
        return await core.train("d", BASE.name, "public", 9.0, "k", within=1.5)

    if opens:
        with pytest.raises(RuntimeError, match="stop here"):
            asyncio.run(go())
    else:
        with pytest.raises(BeetleError) as e:
            asyncio.run(go())
        assert e.value.code == "approval_required"
    assert bool(opened) is opens


def test_a_restart_finishes_a_delivered_job_with_a_name(core):
    ran = []

    async def run(name):
        ran.append(name)

    def job(name, stage, alias=None):
        return SimpleNamespace(rec={"stage": stage, "alias": alias}, run=lambda: run(name), task=None)

    core.jobs = {"a": job("fetching", "delivered", "me"), "b": job("nameless", "delivered"),
                 "c": job("training", "training"), "d": job("done", "settled", "me"), "e": job("dead", "failed", "me")}

    async def go():
        core.resume()
        await asyncio.gather(*(j.task for j in core.jobs.values() if j.task))

    asyncio.run(go())
    assert sorted(ran) == ["fetching", "training"]


def test_a_cancelled_job_stays_cancelled_after_a_restart(core):
    ran, refunds = [], []

    async def run():
        ran.append("ran")

    async def refund(job):
        refunds.append(job)

    job = SimpleNamespace(rec={"stage": "uploading", "alias": None, "cancelled": True}, run=run, task=None)
    done = SimpleNamespace(rec={"stage": "failed", "alias": None, "cancelled": True}, run=run, task=None)
    core.jobs = {"a": job, "b": done}
    core._refund_when_due = refund

    async def go():
        core.resume()
        await job.task

    asyncio.run(go())
    assert not ran and refunds == [job] and done.task is None


def test_cancelling_before_the_start_is_written_down(core):
    async def refund(job):
        return None

    job = SimpleNamespace(rec={"stage": "uploading", "job_id": "a" * 64}, id="a" * 16, job_id=b"\xaa" * 32,
                          task=None, save=lambda: None)
    core.jobs = {job.id: job}
    core.chain = SimpleNamespace(get_job=lambda job_id: SimpleNamespace(state="Opened"))
    core._refund_when_due = refund

    async def go():
        await core.cancel(job.id)
        await job.task

    asyncio.run(go())
    assert job.rec["cancelled"] is True and job.rec["ended"].startswith("cancelled")


def test_running_out_of_funds_at_the_end_keeps_the_result(tmp_path):
    job = train_job(tmp_path, Refuses(), stage="delivered", alias="me", uploaded=1, released=1, received=1)

    async def fetch(ident, alias):
        raise BeetleError(402, "no_funds", "this wallet holds $0.10 and this needs $0.50")

    job.core.fetch = fetch
    asyncio.run(job.run())
    assert job.rec["stage"] == "delivered" and job.rec["error"]["code"] == "no_funds"


def test_any_other_failure_still_ends_the_job(tmp_path):
    job = train_job(tmp_path, Refuses(), stage="delivered", alias="me", uploaded=1, released=1, received=1)

    async def fetch(ident, alias):
        raise RuntimeError("broken")

    job.core.fetch = fetch
    asyncio.run(job.run())
    assert job.rec["stage"] == "failed"


@pytest.mark.parametrize("tier, reads", [("T3-C", True), ("T3-B", True), ("MOCK", True), ("T2", False), ("T1", False)])
def test_the_training_note_says_who_can_read_it(tier, reads):
    assert ("can read" in training_note(tier)) is reads
    assert ("ciphertext" in training_note(tier)) is not reads


def test_a_marked_step_leaves_the_stage_alone(tmp_path):
    job = train_job(tmp_path, Refuses())
    job.mark("connected", "direct")
    assert job.rec["stage"] == "training" and job.rec["timeline"][-1]["stage"] == "connected"


def test_inputs_count_as_sent_only_once_the_link_is_up(tmp_path, monkeypatch):
    job = train_job(tmp_path, Refuses())
    (job.root / "in").mkdir()
    (job.root / "in" / "spec.json").write_text("{}")

    async def send(link, msg):
        pass

    async def receive(link):
        return {"t": "need", "names": []}

    async def ask(step):
        job.mark("connected")  # dialling takes a while, and comes first
        return await step(None)

    monkeypatch.setattr(wire, "send_msg", send)
    monkeypatch.setattr(wire, "recv_msg", receive)
    job.ask = ask
    asyncio.run(job._upload())
    assert [t["stage"] for t in job.rec["timeline"]] == ["connected", "uploading", "evidence"]


def ready_to_train(core, repeat=None, usd=1.5):
    """Train as an agent: jobs open for real as far as the approvals can tell."""
    core.policy.save(per_job_usd=5, per_day_usd=20, min_tier="public", auto_approve_under_usd=1)
    core.quote = lambda *a: {"_offer": None, "base": BASE.name, "offer": "o", "usd": usd}
    core.datasets = SimpleNamespace(get=lambda name: {"sha256": "x"})
    core._repeat_of = lambda *a: repeat
    core.view = lambda job: "opened"
    core.opened, core.failing = [], False

    async def run():
        pass

    async def open_job(*a):
        await asyncio.sleep(0.05)  # the chain takes a while, and another call gets this far
        if core.failing:
            raise RuntimeError("stop here")
        core.opened.append(a)
        return SimpleNamespace(run=run)

    core._open_job = open_job

    async def train(key, **kw):
        return await core.train("d", BASE.name, "public", 9.0, key, **kw)

    return train


def ticket_for(train, key="asking"):
    with pytest.raises(BeetleError) as e:
        asyncio.run(train(key))
    assert e.value.code == "approval_required"
    return e.value.extra["approval"]


@pytest.mark.parametrize("repeat, usd", [(None, 1.5), ("earlier", 0.5)])
def test_one_approval_lets_one_job_through(core, repeat, usd):
    train = ready_to_train(core, repeat, usd)
    first = ticket_for(train)
    core.approvals.approve(first)
    assert asyncio.run(train("one")) == "opened"
    assert ticket_for(train, "two") != first
    assert len(core.opened) == 1


def test_two_jobs_at_once_cannot_share_an_approval(core):
    train = ready_to_train(core)
    core.approvals.approve(ticket_for(train))

    async def go():
        return await asyncio.gather(train("one"), train("two"), return_exceptions=True)

    out = asyncio.run(go())
    assert sorted(type(r).__name__ for r in out) == ["BeetleError", "str"]
    assert len(core.opened) == 1


def test_a_job_that_never_opened_keeps_its_approval(core):
    train = ready_to_train(core)
    core.approvals.approve(ticket_for(train))
    core.failing = True
    with pytest.raises(RuntimeError, match="stop here"):
        asyncio.run(train("one"))
    core.failing = False
    assert asyncio.run(train("two")) == "opened"


def test_a_retry_at_the_approved_price_needs_no_second_approval(core):
    train = ready_to_train(core)
    core.approvals.approve(ticket_for(train))
    asyncio.run(train("one"))
    assert asyncio.run(train("one+1", within=1.5)) == "opened"


def test_the_owner_starting_a_job_leaves_an_agent_approval_alone(core):
    train = ready_to_train(core)
    core.approvals.approve(ticket_for(train))
    asyncio.run(train("owner", human=True))
    assert asyncio.run(train("agent")) == "opened"


def test_prompt_counts_include_the_role_markers():
    counter = TokenCounter()
    assert counter.prompt([{"role": "user", "content": "abcd" * 10}]) == 10 + MESSAGE_OVERHEAD


def test_a_sandbox_is_held_to_what_it_claimed_at_release(tmp_path):
    job = train_job(tmp_path, Keeper(), verdict={"type": "NONE", "measurement": "ab" * 32})
    assert job._expected_measurement() == "ab" * 32


def test_evidence_with_a_measurement_is_held_to_the_public_code(tmp_path):
    import beetle_crypto as bc
    from job_defaults import sandbox_code_files

    job = train_job(tmp_path, Keeper(), verdict={"type": "TDX", "measurement": "ab" * 32})
    assert job._expected_measurement() == bc.code_measurement(sandbox_code_files())


def settle_case(core, accept):
    core.chain = SimpleNamespace(accept=accept, get_job=lambda job_id: None)
    core.wallet = Keypair()
    delivered = SimpleNamespace(state="Delivered", job_id=b"j", expected_dues=lambda accepted=False: (900_000, 100_000))
    return asyncio.run(core._settle({"usd": 1.0}, {"passed": True, "work_bps": 10_000}, delivered))


def test_accepting_pays_what_the_chain_says(core):
    assert settle_case(core, lambda wallet, job: "sig") == 0.9


def test_losing_the_race_with_finalize_is_not_an_error(core):
    def too_late(wallet, job):
        raise RuntimeError("AccountNotInitialized")

    assert settle_case(core, too_late) == pytest.approx(0.2)   # the floor, off T2


# --- a channel's money comes back whatever happened ---------------------------------------


def channel_case(core, status: str, *, closed_at: float | None = 0.0, closing_for: int = 0, settled: int = 0):
    calls: list[str] = []
    ch = None if status == "gone" else SimpleNamespace(status=status, settled=settled, closure_started_at=1000,
                                                       grace_period=100)
    core.chain = SimpleNamespace(get_channel=lambda a: ch, chain_time=lambda: 1000 + closing_for,
                                 request_close=lambda w, c: calls.append("request_close"),
                                 seal_channel=lambda w, c: calls.append("seal_channel"),
                                 distribute=lambda w, c: calls.append("distribute"))
    core.wallet = Keypair()
    entry = {"kind": "infer", "state": "escrowed", "usd": 0.5, "at": time.time(), "used": 0.1, "signed": 0.11,
             "channel": str(Keypair().pubkey())}
    if closed_at is not None:
        entry["closed_at"] = closed_at
    core.ledger.entries["c"] = entry
    core.sweep_channels()
    return calls, entry


def test_a_seller_gets_a_moment_to_settle_a_closed_session(core):
    calls, entry = channel_case(core, "Open", closed_at=time.time())
    assert calls == [] and entry["state"] == "escrowed"


def test_a_channel_nobody_settled_is_closed_by_the_buyer(core):
    assert channel_case(core, "Open", closed_at=None)[0] == ["request_close"]


def test_the_grace_period_is_waited_out(core):
    assert channel_case(core, "Closing", closing_for=50)[0] == []
    assert channel_case(core, "Closing", closing_for=500)[0] == ["seal_channel"]
    assert channel_case(core, "Sealed")[0] == ["distribute"]


def test_the_ledger_settles_at_what_the_chain_paid(core):
    _, entry = channel_case(core, "Distributed", settled=250_000)
    assert entry["state"] == "settled" and entry["usd"] == 0.25
    _, entry = channel_case(core, "gone")  # no account left to ask: the last voucher is the most it took
    assert entry["state"] == "settled" and entry["usd"] == 0.11


# --- a local server, a named offer, a refused delivery ------------------------------------


def test_a_local_server_decides_what_runs_here():
    from client import doctor

    caps = {"local_engine": "server", "local_models": ["qwen3.5-2b"], "budget_gb": 0, "on_power": True, "disk_gb": 1}
    assert doctor.can_serve(caps, BASE) and doctor.path_for(caps, BASE)["path"] == "hybrid"
    assert doctor.path_for(caps, mp.get("bolt-e4b"))["path"] == "remote"


def test_a_core_with_a_local_server_uses_it(tmp_path, monkeypatch):
    from client.local import ServerEngine

    monkeypatch.setenv("BEETLE_HOME", str(tmp_path))
    wire.write_json(tmp_path / "config.json", {"chain": None, "rendezvous": {"kind": "file", "dir": "x"},
                                               "local_backend": {"url": "http://127.0.0.1:1", "models": {"a": "b"}}})
    assert isinstance(Core(tmp_path).engine, ServerEngine)


def test_a_server_adapter_stays_off_a_disk(monkeypatch, tmp_path):
    import client.local as local

    monkeypatch.setattr(local, "is_tmpfs", lambda path: False)
    engine = local.ServerEngine({"url": "http://127.0.0.1:1", "models": {}, "scratch": str(tmp_path)})

    async def go():
        with pytest.raises(RuntimeError, match="tmpfs"):
            await engine._served(BASE, "job", lambda: {})

    asyncio.run(go())
    assert list(tmp_path.iterdir()) == []


def test_a_named_offer_says_what_is_wrong_with_it(core):
    inference = SimpleNamespace(kind=KIND_INFERENCE, model_root=BASE.root, tier=4)
    core.chain = SimpleNamespace(get_offer=lambda a: inference)
    error = core._why_not(str(Keypair().pubkey()), BASE, "public")
    assert error.status == 409 and error.code == "offer_not_training" and "inference" in error.message


def test_a_proven_result_cannot_be_refused(core):
    core.chain = SimpleNamespace(get_job=lambda j: SimpleNamespace(state="Delivered", tier=2))
    core.jobs["ab"] = SimpleNamespace(job_id=b"j", rec={})

    async def go():
        with pytest.raises(BeetleError) as e:
            await core.reject("ab")
        assert "T2" in e.value.message

    asyncio.run(go())


def test_only_a_delivery_can_be_refused(core):
    core.chain = SimpleNamespace(get_job=lambda j: None)
    core.jobs["ab"] = SimpleNamespace(job_id=b"j", rec={})

    async def go():
        with pytest.raises(BeetleError) as e:
            await core.reject("ab")
        assert e.value.status == 400

    asyncio.run(go())


def test_a_dead_sellers_offer_falls_through_to_the_next(core, monkeypatch):
    tried = []

    async def start(target, floor, max_usd, offer):
        tried.append(offer.price_a)
        if offer.price_a == 1:
            raise BeetleError(503, "no_seller", "the seller did not answer")
        return SimpleNamespace(pipe_alive=lambda: True, price=offer.price_a)

    offers = [SimpleNamespace(price_a=p, price_b=0, address=p) for p in (2, 1)]
    core.chain = object()
    monkeypatch.setattr(core, "_matching", lambda *a: offers)
    monkeypatch.setattr(core, "_start_session", start)

    async def go():
        return await core.remote_session(Target("x", BASE), "public", None)

    assert asyncio.run(go()).price == 2 and tried == [1, 2]


def test_a_named_seller_is_the_only_one_tried(core, monkeypatch):
    offers = [SimpleNamespace(price_a=1, price_b=0, address="a"), SimpleNamespace(price_a=2, price_b=0, address="b")]
    tried = []

    async def start(target, floor, max_usd, offer):
        tried.append(offer.address)
        return SimpleNamespace(pipe_alive=lambda: True)

    core.chain = object()
    monkeypatch.setattr(core, "_matching", lambda *a: offers)
    monkeypatch.setattr(core, "_start_session", start)
    asyncio.run(core.remote_session(Target("x", BASE), "public", None, "b"))
    assert tried == ["b"]
