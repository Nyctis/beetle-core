"""A small market on a local chain: many sellers and buyers, each its own process
with its own wallet, starting at random times and quitting at random moments.

    scripts/setup_chain.sh local
    uv run python scripts/sim_market.py --sellers 3 --buyers 8 --mint-authority .keys/deployer.json

What is real: the chain, escrow, offers, stakes, slashing, payment channels,
the P2P links, encryption, the daemon and client code. What is simulated: the
sandbox trains a tiny numpy model in place of a LoRA (DemoSandbox from
scripts/market_demo.py) and the inference backend echoes. So this tests how
money and state behave when parties disappear, not machine learning.

A buyer can quit before the key release, or after delivery. A seller can quit
before starting a job, or during one. A keeper, who could be anyone, then ends
what's stuck. Every job's result goes into --out, and the run fails if money
isn't conserved or a job stays open.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import client  # noqa: E402,F401  (import paths)
import client.remote  # noqa: E402

import beetle_crypto as bc  # noqa: E402
import model_profiles as mp  # noqa: E402
from beetle_chain import (FINISHED, KIND_INFERENCE, KIND_TRAINING, Chain, OfferTerms,  # noqa: E402
                          load_keypair, net_keypair)
from client.core import Core  # noqa: E402
from client.datasets import Datasets  # noqa: E402
from client.errors import BeetleError  # noqa: E402
from client.policy import Policy  # noqa: E402
from job_defaults import TIERS, sandbox_code_files  # noqa: E402
from market_demo import DemoSandbox  # noqa: E402
from node import environment, wire  # noqa: E402
from node.daemon import Node  # noqa: E402
from serve.gateway import Backend  # noqa: E402
from solders.pubkey import Pubkey  # noqa: E402

USDC = 1_000_000
MODEL = "smollm2-135m"
STAKE = 30 * USDC
FUNDS = 60 * USDC
PRICE_PER_MTOK = 40 * USDC
DRAIN_TIMEOUT = 480  # a killed seller's job only ends at its deadline, which has a floor of minutes


class SimRuntime:
    """The sandbox's file protocol, without the sandbox."""

    def __init__(self, chain: Chain, work_s: float):
        self.chain, self.work_s = chain, work_s
        self.running: dict[str, threading.Thread] = {}
        self.measurement = bc.code_measurement(sandbox_code_files())

    def prepare(self) -> None:
        pass

    def launch(self, name: str, model: Path, inbox: Path, outbox: Path) -> None:
        job = json.loads((inbox / "job.json").read_text())
        spec = json.loads((inbox / "spec.json").read_text())
        sandbox = DemoSandbox(inbox.parent, job, spec, self.measurement)
        evidence = sandbox.evidence()
        (outbox / "attestation.json").write_text(json.dumps(evidence["attestation"]))
        (outbox / "attestation.sig").write_text(evidence["signature"])
        self.running[name] = threading.Thread(target=self._train, args=(sandbox, inbox, outbox))
        self.running[name].start()

    def _train(self, sandbox: DemoSandbox, inbox: Path, outbox: Path) -> None:
        sealed = inbox / "sealed_dek.bin"
        deadline = time.time() + 60
        while not sealed.exists():
            if time.time() > deadline:
                return
            time.sleep(0.2)
        time.sleep(self.work_s)
        report, report_sig, settle_sig, adapter_ct, private_ct = sandbox.train(self.chain, sealed.read_bytes())
        (outbox / "adapter.enc").write_bytes(adapter_ct)
        (outbox / "private_report.enc").write_bytes(private_ct)
        (outbox / "report.json").write_text(json.dumps(report))
        (outbox / "report.sig").write_text(report_sig)
        (outbox / "settle.sig").write_text(settle_sig.hex())

    def attested(self, name: str, outbox: Path, timeout: float) -> dict | None:
        return json.loads((outbox / "attestation.json").read_text())

    def wait(self, name: str, timeout: float) -> bool:
        self.running[name].join(timeout)
        return not self.running[name].is_alive()

    def stop(self, name: str) -> str:
        return ""


class EchoBackend(Backend):
    """An inference server that repeats the question."""

    def __init__(self):
        pass

    async def models(self) -> list[str]:
        return [MODEL]

    async def chat(self, body: dict):
        words = body["messages"][-1]["content"].split()
        for word in words:
            yield {"choices": [{"delta": {"content": word + " "}}]}
        yield {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": len(words)}}


class SimNode(Node):
    def _model_for(self, spec: dict) -> Path:
        return Path(".")

    def _backend(self) -> Backend:
        return EchoBackend()

    def environment_cmd(self) -> list[str]:
        return [sys.executable, __file__, "--role", "environment"]


class SimCore(Core):
    async def _check_effect(self, job, files: dict) -> dict:
        return {"where": "sim", "asked": 0, "changed": 0}

    def _offers(self, kind: int):
        """Only this run's sellers: the chain still lists the ones earlier runs left behind."""
        ours = os.environ["SIM_SELLERS"].split(",")
        return [o for o in super()._offers(kind) if str(o.seller) in ours]


def setup_party(chain: Chain, mint_authority, home: Path) -> Path:
    home.mkdir(parents=True)
    kp = wire.new_wallet(home)
    wire.write_json(home / "chain.json", chain.cfg)
    wire.write_json(home / "config.json", {"chain": "chain.json", "traffic": "burst", "lan": True,
                                           "rendezvous": {"kind": "chain"},
                                           "host": {"gpu": False, "models": [MODEL], "backend": "sim", "port": 0,
                                                    "min_grace_s": 1, "retain_s": 30, "accept_wait_s": 20,
                                                    "scratch": str(home / "scratch"),
                                                    "allow_disk_scratch": True}})
    chain.airdrop(kp.pubkey())
    chain.create_ata(kp, kp.pubkey())
    chain.mint_test_usdc(mint_authority, kp.pubkey(), FUNDS)
    return home


def post_offers(chain: Chain, home: Path, rng: random.Random) -> None:
    kp = wire.load_wallet(home)
    chain.register_seller(kp, net_pk=net_keypair(wire.secret(home, "net.key")).public_bytes)
    chain.stake(kp, STAKE)
    root, measurement = mp.get(MODEL).root, bytes.fromhex(bc.code_measurement(sandbox_code_files()))
    common = dict(tier=TIERS["t3c"], measurement=measurement, model_root=root,
                  valid_until=chain.chain_time() + 86_400, slots_total=2)
    chain.post_offer(kp, kp.pubkey(), 0, OfferTerms(KIND_TRAINING, price_a=PRICE_PER_MTOK + rng.randrange(USDC),
                                                    start_window_s=20, grace_s=150, tokens_per_s=100, **common))
    chain.post_offer(kp, kp.pubkey(), 1, OfferTerms(KIND_INFERENCE, price_a=USDC, price_b=3 * USDC, **common))


async def run_seller() -> None:
    where = wire.home()
    cfg = wire.load_config(where)
    chain, kp = wire.load_chain(cfg, where), wire.load_wallet(where)
    await SimNode(where, cfg, chain, kp, SimRuntime(chain, float(os.environ.get("SIM_WORK_S", 3)))).run()


async def run_buyer() -> None:
    """Exit codes say what became of it, for the record."""
    quit_at = os.environ.get("SIM_QUIT", "")
    core = SimCore()
    await asyncio.sleep(float(os.environ.get("SIM_DELAY", 0)))
    view = await until_a_seller_is_free(core)
    job = core.job(view["job"])
    print(json.dumps({"job": job.rec["job_id"]}), flush=True)
    while job.rec["stage"] not in ("delivered", "failed"):
        if job.rec["stage"] == quit_at:
            os._exit(9)
        await asyncio.sleep(0.2)
    if job.rec["stage"] == "failed":
        os._exit(3)
    if quit_at == "delivered":
        os._exit(9)
    view = await core.fetch(job.id, "me")
    print(json.dumps({"settled": view["result"]}), flush=True)
    if os.environ.get("SIM_CHAT"):
        await chat_once(core)
    os._exit(0)


async def until_a_seller_is_free(core: SimCore) -> dict:
    key = os.urandom(8).hex()
    for _ in range(30):
        try:
            return await core.train("sim", MODEL, "public", 5.0, key, human=True)
        except BeetleError as e:
            if e.code != "no_seller":
                raise
            await asyncio.sleep(3)
    os._exit(4)


async def chat_once(core: SimCore) -> None:
    """A seller can vanish here too. The deposit then comes back the slow way."""
    client.remote.GRACE_S = 2
    try:
        route = await core.open_chat(MODEL, [{"role": "user", "content": "one two three"}], {}, {"route": "remote"})
        text = "".join([p async for p in route.stream()])
        print(json.dumps({"chat": text.strip(), "cost_usd": route.cost_usd()}), flush=True)
        await core.close_idle(0)
    except BeetleError as e:
        print(json.dumps({"chat_error": e.code}), flush=True)
    for _ in range(60):
        await asyncio.to_thread(core.sweep_channels)
        if not any(e["state"] == "escrowed" and e["kind"] == "infer" for e in core.ledger.entries.values()):
            return
        await asyncio.sleep(1)


def keep(chain: Chain, keeper, sellers: list[Pubkey]) -> tuple[int, list]:
    """Anyone can end what is stuck. Returns how many jobs remain open."""
    now, left = chain.chain_time(), []
    for seller in sellers:
        for job in chain.jobs_for_host(seller):
            left.append(job)
            try:
                if job.state == "Opened" and now > job.start_by:
                    chain.timeout_start(keeper, job)
                elif job.state == "Started" and now > job.release_by:
                    chain.claim_no_release(keeper, job)
                elif job.state == "Running" and now > job.deadline:
                    chain.timeout_run(keeper, job)
                elif job.state == "Delivered" and now >= job.challenge_end:
                    chain.finalize(keeper, job)
                elif job.state in FINISHED:
                    chain.claim_due(keeper, job.job_id)
            except RuntimeError:
                pass
    return len(left), left


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", default=str(ROOT / "chain.json"))
    ap.add_argument("--mint-authority", default=str(ROOT / ".keys/deployer.json"))
    ap.add_argument("--sellers", type=int, default=3)
    ap.add_argument("--buyers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quit-rate", type=float, default=0.4, help="share of parties that quit midway")
    ap.add_argument("--out", default="sim_results.json")
    ap.add_argument("--dir", help="working directory; default: a new temporary one")
    ap.add_argument("--role", choices=["seller", "buyer", "environment"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.role:
        asyncio.run({"seller": run_seller, "buyer": run_buyer,
                     "environment": lambda: environment.serve(EchoBackend())}[args.role]())
        return 0

    rng = random.Random(args.seed)
    chain = Chain.from_file(Path(args.chain))
    assert chain.cfg["network"] == "local", "this waits out short windows: local chain only"
    bc.configure(bytes(chain.program_id), chain.chain_ref.hex().encode())
    mint = load_keypair(Path(args.mint_authority))
    work = Path(args.dir or tempfile.mkdtemp(prefix="beetle-sim-"))
    procs: dict[str, dict] = {}
    balances0: dict[str, int] = {}  # taken before a party can do anything

    def start(name: str, role: str, home: Path, **env) -> None:
        balances0[name] = chain.token_amount(wire.load_wallet(home).pubkey())
        log = open(home / "process.log", "ab")
        p = subprocess.Popen([sys.executable, __file__, "--role", role],
                             stdout=subprocess.PIPE if role == "buyer" else log, stderr=log,
                             env={**os.environ, "BEETLE_HOME": str(home), **env}, text=True)
        procs[name] = {"process": p, "role": role, "home": home, "plan": env.get("SIM_QUIT") or "stay", "lines": []}
        if role == "buyer":
            threading.Thread(target=lambda: procs[name]["lines"].extend(p.stdout), daemon=True).start()

    sellers = []
    for i in range(args.sellers):
        home = setup_party(chain, mint, work / f"seller{i}")
        post_offers(chain, home, rng)
        sellers.append(wire.load_wallet(home).pubkey())
        plan = "quit" if rng.random() < args.quit_rate else ""
        start(f"seller{i}", "seller", home, SIM_WORK_S=f"{rng.uniform(2, 5):.1f}", SIM_PLAN=plan)
        procs[f"seller{i}"]["plan"] = "quit" if plan else "stay"
        print(f"seller{i} {sellers[-1]} {'will quit' if plan else 'stays'}", flush=True)
    ours = ",".join(str(chain.seller_pda(s)) for s in sellers)
    keeper = setup_party(chain, mint, work / "keeper")
    keeper_kp = wire.load_wallet(keeper)

    plans = ("uploading", "evidence", "training", "delivered")
    data = ROOT / "demo/memories.json"
    for i in range(args.buyers):
        home = setup_party(chain, mint, work / f"buyer{i}")
        shutil.copy(data, home / "data.json")
        Datasets(home / "datasets.json").add(home / "data.json", "sim", "synthetic")
        Policy(home / "policy.json").save(per_job_usd=5.0, per_day_usd=50.0, min_tier="public",
                                          auto_approve_under_usd=5.0)
        plan = rng.choice(plans) if rng.random() < args.quit_rate else ""
        start(f"buyer{i}", "buyer", home, SIM_QUIT=plan, SIM_DELAY=f"{rng.uniform(0, 8):.1f}",
              SIM_CHAT="1" if not plan else "", SIM_SELLERS=ours)
        print(f"buyer{i} {'quits at ' + plan if plan else 'stays'}", flush=True)

    start_time = time.time()
    # A seller that plans to quit does so once it holds a job, at a random moment.
    quitters = [n for n, p in procs.items() if p["role"] == "seller" and p["plan"] == "quit"]
    quit_at = {n: time.time() + rng.uniform(5, 25) for n in quitters}
    while time.time() - start_time < DRAIN_TIMEOUT:
        for name in list(quit_at):
            if time.time() > quit_at[name]:
                procs[name]["process"].send_signal(signal.SIGKILL)
                procs[name]["plan"] = f"killed at {time.time() - start_time:.0f}s"
                del quit_at[name]
                print(f"{name} killed", flush=True)
        n_open, _ = keep(chain, keeper_kp, sellers)
        buyers_done = all(p["process"].poll() is not None for p in procs.values() if p["role"] == "buyer")
        channels = sum(c.status != "Distributed" for s in sellers for c in chain.channels_for(s))
        if buyers_done and n_open == 0 and channels == 0 and not quit_at:
            break
        time.sleep(2)

    n_open, stuck = keep(chain, keeper_kp, sellers)
    for p in procs.values():
        if p["process"].poll() is None:
            p["process"].terminate()
    time.sleep(1)

    balances1 = {n: chain.token_amount(wire.load_wallet(p["home"]).pubkey()) for n, p in procs.items()}
    stakes = {str(s): chain.get_seller(s).stake_total for s in sellers}
    rows = []
    for name, p in procs.items():
        lines = [json.loads(line) for line in p["lines"] if line.startswith("{")]
        rows.append({"party": name, "role": p["role"], "plan": p["plan"], "exit": p["process"].poll(),
                     "delta_usdc": (balances1[name] - balances0[name]) / USDC,
                     **{k: v for line in lines for k, v in line.items()}})
    moved = sum(balances1[n] - balances0[n] for n in procs)
    staked = sum(stakes.values()) - STAKE * args.sellers
    result = {"seconds": round(time.time() - start_time), "open_jobs_left": n_open, "parties": rows,
              "wallets_delta_usdc": moved / USDC, "stake_delta_usdc": staked / USDC,
              "stuck": [{"job": j.job_id.hex()[:16], "state": j.state} for j in stuck]}
    Path(args.out).write_text(json.dumps(result, indent=1))
    for r in rows:
        print(r)
    print(f"money: wallets {moved / USDC:+.4f}, stakes {staked / USDC:+.4f} USDC; open jobs left {n_open}")
    return 0 if n_open == 0 and moved + staked == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
