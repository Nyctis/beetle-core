"""What the daemon does: market, quotes, jobs, the model library and routing.

The checks before a job runs, in order. Any failure stops there, with no key out:

    1 data     the dataset is registered; nothing else can be named
    2 floor    strictest(policy, the data's label, the request)
    3 budget   the amount is reserved, or the request stops
    4 quote    an offer that meets the floor, at or under the cap
    5 escrow   open_job, with that price as the most the chain may take
    6 release  on this machine: the evidence binds this job, these keys, a
               measurement on the allowlist, and a tier the floor accepts

Routing can be wrong and it costs a trip, not a secret. The floor is step 6.
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from solders.pubkey import Pubkey

import model_profiles as mp
from beetle_chain import (BPS, KIND_INFERENCE, KIND_TRAINING, MAINNET_GENESIS, REACH_ECHO, TIER_T2, Offer,
                          Seller, chain_ref_for)
from client import doctor as doc
from client.datasets import Datasets, paid_units, train_tokens
from client.errors import (BeetleError, approval_required, bad_request, local_only, no_funds, no_route, no_seller,
                           not_found)
from client.jobs import JobEnded, TrainJob, profile_of
from client.local import MlxEngine, ServerEngine
from client.policy import (ACCEPT, CHAIN_TIER, LABEL_FLOOR, SESSION_TIER, USDC, Approvals, Ledger, Policy,
                           accepts, check_tier, strictest)
from client.proof import card, local_card
from client.remote import RemoteSession
from client.verify import PARAMS, compare, questions, require_effect
from job_defaults import COMPUTE_FLOOR_BPS, RUN_RESERVE_S
from node import wire
from requester import Requester, load_or_create_storage_key
from server import dev_policy

METHODS = ("memory",)
OFFER_FRESH_S = 1800  # three missed heartbeats
ECHO_SAMPLE = 3
ECHO_MIN_STAKE = 100 * USDC  # asking a node shows it your IP; this is what that takes to farm
MAX_SELLERS = 3              # a job that can't get going tries this many
SESSION_IDLE_S = 300.0
SELLER_SETTLES_S = 60.0  # how long a closed session gets before the buyer closes its channel
DAY = 86_400
THINK_HOLD = 4000  # characters; past this it wasn't going to end
MIN_FEE_LAMPORTS = 10_000_000  # fees and the rent on a job's accounts


@dataclass
class Target:
    """What a model name in a request stands for."""

    name: str
    profile: mp.Profile
    adapter: str | None = None       # the job that trained it
    adapter_sha256: str | None = None
    label: str | None = None
    dataset: str | None = None


class ModelStore:
    """Aliases like `me`. Each alias keeps its versions, newest first, so a bad
    version is one rollback away. An adapter inherits its data's label."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _all(self) -> dict:
        return wire.read_json(self.path, {})

    def add(self, alias: str, version: dict) -> None:
        models = self._all()
        models.setdefault(alias, {"versions": [], "current": 0})
        models[alias]["versions"].insert(0, version)
        models[alias]["current"] = 0
        wire.write_json(self.path, models)

    def current(self, alias: str) -> dict | None:
        entry = self._all().get(alias)
        return entry["versions"][entry["current"]] if entry else None

    def rollback(self, alias: str) -> dict:
        models = self._all()
        entry = models.get(alias)
        if entry is None:
            raise not_found(f"no model {alias!r}")
        if entry["current"] + 1 >= len(entry["versions"]):
            raise bad_request(f"{alias} has no earlier version")
        entry["current"] += 1
        wire.write_json(self.path, models)
        return entry["versions"][entry["current"]]

    def listing(self) -> dict:
        return self._all()


class Core:
    def __init__(self, where: Path | None = None):
        self.home = where or wire.home()
        self.cfg = wire.load_config(self.home)
        self.chain = wire.load_chain(self.cfg, self.home) if self.cfg.get("chain") else None
        self.wallet = wire.load_wallet(self.home) if self.chain else None
        self.mainnet = False
        if self.chain:  # decided by the pinned chain_ref; the RPC node only has to agree with it
            if chain_ref_for(self.chain.rpc.call("getGenesisHash")) != self.chain.chain_ref:
                raise SystemExit("this RPC node is not the network chain.json names")
            self.mainnet = self.chain.chain_ref == chain_ref_for(MAINNET_GENESIS)
        self.storage_key = load_or_create_storage_key(self.home / "storage.key")
        self.policy = Policy(self.home / "policy.json")
        self.ledger = Ledger(self.home / "ledger.json")
        self.approvals = Approvals(self.home / "approvals.json")
        self.datasets = Datasets(self.home / "datasets.json")
        self.models = ModelStore(self.home / "models.json")
        local = self.cfg.get("local_backend")
        self.engine = ServerEngine(local) if local else MlxEngine()
        self.sessions: dict[tuple, RemoteSession] = {}
        self.opening: dict[tuple, asyncio.Lock] = {}
        self.starting: set[RemoteSession] = set()
        self.last_connection: dict = {}
        self.jobs: dict[str, TrainJob] = {}
        for record in sorted((self.home / "jobs").glob("*/record.json")):
            rec = wire.read_json(record)
            self.jobs[rec["job_id"][:16]] = TrainJob(self, record.parent, rec)

    def resume(self) -> None:
        """Jobs a restart interrupted carry on from what is on disk and on chain. A
        delivered job with a name was being fetched; one without waits for `beetle fetch`.
        A cancelled job only collects its refund: running it would release the key."""
        for job in self.jobs.values():
            stage = job.rec["stage"]
            if stage in ("settled", "failed"):
                continue
            if job.rec.get("cancelled"):
                job.task = asyncio.create_task(self._refund_when_due(job))
            elif stage != "delivered" or job.rec.get("alias"):
                job.task = asyncio.create_task(job.run())

    def evidence_policy(self) -> dict:
        """What evidence this buyer believes. Empty on mainnet until it says, and T3-C
        needs none: nothing there is believed, which is what its floor accepts."""
        path = self.home / "evidence_policy.json"
        if path.exists():
            return wire.read_json(path)
        return {} if self.mainnet else dev_policy(None)  # test networks: MOCK of the public code

    def require_chain(self):
        if self.chain is None:
            raise local_only("this Mac is set up for local use, so there is no wallet or market")
        return self.chain

    # --- market ---------------------------------------------------------------------

    def _offers(self, kind: int) -> list[Offer]:
        return self.require_chain().list_offers(kind=kind, fresh_within=OFFER_FRESH_S)

    def echo_nodes(self) -> list[dict]:
        """Sellers that offer to say how they see a caller. One alone would be a
        central STUN server, so a few, picked at random."""
        if self.chain is None:
            return []
        offers = [o for k in (KIND_TRAINING, KIND_INFERENCE) for o in self._offers(k)
                  if o.reach & REACH_ECHO and o.public_contact]
        staked = [o for o in offers
                  if Seller.decode(o.seller, self.chain._account(o.seller)).stake_total >= ECHO_MIN_STAKE]
        contacts = list({o.public_contact["id"]: o.public_contact for o in staked}.values())
        return random.sample(contacts, min(len(contacts), ECHO_SAMPLE))

    def _row(self, o: Offer) -> dict:
        p = mp.by_root(o.model_root)
        seller = Seller.decode(o.seller, self.chain._account(o.seller))
        kind = "train" if o.kind == KIND_TRAINING else "infer"
        tier = CHAIN_TIER[o.tier] if kind == "train" else SESSION_TIER
        return {"offer": str(o.address), "kind": kind, "model": p.name if p else o.model_root.hex()[:16],
                "tier": tier, "tier_claimed": CHAIN_TIER[o.tier],
                "tier_source": "self-reported; evidence decides before any key is released"
                if kind == "train" else "no evidence yet: every session counts as T3-C",
                "evidence_needed": tier != "T3-C", "on_their_machine": card(tier)["on_their_machine"],
                "usd_per_mtok": {"train": o.price_a / USDC} if kind == "train"
                else {"in": o.price_a / USDC, "out": o.price_b / USDC},
                "seller": str(seller.authority), "stake_usd": seller.stake_total / USDC,
                "settled": seller.n_settled,
                "failed": seller.n_nostart + seller.n_timeout + seller.n_rejected,
                "buyer_no_release": seller.n_norelease,
                "gpu": wire.gpu_name(o.hw_class), "gpu_gb": o.mem_gb, "public_udp": bool(o.reach & 1),
                "slots_free": o.slots_total - o.slots_used, "updated_at": o.updated_at}

    def market(self, kind: str | None = None, model: str | None = None,
               min_tier: str | None = None) -> list[dict]:
        kinds = {"train": [KIND_TRAINING], "infer": [KIND_INFERENCE],
                 None: [KIND_TRAINING, KIND_INFERENCE]}
        if kind not in kinds:
            raise bad_request(f"kind {kind!r}: only train or infer")
        rows = [self._row(o) for k in kinds[kind] for o in self._offers(k)]
        rows = [r for r in rows if model is None or r["model"] == model]
        if min_tier:
            rows = [r for r in rows if accepts(check_tier(min_tier), r["tier"])]
        return sorted(rows, key=lambda r: min(r["usd_per_mtok"].values()))

    def _matching(self, kind: int, profile: mp.Profile, floor: str) -> list[Offer]:
        """A training offer's tier gets proved before the key goes out. An inference
        offer's never does, so what it claims counts for nothing."""
        tier = (lambda o: CHAIN_TIER[o.tier]) if kind == KIND_TRAINING else (lambda o: SESSION_TIER)
        return [o for o in self._offers(kind) if o.model_root == profile.root and accepts(floor, tier(o))]

    def floor_for(self, label: str | None, request: str | None) -> str:
        policy = self.policy.ready()
        return strictest(policy["min_tier"], LABEL_FLOOR.get(label), check_tier(request))

    def quote(self, dataset: str, base: str, method: str = "memory", min_tier: str | None = None,
              max_usd: float | None = None, seller: str | None = None, exclude: tuple = ()) -> dict:
        if method not in METHODS:
            raise bad_request(f"method {method!r}: only {METHODS} so far")
        entry = self.datasets.get(dataset)
        memories, _ = self.datasets.load(dataset)
        profile = self.pick_base(base)
        floor = self.floor_for(entry["label"], min_tier)
        if not ACCEPT[floor]:
            raise no_route(f"floor {floor} allows no remote seller, and training on this device "
                           "isn't available yet")
        units = paid_units(train_tokens(memories, profile.train["epochs"]))
        offers = [o for o in self._matching(KIND_TRAINING, profile, floor) if str(o.address) not in exclude]
        online = len(offers)
        if seller:
            offers = [o for o in offers if str(o.address) == seller]
            if not offers:
                raise self._why_not(seller, profile, floor)
        if not offers:
            others = sorted({r["tier"] for r in self.market("train", profile.name)})
            if not others:
                raise no_seller(f"no one is selling training on {profile.name} right now")
            raise no_route(f"no seller of {profile.name} meets floor {floor}", floor=floor, other_tiers=others)
        offer = min(offers, key=lambda o: o.quote(units)[0])
        price, run_s = offer.quote(units)
        return {"offer": str(offer.address), "base": profile.name, "floor": floor, "units": units,
                "usd": price / USDC, "run_budget_s": run_s, "tier": CHAIN_TIER[offer.tier],
                "sellers_online": online, "dataset_label": entry["label"], "_offer": offer}

    def _why_not(self, seller: str, profile: mp.Profile, floor: str) -> BeetleError:
        """A buyer who named an offer wants to know what is wrong with that offer."""
        offer = self.chain.get_offer(Pubkey.from_string(seller))
        if offer is None:
            return not_found(f"there is no offer {seller}")
        if offer.kind != KIND_TRAINING:
            return BeetleError(409, "offer_not_training", f"offer {seller} sells inference, not training")
        if offer.model_root != profile.root:
            return BeetleError(409, "offer_wrong_model", f"offer {seller} does not train {profile.name}")
        if not accepts(floor, CHAIN_TIER[offer.tier]):
            return no_route(f"offer {seller} is {CHAIN_TIER[offer.tier]}, below the floor {floor}", floor=floor)
        return no_seller(f"offer {seller} is paused, full or gone quiet")

    def pick_base(self, base: str) -> mp.Profile:
        """`auto`: the largest model this machine can serve, else the smallest there is."""
        if base != "auto":
            try:
                return mp.get(base)
            except KeyError as e:
                raise bad_request(str(e.args[0]))
        caps = doc.machine(self.home, self.cfg)
        profiles = sorted(mp.load().values(), key=lambda p: (p.mlx or {}).get("infer_gb", 0))
        servable = [p for p in profiles if doc.can_serve(caps, p)]
        return servable[-1] if servable else profiles[0]

    # --- training -------------------------------------------------------------------

    def _rate_limit(self, policy: dict) -> None:
        running = sum(j.rec["stage"] not in ("delivered", "settled", "failed") for j in self.jobs.values())
        hour = sum(j.rec["created"] > time.time() - 3600 for j in self.jobs.values())
        if running >= policy["max_concurrent"] or hour >= policy["max_per_hour"]:
            raise BeetleError(429, "rate_limited", f"{running} running, {hour} in the last hour")

    def _repeat_of(self, dataset_hash: str, base: str) -> str | None:
        since = time.time() - DAY
        return next((j.id for j in self.jobs.values()
                     if j.rec["dataset_sha256"] == dataset_hash and j.rec["base"] == base
                     and j.rec["created"] > since and j.rec["stage"] != "failed"), None)

    async def train(self, dataset: str, base: str, min_tier: str, max_usd: float,
                    idempotency_key: str, method: str = "memory", seller: str | None = None,
                    alias: str | None = None, human: bool = False, tried: tuple = (),
                    within: float | None = None) -> dict:
        """Both policy arguments and the key are required: no default spends money.
        `within` is a price already approved: a retry at that price or less needs no one."""
        if min_tier is None or max_usd is None or not idempotency_key:
            raise bad_request("min_tier, max_usd and idempotency_key are required")
        for job in self.jobs.values():
            if job.rec["idempotency_key"] == idempotency_key:
                return self.view(job)
        policy = self.policy.ready()
        self._rate_limit(policy)
        q = await asyncio.to_thread(self.quote, dataset, base, method, min_tier, max_usd, seller, tried)
        offer, profile = q.pop("_offer"), mp.get(q["base"])
        human = human or (within is not None and q["usd"] <= within)
        entry = self.datasets.get(dataset)
        request = {"dataset": dataset, "base": profile.name, "offer": q["offer"], "usd": q["usd"]}
        over, again = {**request, "why": "over the automatic amount"}, {**request, "why": "repeat"}
        approved = human or self.approvals.granted(over)
        repeat = self._repeat_of(entry["sha256"], profile.name)
        if repeat and not (human or self.approvals.granted(again)):
            ticket = self.approvals.request(again)
            raise approval_required(f"same data and model as {repeat} within 24 hours", ticket)
        key = f"job-{idempotency_key}"
        self.ledger.reserve(key, q["usd"], "train", policy, cap=max_usd, approved=approved,
                            approvals=self.approvals, reason=request)
        used = [] if human else self.approvals.take(over, again)
        try:
            job = await self._open_job(q, offer, profile, entry, dataset, key, idempotency_key, alias,
                                       {"min_tier": min_tier, "max_usd": max_usd, "tried": [*tried, q["offer"]],
                                        "pinned": seller is not None})
        except Exception:
            self.ledger.release(key)
            self.approvals.give_back(used)
            raise
        self.ledger.update(key, "escrowed")
        job.task = asyncio.create_task(job.run())
        return self.view(job)

    async def _open_job(self, q: dict, offer: Offer, profile: mp.Profile, entry: dict, dataset: str,
                        ledger_key: str, idempotency_key: str, alias: str | None, request: dict) -> TrainJob:
        chain, wallet = self.chain, self.wallet
        await asyncio.to_thread(self.require_funds, q["usd"])
        memories, hidden = self.datasets.load(dataset)
        req = Requester(self.storage_key, bytes(wallet.pubkey()))
        tmp = self.home / "jobs" / f".new-{os.urandom(4).hex()}"
        budget = profile.budget(q["units"])  # within what this offer's deadline leaves
        budget["max_wall_seconds"] = max(min(budget["max_wall_seconds"], q["run_budget_s"] - RUN_RESERVE_S), 1)
        spec_job = await asyncio.to_thread(
            req.prepare_job, tmp, memories, hidden, profile.base_model(), profile.recipe(),
            profile.acceptance(), budget, chain=chain.spec_chain(),
            open_slot=await asyncio.to_thread(chain.current_slot))
        root = self.home / "jobs" / spec_job["job_id"][:16]
        tmp.rename(root)
        (root / "out").mkdir()
        job = TrainJob(self, root, {
            "job_id": spec_job["job_id"], "created": time.time(), "stage": "quoted", "timeline": [],
            "dataset": dataset, "dataset_sha256": entry["sha256"], "label": entry["label"],
            "base": profile.name, "floor": q["floor"], "offer": q["offer"], "units": q["units"],
            "usd": q["usd"], "ledger": ledger_key, "idempotency_key": idempotency_key, "alias": alias,
            "method": "memory", "seller": str(Seller.decode(offer.seller, chain._account(offer.seller)).authority),
            "gpu": {"name": wire.gpu_name(offer.hw_class), "memory_gb": offer.mem_gb}, **request})
        job.seal_secrets(req)
        job.mark("quoted", f"${q['usd']:.2f}")
        try:
            opened = await asyncio.to_thread(chain.open_job, wallet, wallet, offer, job.job_id, req.nonce,
                                             req.open_slot, req.spec_hash, q["units"], offer.quote(q["units"])[0])
        except RuntimeError as e:  # someone took the last slot: there was never a job
            shutil.rmtree(root, ignore_errors=True)
            raise no_seller(f"the offer could not be taken: {str(e)[:120]}", retry_after=10) from None
        job.signed("escrow", opened)
        job.stage("escrowed", f"${q['usd']:.4f} in escrow on chain")
        self.jobs[job.id] = job
        return job

    async def try_next_seller(self, job: TrainJob) -> None:
        """A job that never got going costs nothing. Go on to the next offer, a few
        times, never to one below the floor. Past the approved price, a person decides,
        and a buyer who named a seller meant that seller."""
        rec = job.rec
        if rec.get("pinned"):
            rec["next_error"] = "seller_pinned"
            job.save()
            return
        if len(rec["tried"]) >= MAX_SELLERS:
            return
        try:
            view = await self.train(rec["dataset"], rec["base"], rec["min_tier"], rec["max_usd"],
                                    f"{rec['idempotency_key']}+{len(rec['tried'])}", alias=rec["alias"],
                                    within=rec["usd"], tried=tuple(rec["tried"]))
        except BeetleError as e:
            rec["next_error"] = e.code
        else:
            rec["next_job"] = view["job"]
        job.save()

    def job(self, ident: str) -> TrainJob:
        ident = ident.removeprefix("job_")
        hits = [j for k, j in self.jobs.items() if k.startswith(ident) or j.rec["job_id"] == ident]
        if len(hits) != 1:
            raise not_found(f"no job {ident!r}" if not hits else f"{ident!r} is ambiguous")
        return hits[0]

    def view(self, job: TrainJob) -> dict:
        rec = job.rec
        keep = ("stage", "base", "dataset", "label", "floor", "usd", "offer", "gpu", "timeline", "ended",
                "error", "verdict", "result", "alias", "link_error", "next_job", "next_error")
        return {"job": job.id, "job_id": rec["job_id"], **{k: rec.get(k) for k in keep}}

    async def job_status(self, ident: str) -> dict:
        job = self.job(ident)
        onchain = await asyncio.to_thread(self.chain.get_job, job.job_id)
        return {**self.view(job), "chain_state": onchain.state if onchain else "closed"}

    async def fetch(self, ident: str, alias: str | None = None) -> dict:
        """Verify the result, open the adapter, prove it changes answers, then settle."""
        job = self.job(ident)
        rec = job.rec
        if rec["stage"] not in ("delivered", "settled"):
            raise bad_request(f"job is {rec['stage']}: nothing to fetch yet")
        if rec.get("result") and rec["stage"] == "settled":
            return self.view(job)
        onchain = await asyncio.to_thread(self.chain.get_job, job.job_id)
        checked = await asyncio.to_thread(job.verify_result, onchain)
        if not all(checked["checks"].values()):
            rec["error"] = {"code": "result_rejected", "message": str(checked["checks"])}
            job.save()
            await self._refuse_delivery(job, onchain)
            raise BeetleError(502, "result_rejected", f"result does not verify: {checked['checks']}")
        report, private = checked["report"], checked["private"]
        result = {"passed": report["passed"], "outcome": report["outcome"], "work_bps": report["work_bps"]}
        if private:
            training = private["training"]
            result["scores"] = {"before": private["personal_qa"]["before"],
                                "after": private["personal_qa"]["after"],
                                "regression_delta": private.get("regression_loss", {}).get("delta"),
                                "steps": training.get("steps"), "tokens": training.get("train_tokens"),
                                "wall_s": training.get("wall_seconds")}
        rec["result"] = result
        job.save()
        if report["passed"]:
            job.open_adapter(checked)
            files = job.adapter_files()
            try:
                result["effect"] = await self._check_effect(job, files)
            except BeetleError as e:
                if e.code == "adapter_no_effect":  # the buyer's check failed: that's what reject is for
                    await self._refuse_delivery(job, onchain)
                raise
            version = {"job": job.id, "base": rec["base"], "label": rec["label"], "dataset": rec["dataset"],
                       "adapter_sha256": report["adapter_pt_hash"], "at": int(time.time()),
                       "effect": result["effect"], "scores": result.get("scores")}
            self.models.add(alias or rec.get("alias") or "me", version)
            rec["alias"] = alias or rec.get("alias") or "me"
            job.mark("verified", "the result passed its checks")
        paid = await self._settle(rec, report, onchain)
        self.ledger.update(rec["ledger"], "settled", paid)
        result["paid_usd"] = paid
        rec.pop("error", None)
        job.stage("settled", f"paid ${paid:.4f}")
        job.forget_ciphertext()
        return self.view(job)

    async def _settle(self, rec: dict, report: dict, onchain) -> float:
        """What this job cost. Accepting races the seller's finalize, and losing that
        race is fine: it settled without this buyer, at the floor unless T2 proved a pass."""
        chain, wallet = self.chain, self.wallet
        if onchain is not None and onchain.state == "Delivered":
            try:
                rec.setdefault("tx", {})["settlement"] = await asyncio.to_thread(chain.accept, wallet, onchain)
                return onchain.expected_dues(accepted=True)[0] / USDC
            except RuntimeError:
                if await asyncio.to_thread(chain.get_job, onchain.job_id) is not None:
                    raise
                return self._paid_without_accepting(rec, report)
        if onchain is not None:
            rec.setdefault("tx", {})["settlement"] = await asyncio.to_thread(chain.claim_due, wallet, onchain.job_id)
        return self._paid_without_accepting(rec, report)

    @staticmethod
    def _paid_without_accepting(rec: dict, report: dict) -> float:
        if report["passed"] and rec.get("verdict", {}).get("tier") == "T2":
            return rec["usd"]
        return rec["usd"] * COMPUTE_FLOOR_BPS * report["work_bps"] / BPS**2

    async def reject(self, ident: str) -> dict:
        """The buyer's own verdict on a delivery: settle at the floor. Off T2 only."""
        job = self.job(ident)
        onchain = await asyncio.to_thread(self.chain.get_job, job.job_id)
        if onchain is None or onchain.state != "Delivered":
            raise bad_request(f"job is {onchain.state if onchain else 'closed'}: nothing to reject")
        if onchain.tier == TIER_T2:
            raise bad_request("a T2 result is proven by its sandbox and can't be refused")
        await self._refuse_delivery(job, onchain)
        return self.view(job)

    async def _refuse_delivery(self, job: TrainJob, onchain) -> None:
        """Off T2 a seller's word pays nothing extra: settle at the floor."""
        if onchain is not None and onchain.state == "Delivered" and onchain.tier != TIER_T2:
            try:
                await asyncio.to_thread(self.chain.reject, self.wallet, onchain)
            except RuntimeError:  # finalize got there first, with the same floor
                pass
            paid = onchain.expected_dues()[0] / USDC
            self.ledger.update(job.rec["ledger"], "settled", paid)
            job.rec.setdefault("result", {})["paid_usd"] = paid
            job.stage("settled", f"delivery refused; paid the floor, ${paid:.4f}")

    async def _check_effect(self, job: TrainJob, files: dict) -> dict:
        """On the route this model will use. Remote checks happen at every session."""
        profile = profile_of(job.rec)
        _, hidden = self.datasets.load(job.rec["dataset"])
        items = questions(hidden)
        caps = await asyncio.to_thread(doc.machine, self.home, self.cfg)
        if doc.can_serve(caps, profile):
            ask = [[{"role": "user", "content": it["question"]}] for it in items]
            base = [await self.engine.answer(profile, None, None, m, dict(PARAMS)) for m in ask]
            tuned = [await self.engine.answer(profile, job.id, lambda: files, m, dict(PARAMS)) for m in ask]
            return {"where": "local", **require_effect(compare(items, base, tuned), "this machine")}
        target = Target(job.rec["alias"] or job.id, profile, job.id, label=job.rec["label"],
                        dataset=job.rec["dataset"])
        floor = self.floor_for(job.rec["label"], None)
        session = await self.remote_session(target, floor, None)
        return {"where": "remote", **session.effect}

    async def cancel(self, ident: str) -> dict:
        """Before the key is out, the job stops and the escrow comes back: in full
        once the seller misses its start window, less a capped fee after `start`.
        After release a job can't be recalled; it ends by the chain's rules."""
        job = self.job(ident)
        chain, wallet = self.chain, self.wallet
        onchain = await asyncio.to_thread(chain.get_job, job.job_id)
        if job.task:
            job.task.cancel()
        if onchain is not None and onchain.state == "Started":
            await asyncio.to_thread(chain.reject_evidence, wallet, onchain)
            await asyncio.to_thread(chain.claim_due, wallet, job.job_id)
            job.rec["ended"] = "cancelled before the key was released; refunded less the capped fee"
            self.ledger.release(job.rec["ledger"])
            job.stage("failed", job.rec["ended"])
        elif onchain is not None and onchain.state == "Opened":
            job.task = asyncio.create_task(self._refund_when_due(job))
            job.rec["cancelled"] = True
            job.rec["ended"] = "cancelled; the refund follows the seller's start window"
            job.save()
        else:
            job.rec["ended"] = f"too late to cancel: job is {onchain.state if onchain else 'closed'}"
            job.save()
        return self.view(job)

    async def _refund_when_due(self, job: TrainJob) -> None:
        try:
            while True:
                await job.check_alive()
                await asyncio.sleep(5)
        except JobEnded:
            pass

    # --- chat -----------------------------------------------------------------------

    def resolve(self, model: str) -> Target:
        name = model
        if model == "auto":
            name = "me" if self.models.current("me") else self.pick_base("auto").name
        version = self.models.current(name)
        if version:
            return Target(name, mp.get(version["base"]), version["job"], version["adapter_sha256"],
                          version["label"], version["dataset"])
        try:
            return Target(name, mp.get(name))
        except KeyError:
            raise not_found(f"no model {model!r}") from None

    async def remote_session(self, target: Target, floor: str, max_usd: float | None,
                             seller: str | None = None) -> RemoteSession:
        if not accepts(floor, SESSION_TIER):  # no market lookup, no channel, nothing sent
            raise no_route(f"inference shows no evidence yet, so it can't meet floor {floor}", floor=floor)
        key = (target.profile.name, target.adapter, seller)
        async with self.opening.setdefault(key, asyncio.Lock()):
            session = self.sessions.get(key)
            if session and session.pipe_alive():
                return session
            offers = await asyncio.to_thread(self._matching, KIND_INFERENCE, target.profile, floor)
            offers = [o for o in offers if seller is None or str(o.address) == seller]
            if not offers:
                others = await asyncio.to_thread(self.market, "infer", target.profile.name)
                if others:
                    raise no_route(f"no seller of {target.profile.name} meets floor {floor}",
                                   floor=floor, other_tiers=sorted({r["tier"] for r in others}))
                raise no_seller(f"no one is selling inference on {target.profile.name} right now")
            error = None
            for offer in sorted(offers, key=lambda o: o.price_a + o.price_b)[:MAX_SELLERS]:
                try:
                    session = await self._start_session(target, floor, max_usd, offer)
                except BeetleError as e:
                    if e.code != "no_seller":
                        raise
                    error = e  # an offer can outlive its seller: the next one, same floor
                    continue
                self.sessions[key] = session
                return session
            raise error

    async def _start_session(self, target: Target, floor: str, max_usd: float | None, offer: Offer) -> RemoteSession:
        policy = self.policy.ready()
        deposit_usd = min(policy["infer_deposit_usd"], policy["per_job_usd"],
                          max_usd if max_usd is not None else float("inf"))
        await asyncio.to_thread(self.require_funds, deposit_usd)
        ledger_key = f"channel-{os.urandom(4).hex()}"
        self.ledger.reserve(ledger_key, deposit_usd, "infer", policy, approved=True)
        session = RemoteSession(self, offer, target.profile)
        session.ledger_key = ledger_key
        self.starting.add(session)
        try:
            await session.open(int(deposit_usd * USDC), floor)
            if target.adapter:
                files = self.jobs[target.adapter].adapter_files()
                hidden = self.datasets.load(target.dataset)[1] if target.dataset else None
                await session.load_adapter(files, hidden)
        except BaseException as e:
            await session.abandon()  # a channel already open comes back through sweep_channels
            if isinstance(e, TimeoutError):
                raise no_seller("the seller did not answer", retry_after=30) from None
            raise
        finally:
            self.starting.discard(session)
        session.last_used = time.time()  # or the next sweep sees it as long idle
        return session

    def _routes(self, target: Target, want: str, floor: str, caps: dict) -> list[str]:
        local = doc.can_serve(caps, target.profile)
        remote = bool(ACCEPT[floor]) and self.chain is not None
        if want == "local":
            if not local:
                raise no_route(f"{target.profile.name} can't run on this machine",
                               reason=doc.path_for(caps, target.profile)["why"])
            return ["local"]
        if want == "remote":
            if not remote:
                raise no_route(f"floor {floor} rules out remote sellers", floor=floor)
            return ["remote"]
        routes = (["local"] if local else []) + (["remote"] if remote else [])
        if not routes:
            raise no_route(f"nothing meets floor {floor} for {target.profile.name}", floor=floor)
        return routes

    async def open_chat(self, model: str, messages: list[dict], params: dict,
                        options: dict) -> "Route":
        """Pick a route and get the first piece out of it. A failure before the first
        piece tries the next route; after it, the caller sees the error. Never a
        route below the floor: T0 is always possible and nothing else is assumed."""
        target = self.resolve(model)
        floor = self.floor_for(target.label, options.get("min_tier"))
        caps = await asyncio.to_thread(doc.machine, self.home, self.cfg)
        error = None
        for kind in self._routes(target, options.get("route", "auto"), floor, caps):
            route = Route(self, kind, target, messages, params)
            try:
                await route.start(floor, options.get("max_usd"), options.get("seller"))
                return route
            except BeetleError as e:
                error = e
            except Exception as e:
                error = BeetleError(502, "route_failed", f"{kind}: {e!r}"[:200])
        raise error

    async def close_idle(self, idle_s: float = SESSION_IDLE_S) -> None:
        for key, session in list(self.sessions.items()):
            if time.time() - session.last_used >= idle_s or not session.pipe_alive():
                del self.sessions[key]
                try:
                    await session.close()
                except Exception:  # sweep_channels gets the money back
                    pass

    def sweep_channels(self) -> None:
        """Every channel that isn't in use ends here. A seller normally settles it when
        the session closes. One that left: ask to close, wait out the grace, take the rest."""
        chain, wallet = self.chain, self.wallet
        live = {s.channel for s in [*self.sessions.values(), *self.starting]}
        for key, entry in list(self.ledger.entries.items()):
            if entry["kind"] != "infer" or entry["state"] != "escrowed" or "channel" not in entry:
                continue
            address = Pubkey.from_string(entry["channel"])
            if address in live:
                continue
            ch = chain.get_channel(address)
            if ch is None or ch.status == "Distributed":
                # A closed account says nothing. The last voucher is what a seller could take.
                self.ledger.update(key, "settled", ch.settled / USDC if ch else entry.get("signed", 0.0))
            elif ch.status == "Open":
                if time.time() - entry.get("closed_at", 0) > SELLER_SETTLES_S:
                    chain.request_close(wallet, ch)
            elif ch.status == "Closing" and chain.chain_time() > ch.closure_started_at + ch.grace_period:
                chain.seal_channel(wallet, ch)
            elif ch.status == "Sealed":
                chain.distribute(wallet, ch)

    def save_proof(self, proof: dict) -> str:
        ident = f"prf_{os.urandom(6).hex()}"
        wire.write_json(self.home / "proofs" / f"{ident}.json", proof)
        return ident

    def proof(self, ident: str) -> dict:
        if ident.startswith("prf_"):
            found = wire.read_json(self.home / "proofs" / f"{ident}.json")
            if found is None:
                raise not_found(f"no proof {ident}")
            return found
        job = self.job(ident)
        found = job.rec.get("proof") or card("no_evidence", evidence={"reason": "no evidence yet"})
        return {**found, "receipt": self.receipt(job)}

    def receipt(self, job: TrainJob) -> dict:
        """What one job was: on what, with what, for how much, and how it ended."""
        rec, profile = job.rec, mp.get(job.rec["base"])
        result, times = rec.get("result") or {}, [t["at"] for t in rec["timeline"]]
        scores, effect = result.get("scores") or {}, result.get("effect") or {}
        ended = {"settled": "settled", "failed": rec.get("ended") or "failed"}.get(rec["stage"], "in escrow")
        return {"gpu": rec.get("gpu"), "model": profile.name, "model_revision": profile.revision,
                "model_hash": profile.manifest_sha256, "dataset_hash": rec["dataset_sha256"],
                "job_id": rec["job_id"], "duration_s": times[-1] - times[0] if times else None,
                "cost_usd": result.get("paid_usd"), "held_usd": rec["usd"] if ended == "in escrow" else None,
                "evaluation": {"passed": result.get("passed"), "before": scores.get("before"),
                               "after": scores.get("after"), "asked": effect.get("asked"),
                               "base_correct": effect.get("base_correct"),
                               "adapter_correct": effect.get("adapter_correct")},
                "settlement": ended, "transactions": rec.get("tx", {})}

    def open_jobs(self) -> list[dict]:
        """What anyone can read on chain: addresses, amounts and states, no content."""
        return [{"job": j.job_id.hex()[:16], "buyer": str(j.requester), "seller": str(j.host),
                 "usd": j.price / USDC, "state": j.state, "opened_at": j.opened_at}
                for j in self.require_chain().open_jobs()]

    def balance(self) -> float:
        try:
            return self.chain.token_balance(self.wallet.pubkey())
        except RuntimeError:  # no token account until the first deposit
            return 0.0

    def require_funds(self, usd: float) -> None:
        """An empty wallet says so here, not as a failed transaction."""
        if self.chain.sol_lamports(self.wallet.pubkey()) < MIN_FEE_LAMPORTS:
            raise no_funds("this wallet has no SOL to pay network fees")
        if (have := self.balance()) < usd:
            raise no_funds(f"this wallet holds ${have:.2f} and this needs ${usd:.2f}")

    def wallet_view(self) -> dict:
        chain = self.require_chain()
        me = self.wallet.pubkey()
        usdc = self.balance()
        policy = self.policy.load()
        return {"address": str(me), "usdc": usdc, "sol": chain.sol_lamports(me) / 1e9,
                "spending": self.ledger.summary(), "limits": {k: policy.get(k) for k in (
                    "per_job_usd", "per_day_usd", "auto_approve_under_usd", "min_tier")},
                "entries": self.ledger.entries}


async def without_thinking(pieces, end: str | None):
    """Some models open with reasoning that ends at a marker. The caller of an
    OpenAI-style API wants the answer. Without the marker nothing is held back
    for long; without a marker configured, nothing at all."""
    held, passing = "", end is None
    async for piece in pieces:
        if passing:
            yield piece
            continue
        held += piece
        if end in held:
            passing, held = True, held.split(end, 1)[1].lstrip()
        elif len(held) > THINK_HOLD:
            passing = True
        if passing and held:
            yield held
    if not passing and held:
        yield held


class Route:
    """One chat on one route, with what the response headers must say."""

    def __init__(self, core: Core, kind: str, target: Target, messages: list[dict], params: dict):
        self.core, self.kind, self.target = core, kind, target
        self.messages, self.params = messages, params
        self.usage: dict = {}
        self.tier = "T0"
        self.session: RemoteSession | None = None
        self.cost = 0
        self.proof_id = ""
        self.pieces = None
        self.first = ""

    async def start(self, floor: str, max_usd: float | None, seller: str | None = None) -> None:
        core, target = self.core, self.target
        if self.kind == "local":
            files = core.jobs[target.adapter].adapter_files if target.adapter else None
            self.pieces = core.engine.stream(target.profile, target.adapter, files, self.messages,
                                             self.params, self.usage)
            self.proof_id = core.save_proof(local_card())
        else:
            self.session = await core.remote_session(target, floor, max_usd, seller)
            self.tier = self.session.tier
            self.session.last_used = time.time()
            if not accepts(floor, self.tier):
                raise no_route(f"the session proved {self.tier}, below the floor {floor}", floor=floor)
            self.pieces = self.session.chat(self.messages, self.params, self.session.adapter)
            self.proof_id = core.save_proof(self.session.proof)
        self.pieces = without_thinking(self.pieces, target.profile.chat.get("think_end"))
        self.first = await anext(self.pieces, "")

    async def stream(self):
        finished = False
        try:
            if self.first:
                yield self.first
            async for piece in self.pieces:
                yield piece
            finished = True
            if self.session:
                self.usage.update(self.session.usage)
                self.cost = self.session.last_cost
                self.session.last_used = time.time()
        finally:
            if not finished:  # the caller left; a session must still read to the end
                asyncio.ensure_future(self._drain())

    async def _drain(self) -> None:
        async for _ in self.pieces:
            pass

    @property
    def model_id(self) -> str:
        base = f"sha256:{self.target.profile.manifest_sha256}"
        return f"{base}+sha256:{self.target.adapter_sha256}" if self.target.adapter_sha256 else base

    def cost_usd(self) -> float:
        return self.cost / USDC

    def headers(self) -> dict:
        return {"x-beetle-route": self.kind, "x-beetle-tier": self.tier,
                "x-beetle-model": self.model_id, "x-beetle-proof": self.proof_id}

