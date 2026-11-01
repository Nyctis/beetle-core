"""One training job, from the escrow to the settlement, as the buyer runs it.

    escrowed -> uploading -> evidence -> released -> training -> delivered -> settled

The result comes in two parts. The encrypted adapter arrives first, and the
buyer's signature says it holds it. The report, which carries the adapter's
sealed key, comes only once the seller's result is on chain, so nobody holds both
a usable adapter and a claim to a refund. A buyer who was away finds the seller
afterwards, for as long as the seller's retention lasts: the job's account may be
gone by then, so it reaches the seller through memos on the job's address.

Everything a restart needs is on disk or on chain: record.json holds the stage,
secrets.bin the job's one-time keys (sealed with the storage key), so the daemon
picks a job up where it stopped. A job that ended badly says how the money went.

The buyer holds the only checks that count. Spec and inputs are bound by hashes
on chain. The evidence must match the hash the seller committed to, verify here,
and meet the floor, or the key is never released. Results must be signed by the
sandbox key from the chain. Links carry ciphertext and may fail at any point;
each step retries on a new link.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path

from solders.pubkey import Pubkey

import beetle_crypto as bc
import model_profiles as mp
from beetle_chain import FINISHED
from client.errors import evidence_rejected
from client.policy import CHAIN_TIER, USDC, accepts
from client.proof import card
from job_defaults import sandbox_code_files
from net import transfer
from net.peer import Peer
from node import wire
from requester import AttestationRejected, Requester, load_stored_adapter
from verifier import EvidenceVerifier

STAGES = ("quoted", "escrowed", "uploading", "evidence", "released", "training", "delivered",
          "settled", "failed")
WINDOW = 30.0
POLL_CHAIN = 1.0
POLL_RESULTS = 2.0
RETRY = 3.0
RELEASE_MARGIN = 3    # seconds left in the release window when evidence is given up on
RECOVER_TRIES = 5     # dials to a seller after the job's account has closed
OUTPUTS = ("report.json", "report.sig", "adapter.enc", "private_report.enc")
MAX_OUTPUT = 1 << 30
RETRY_ELSEWHERE = {"Declined", "Unstarted", "NoStart"}  # a job that never got going


class JobEnded(Exception):
    """The job can't go on; `record.json` already says why."""


def training_note(tier: str) -> str:
    if tier in ("T1", "T2"):
        return "sandbox is training; the seller sees ciphertext and phase names"
    return "the seller's machine is training; its owner can read the data"


class TrainJob:
    def __init__(self, core, root: Path, record: dict):
        self.core, self.root, self.rec = core, root, record
        self.task: asyncio.Task | None = None
        self.peer: Peer | None = None
        self.link = None
        self.closed = False       # the account is gone, the seller may still hold the result
        self.recover_tries = 0

    @property
    def job_id(self) -> bytes:
        return bytes.fromhex(self.rec["job_id"])

    @property
    def id(self) -> str:
        return self.rec["job_id"][:16]

    def save(self) -> None:
        wire.write_json(self.root / "record.json", self.rec)

    def stage(self, name: str, note: str = "") -> None:
        self.rec["stage"] = name
        self.mark(name, note)

    def mark(self, name: str, note: str = "") -> None:
        """A step to show in the timeline. On its own it doesn't move the job on."""
        self.rec["timeline"].append({"stage": name, "at": int(time.time()), "note": note})
        self.save()

    def signed(self, name: str, signature: str | None) -> None:
        self.rec.setdefault("tx", {})[name] = signature
        self.save()

    def requester(self) -> Requester:
        blob = (self.root / "secrets.bin").read_bytes()
        secrets = json.loads(bc.open_blob(self.core.storage_key, blob, b"job-secrets"))
        return Requester.restore(self.core.storage_key, bytes(self.core.wallet.pubkey()), secrets)

    def seal_secrets(self, req: Requester) -> None:
        path = self.root / "secrets.bin"
        path.write_bytes(bc.seal_blob(self.core.storage_key, bc.canonical_json(req.secrets()), b"job-secrets"))
        path.chmod(0o600)

    async def chain(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    # --- the link ------------------------------------------------------------------

    async def _dial(self):
        cfg = self.core.cfg
        if self.peer is None:
            self.peer = await wire.bound_peer(cfg)  # a new identity per job, so sellers can't link jobs
        rendezvous = await self.chain(self._rendezvous)
        nodes = await asyncio.to_thread(self.core.echo_nodes)
        await rendezvous.publish({"id": self.peer.id, "addrs": await wire.candidates(self.peer, cfg, nodes)})
        them = await rendezvous.wait(WINDOW)
        self.link = await self.peer.link(them["id"], them["addrs"], initiator=True, window=WINDOW)
        if hasattr(rendezvous, "close"):  # the on-chain account's deposit comes back
            try:
                await rendezvous.close()
            except RuntimeError as e:  # a good link matters more than the deposit, for now
                self.rec["rendezvous_note"] = str(e)[:120]
        self.rec["path"] = wire.path_of(self.link)
        self.rec["peer_endpoint"] = them["id"]
        if not any(t["stage"] == "connected" for t in self.rec["timeline"]):
            self.mark("connected", "direct" if self.rec["path"].get("direct") else "through a relay")
        self.save()

    def _rendezvous(self):
        chain, wallet, cfg = self.core.chain, self.core.wallet, self.core.cfg
        if not self.closed:
            return wire.job_rendezvous(cfg, self.job_id, "buyer", chain, wallet)
        seller = Pubkey.from_string(self.rec["seller"])
        return wire.closed_job_rendezvous(cfg, self.job_id, "buyer", chain, wallet, seller,
                                          net_pk=chain.get_seller(seller).net_pk)

    async def ask(self, step) -> dict:
        """Run a step on the link, dialing or redialing as needed, until the job ends."""
        while True:
            try:
                if self.link is None:
                    await self._dial()
                return await step(self.link)
            except (JobEnded, transfer.TransferError, asyncio.CancelledError):
                raise
            except Exception as e:  # any link can fail; the chain says whether to go on
                self.link = None
                self.rec["link_error"] = repr(e)[:160]
                await self.check_alive(unreachable=True)
                await asyncio.sleep(RETRY)

    async def check_alive(self, unreachable: bool = False) -> None:
        """Take what the chain owes when the other side stopped. Most of this
        anyone may do. A seller that can't be reached before it started isn't
        at fault, as far as anyone can tell: refund, no slash."""
        chain, wallet = self.core.chain, self.core.wallet
        job = await self.chain(chain.get_job, self.job_id)
        if self.rec.get("released") and (job is None or job.state == "Settled"):
            self.closed = True  # paid or not, the seller may still hold what this job made
            self.recover_tries += unreachable
            if self.recover_tries >= RECOVER_TRIES:
                await self._give_up_on_result(job)
            return
        if job is None or job.state in FINISHED:
            await self.finished(job, f"job ended on chain: {job.state if job else 'closed'}",
                                retry=job is not None and job.state in RETRY_ELSEWHERE)
        now = await self.chain(chain.chain_time)
        config = await self.chain(chain.get_config)
        if unreachable and job.state == "Opened" and job.rdv_opened and now >= job.opened_at + config["rdv_abort_after"]:
            await self.chain(chain.abort_unstarted, wallet, job)
            await self.chain(chain.claim_due, wallet, self.job_id)
            if job.rdv_answered:  # it was there, and now this buyer has its address
                fee = min(config["reject_fee_cap"], job.price) / USDC
                await self.finished(None, f"no link to a seller that answered: refunded less a ${fee:.2f} fee, "
                                          "the chain's price for leaving after reading a seller's address",
                                    retry=True, kept=fee)
            await self.finished(None, "no link to the seller: refunded, no fault", retry=True)
        if job.state == "Opened" and now > job.start_by:
            await self.chain(chain.timeout_start, wallet, job)
            await self.chain(chain.claim_due, wallet, self.job_id)
            await self.finished(None, "seller never started: refunded, seller penalized", retry=True)
        if job.state == "Running" and now > job.deadline:
            await self.chain(chain.timeout_run, wallet, job)
            await self.chain(chain.claim_due, wallet, self.job_id)
            await self.finished(None, "seller ran out of time: refunded, seller penalized")

    async def _give_up_on_result(self, job) -> None:
        """A closed account no longer says whether this was refunded. The limit
        counts it as spent: better to under-spend a day than to over-spend one."""
        self.rec["ended"] = "the seller did not answer; its copy of the result is gone"
        self.core.ledger.update(self.rec["ledger"], "settled", self.rec["usd"])
        self.stage("failed", self.rec["ended"])
        raise JobEnded(self.rec["ended"])

    async def finished(self, job, note: str, retry: bool = False, kept: float = 0.0) -> None:
        if job is not None:  # money still owed to this wallet goes out with the last claim
            await self.chain(self.core.chain.claim_due, self.core.wallet, self.job_id)
        self.rec["ended"], self.rec["retry"] = note, retry
        if kept:
            self.core.ledger.update(self.rec["ledger"], "settled", kept)
        else:
            self.core.ledger.release(self.rec["ledger"])
        self.stage("failed", note)
        raise JobEnded(note)

    # --- the steps -----------------------------------------------------------------

    async def run(self) -> None:
        try:
            for name, step in (("uploaded", self._upload), ("released", self._release),
                               ("received", self._collect)):
                if not self.rec.get(name):
                    await step()
            self.stage("delivered", "result held here, sealed")
            if self.rec.get("alias"):
                await self.core.fetch(self.id, self.rec["alias"])
        except JobEnded:
            if self.rec.get("retry"):
                await self.core.try_next_seller(self)
        except Exception as e:
            self.rec["error"] = {"code": getattr(e, "code", "internal"),
                                 "message": str(e)[:300]}
            if self.rec["error"]["code"] == "no_funds":  # the result is safe here: add funds, fetch again
                self.save()
            else:
                self.stage("failed", str(e)[:200])
        finally:
            if self.peer:
                await self.peer.close()
                self.peer = None
            self.link = None

    async def _upload(self) -> None:
        inbox = self.root / "in"
        spec = json.loads((inbox / "spec.json").read_text())

        async def step(link) -> dict:
            self.stage("uploading", "sealed inputs going to the seller")
            await wire.send_msg(link, {"t": "spec", "spec": spec})
            answer = await wire.recv_msg(link)
            if answer["t"] != "need":
                raise RuntimeError(f"seller refused the spec: {answer}")
            for name in answer["names"]:
                await transfer.send(link, inbox / f"{name}.enc")
            return answer

        await self.ask(step)
        self.rec["uploaded"] = int(time.time())
        self.stage("evidence", "waiting for the sandbox's evidence")

    async def _release(self) -> None:
        chain, wallet = self.core.chain, self.core.wallet

        async def step(link) -> dict:
            await wire.send_msg(link, {"t": "evidence", "wait": 4})
            return await wire.recv_msg(link)

        while True:
            answer = await self.ask(step)
            if answer["t"] == "evidence":
                break
            await self.check_alive()
            await self._give_up_on_evidence()
        evidence = answer["evidence"]
        wire.write_json(self.root / "out" / "evidence.json", evidence)

        job = await self.chain(chain.get_job, self.job_id)
        while job is None or job.state == "Opened":
            await self.check_alive()
            await asyncio.sleep(POLL_CHAIN)
            job = await self.chain(chain.get_job, self.job_id)
        if job.state != "Started":
            await self.finished(job, f"job is {job.state}")

        req = self.requester()
        verifier = EvidenceVerifier(self.core.evidence_policy(), self.core.home / "verifier_state.json")
        floor = self.rec["floor"]
        try:
            verdict = req.check_evidence(verifier, evidence, job, self.core.mainnet)
            tier = CHAIN_TIER[verdict.tier]
            if not accepts(floor, tier):
                raise AttestationRejected(f"evidence shows {tier}, below the floor {floor}")
        except AttestationRejected as e:
            await self.chain(chain.reject_evidence, wallet, job)
            await self.chain(chain.claim_due, wallet, self.job_id)
            self.rec["verdict"] = {"rejected": str(e)}
            self.rec["proof"] = card("rejected", evidence={"verified": False, "reason": str(e)},
                                     path=self.rec.get("path"))
            self.core.ledger.release(self.rec["ledger"])
            self.stage("failed", f"key not released: {e}")
            raise evidence_rejected(str(e))
        self.rec["verdict"] = {"type": verdict.evidence_type, "tier": tier,
                               "measurement": verdict.measurement.hex(), "at": int(time.time())}
        self.rec["sandbox"] = {"sign_pk": job.sign_pk.hex(), "enc_pk": job.enc_pk.hex()}
        self.rec["proof"] = self._proof(tier, verdict.evidence_type, job)
        self.save()

        sealed = req.seal_dek()
        self.signed("key_release", await self.chain(chain.release_key, wallet, self.job_id, sealed, job.evidence_hash))
        self.rec["released"] = int(time.time())
        self.stage("released", "data key sealed to the sandbox, after local verification")
        self.stage("training", training_note(tier))

    async def _give_up_on_evidence(self) -> None:
        """Started, and nothing to verify: refusing costs less than letting the window lapse."""
        chain = self.core.chain
        job = await self.chain(chain.get_job, self.job_id)
        if job is not None and job.state == "Started" \
                and await self.chain(chain.chain_time) >= job.release_by - RELEASE_MARGIN:
            await self.chain(chain.reject_evidence, self.core.wallet, job)
            await self.chain(chain.claim_due, self.core.wallet, self.job_id)
            self.rec["proof"] = card("rejected", evidence={"verified": False, "reason": "no evidence arrived"})
            await self.finished(None, "the seller started but sent no evidence: refunded less the capped fee")

    def _proof(self, tier: str, kind: str, job) -> dict:
        return card(tier, peer={"seller": str(job.host), "offer": str(job.offer),
                                "transport": self.rec.get("peer_endpoint"), "environment": job.enc_pk.hex()},
                    evidence={"type": kind, "verified": kind != "NONE", "where": "on this machine",
                              "measurement": job.measurement.hex()},
                    path=self.rec.get("path"),
                    bindings={"job_id": True, "spec_hash": True, "sandbox_keys": True} if kind != "NONE" else {})

    async def _take(self, link, kind: str) -> dict:
        await wire.send_msg(link, {"t": kind})
        answer = await wire.recv_msg(link)
        if answer["t"] == kind:
            for name, root in answer["files"]:
                if name not in OUTPUTS:
                    raise RuntimeError(f"unexpected file {name!r} from the seller")
                await transfer.receive(link, bytes.fromhex(root), self.root / "out" / name, MAX_OUTPUT)
        return answer

    async def _until(self, step, kind: str) -> None:
        while True:
            if (await self.ask(step))["t"] == kind:
                return
            await self.check_alive()
            await asyncio.sleep(POLL_RESULTS)

    async def _take_ciphertext(self, link) -> dict:
        """Say so with a signature, or the seller waits out its delivery window."""
        answer = await self._take(link, "results")
        if answer["t"] == "results" and answer["files"]:
            held = bc.file_sha256(self.root / "out" / "adapter.enc")
            sig = bytes(self.core.wallet.sign_message(wire.ack_digest(self.job_id, held)))
            await wire.send_msg(link, {"t": "ack", "sig": sig.hex()})
            await wire.recv_msg(link)
        return answer

    async def _take_report(self, link) -> dict:
        answer = await self._take(link, "report")
        if answer["t"] == "report":
            sig = bytes(self.core.wallet.sign_message(wire.taken_digest(self.job_id)))
            await wire.send_msg(link, {"t": "done", "sig": sig.hex()})
            await wire.recv_msg(link)
        return answer

    async def _collect(self) -> None:
        await self.check_alive()
        if not self.rec.get("ciphertext"):
            await self._until(self._take_ciphertext, "results")
            self.rec["ciphertext"] = int(time.time())
            self.stage("training", "encrypted adapter held; its key comes once the result is on chain")
        await self._until(self._take_report, "report")
        self.rec["received"] = int(time.time())
        self.save()

    # --- after delivery -------------------------------------------------------------

    def verify_result(self, job) -> dict:
        """Everything a result has to be, before it is believed. `job` is None
        when the chain account is already closed; the signature is what counts."""
        out, req = self.root / "out", self.requester()
        req.attested_sign_pk = bytes.fromhex(self.rec["sandbox"]["sign_pk"])
        report = json.loads((out / "report.json").read_text())
        signature = (out / "report.sig").read_text()
        if report["private_report_ct_hash"] is None:
            ok = bc.verify(req.attested_sign_pk, bytes.fromhex(signature),
                           bc.digest("report", bc.canonical_json(report)))
            return {"report": report, "private": None, "checks": {"signature": ok}}
        adapter_ct, private_ct = (out / "adapter.enc").read_bytes(), (out / "private_report.enc").read_bytes()
        checks = req.verify_report(report, signature, adapter_ct, private_ct, self._expected_measurement())
        if job is not None and job.state == "Delivered":
            sealed = report["sealed_adapter_key"]
            checks["matches_chain"] = (job.report_hash == bc.sha256(bc.canonical_json(report))
                                       and job.outcome == report["outcome"]
                                       and job.adapter_ct_hash.hex() == report["adapter_ct_hash"]
                                       and (job.sealed_adapter_key.hex() if job.has_adapter_key else None) == sealed)
        return {"report": report, "checks": checks, "req": req, "adapter_ct": adapter_ct,
                "private": req.open_private_report(private_ct) if all(checks.values()) else None}

    def _expected_measurement(self) -> str:
        """What ran is judged once, when the evidence is. A sandbox that showed none
        claimed a measurement then, and this holds it to that claim; another version
        of the public code is no reason to refuse a delivery."""
        verdict = self.rec["verdict"]
        if verdict["type"] in ("NONE", "MOCK"):
            return verdict["measurement"]
        return bc.code_measurement(sandbox_code_files())

    def open_adapter(self, checked: dict) -> list[str]:
        return checked["req"].open_adapter(checked["report"], checked["adapter_ct"], self.root)

    def adapter_files(self) -> dict[str, bytes]:
        return load_stored_adapter(self.core.storage_key, self.root / "adapter.store", self.job_id)

    def forget_ciphertext(self) -> None:
        shutil.rmtree(self.root / "in", ignore_errors=True)
        (self.root / "out" / "adapter.enc").unlink(missing_ok=True)


def profile_of(rec: dict) -> mp.Profile:
    return mp.get(rec["base"])
