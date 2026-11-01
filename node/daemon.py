"""The seller's daemon: `beetle host run`.

Watches the chain for jobs and payment channels aimed at this seller, runs each
job in the offline sandbox, keeps offers fresh and collects what it is owed. It
holds the wallet, so it holds nothing else: the buyer's data key reaches the
sandbox sealed, only ciphertext crosses this process, and inference plaintext
stays in a child process that has no wallet, node/environment.py.

One job, as this process sees it:

    open_job on chain -> buyer links in, sends the public spec and sealed inputs
    -> sandbox starts and attests -> evidence goes to the buyer, `start` on chain
    -> sealed key read from the chain, handed to the sandbox -> sandbox trains
    -> the encrypted adapter goes to the buyer -> submit_result on chain
    -> the report goes to the buyer -> settled -> this job's inputs are deleted

The chain only ever sees hashes. Files travel over a direct link, so nothing
waits on shared storage, and the seller stays the only holder of the result
until the buyer has it. Three rules follow from that:

- The ciphertext goes first, with no way to open it, and the seller waits at most
  DELIVERY_WAIT for a buyer who is away, or it would lose the bond.
- `report.json` carries the sealed adapter key, so it goes out only after
  submit_result landed. Before that a buyer holding it could take the adapter and
  still claim a timeout refund.
- Settling takes an hour, and a buyer may be offline for longer. Results stay
  until the buyer says it has them, or for `retain_s`, after the job's account has
  closed. Then a buyer finds this seller through memos on the job's address.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
from solders.pubkey import Pubkey

import node  # noqa: F401  (import paths)

import beetle_crypto as bc
import model_profiles as mp
from attester import Registration
from beetle_chain import (FINISHED, NO_VERIFIER, OFFER_ACTIVE, OUTCOME_PASSED, TIER_T2, Job, Offer, OfferTerms,
                          evidence_hash)
from evidence import NoEvidence, TpmEvidence, register, report_data_for
from job_defaults import TIERS, sandbox_code_files, spec_problem
from net import echo, transfer
from node import wire
from node.metered import MIN_GRACE_S, Settler
from node.runtime import DockerRuntime
from requester import JOB_INPUTS
from sandbox_host import ensure_model, model_dir
from serve.gateway import Backend

POLL = 3.0
HEARTBEAT = 600.0
OFFER_LIFETIME = 6 * 86_400
ATTEST_TIMEOUT = 120.0
DELIVERY_WAIT = 300.0
SUBMIT_MARGIN = 120
DIAL_WINDOW = 30.0
ENV_START = 60.0
RETAIN_S = 7 * 86_400
ACCEPT_WAIT = 86_400  # off T2 finalize pays only the floor, so a buyer gets this long to accept
MAX_INPUT = 1 << 30
DECLINE_MARGIN = 2  # seconds before the chain stops letting a seller back out
NO_SHOW_S = 60      # how long a buyer gets to appear on the rendezvous
REPORTS = ("report.json", "report.sig", "private_report.enc")  # after submit_result only


class CannotServe(Exception):
    pass


def log(msg: str) -> None:
    print(f"[host] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def terms_of(offer: Offer, **changes) -> OfferTerms:
    """An offer's current terms, to post again with `changes`."""
    fields = {f: getattr(offer, f) for f in OfferTerms.__dataclass_fields__}
    return OfferTerms(**{**fields, **changes})


def measurement(host: dict) -> bytes:
    """What this machine's sandbox will attest."""
    image = bytes.fromhex(bc.code_measurement(sandbox_code_files()))
    if host.get("evidence") == "tpm":
        import tpm_attest as tpm

        return tpm.pcr_extend(bytes(32), image)
    return image


def offers_of(chain, authority: Pubkey) -> list[Offer]:
    seller = chain.seller_pda(authority)
    return [o for o in chain.list_offers(active_only=False, fresh_within=None) if o.seller == seller]


class JobBox:
    """One job as its links see it: ciphertext and public metadata, nothing else."""

    def __init__(self, job: Job, root: Path, vet=None):
        self.job, self.root, self.vet = job, root, vet
        self.inbox, self.outbox = root / "in", root / "out"
        self.spec: dict | None = None
        self.inputs = asyncio.Event()
        self.refused = ""
        self.evidence: dict | None = None
        self.attested = asyncio.Event()
        self.sealed: list[Path] | None = None  # what may leave before submit_result
        self.acked = asyncio.Event()
        self.submitted = False
        self.taken = False                     # the buyer signed that it has it all
        self.keep_until: float | None = None
        self.listener: asyncio.Task | None = None
        self.contacted = False  # the buyer has shown up on the rendezvous
        self.finished = False   # the daemon's part is over; only collecting remains

    def prepare(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.inbox.mkdir(parents=True)
        self.outbox.mkdir()
        self.outbox.chmod(0o777)

    def save(self) -> None:
        """What a restart needs to go on serving this result."""
        wire.write_json(self.root / "retained.json", {
            "job_id": self.job.job_id.hex(), "requester": str(self.job.requester),
            "adapter": bool(self.sealed), "keep_until": self.keep_until})

    @classmethod
    def retained(cls, root: Path) -> "JobBox | None":
        saved = wire.read_json(root / "retained.json")
        if saved is None:
            return None
        job = SimpleNamespace(job_id=bytes.fromhex(saved["job_id"]),
                              requester=Pubkey.from_string(saved["requester"]))
        box = cls(job, root)
        box.sealed = [box.outbox / "adapter.enc"] if saved["adapter"] else []
        box.submitted, box.finished, box.keep_until = True, True, saved["keep_until"]
        return box

    def release_inputs(self) -> None:
        shutil.rmtree(self.inbox, ignore_errors=True)
        (self.outbox / "settle.sig").unlink(missing_ok=True)

    async def handle(self, link) -> None:
        handlers = {"spec": self._spec, "evidence": self._evidence, "results": self._results,
                    "ack": self._ack, "report": self._report, "done": self._done}
        while True:
            msg = await wire.recv_msg(link)
            await handlers[msg["t"]](link, msg)

    async def _spec(self, link, msg: dict) -> None:
        spec = msg["spec"]
        if bc.sha256(bc.canonical_json(spec)) != self.job.spec_hash or set(spec["inputs"]) != set(JOB_INPUTS):
            await wire.send_msg(link, {"t": "refused", "reason": "spec does not match the chain"})
            return
        self.refused = (self.vet(spec) if self.vet else "") or ""
        if self.refused:
            await wire.send_msg(link, {"t": "refused", "reason": self.refused})
            self.inputs.set()
            return
        self.spec = spec
        (self.inbox / "spec.json").write_text(json.dumps(spec, indent=2))
        (self.inbox / "job.json").write_text(json.dumps(
            {"job_id": self.job.job_id.hex(), "spec_hash": self.job.spec_hash.hex(), **self.job.terms}))
        need = [n for n in spec["inputs"] if not self._holds(n)]
        await wire.send_msg(link, {"t": "need", "names": need})
        for name in need:
            path = self.inbox / f"{name}.enc"
            await transfer.receive(link, bytes.fromhex(spec["input_roots"][name]), path, MAX_INPUT)
            if not self._holds(name):
                path.unlink()
                raise transfer.TransferError(f"{name} does not match the spec")
        self.inputs.set()

    def _holds(self, name: str) -> bool:
        path = self.inbox / f"{name}.enc"
        return path.exists() and bc.file_sha256(path) == self.spec["inputs"][name]

    async def _evidence(self, link, msg: dict) -> None:
        try:
            await asyncio.wait_for(self.attested.wait(), msg.get("wait", 20))
            await wire.send_msg(link, {"t": "evidence", "evidence": self.evidence})
        except TimeoutError:
            await wire.send_msg(link, {"t": "wait"})

    async def _send_files(self, link, kind: str, paths: list[Path]) -> None:
        manifests = [(p, await asyncio.to_thread(transfer.manifest, p)) for p in paths]
        await wire.send_msg(link, {"t": kind, "files": [[p.name, transfer.root_of(m).hex()]
                                                        for p, m in manifests]})
        for p, m in manifests:
            await transfer.send(link, p, m)

    async def _results(self, link, msg: dict) -> None:
        """The ciphertext, which nothing opens yet."""
        if self.sealed is None:
            await wire.send_msg(link, {"t": "wait"})
            return
        await self._send_files(link, "results", self.sealed)

    async def _report(self, link, msg: dict) -> None:
        """Only once the chain holds the result: this carries the adapter's key."""
        if not self.submitted:
            await wire.send_msg(link, {"t": "wait"})
            return
        await self._send_files(link, "report", [self.outbox / f for f in REPORTS
                                                if (self.outbox / f).exists()])

    async def _ack(self, link, msg: dict) -> None:
        """The buyer's signature that it holds exactly what the sandbox sealed."""
        report = json.loads((self.outbox / "report.json").read_text())
        digest = wire.ack_digest(self.job.job_id, report["adapter_ct_hash"])
        if bc.verify(bytes(self.job.requester), bytes.fromhex(msg["sig"]), digest):
            self.acked.set()
            (self.root / "delivered.sig").write_text(msg["sig"])
        await wire.send_msg(link, {"t": "ok"})

    async def _done(self, link, msg: dict) -> None:
        """Nothing here is needed any more."""
        if bc.verify(bytes(self.job.requester), bytes.fromhex(msg["sig"]),
                     wire.taken_digest(self.job.job_id)):
            self.taken = True
        await wire.send_msg(link, {"t": "ok"})


class Node:
    def __init__(self, where: Path, cfg: dict, chain, wallet, runtime):
        self.where, self.cfg, self.host_cfg = where, cfg, cfg["host"]
        self.chain, self.wallet, self.runtime = chain, wallet, runtime
        self.authority = wallet.pubkey()
        self.boxes: dict[bytes, JobBox] = {}
        self.sessions: dict[Pubkey, asyncio.Task] = {}
        self.tasks: set[asyncio.Task] = set()
        self.environment: asyncio.subprocess.Process | None = None
        self.env_contact: dict = {}
        self.settler = Settler(chain, wallet, where / "vouchers", log)
        self.min_grace = self.host_cfg.get("min_grace_s", MIN_GRACE_S)
        self.retain_s = self.host_cfg.get("retain_s", RETAIN_S)
        self.beat = 0.0
        self.provider, self.tier = self._evidence_provider()

    def _evidence_provider(self):
        image = bytes.fromhex(bc.code_measurement(sandbox_code_files()))
        if self.host_cfg.get("evidence") == "tpm":
            return (TpmEvidence(self.host_cfg["attester"], self.host_cfg.get("tpm_container", "beetle-host-tpm"),
                                image, log), TIERS["t3b"])
        return NoEvidence(), TIERS["t3c"]

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self._ended)
        return task

    def _ended(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        if not task.cancelled() and task.exception():
            log(f"task failed: {task.exception()!r}")

    async def call(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    async def run(self) -> None:
        self.peer = await wire.bound_peer(self.cfg, self.host_cfg.get("port", 0), services={echo.ALPN: echo.answer},
                                          secret=wire.secret(self.where, "node.key"))
        public = self.host_cfg.get("public_addr")
        self.addrs = list(dict.fromkeys(([public] if public else []) + await wire.candidates(self.peer, self.cfg)))
        if not self.host_cfg.get("infer_only"):
            await self.call(self.runtime.prepare)
        await self.call(self.refresh_offers)
        self.restore_retained()
        log(f"seller {self.authority} on {self.chain.cfg['cluster']}, endpoint {self.peer.id[:12]}, "
            f"addresses {self.addrs}")
        while True:
            try:
                await self.tick()
            except Exception as e:  # RPC nodes and backends come and go; the next tick retries
                log(f"tick failed: {e!r}"[:200])
            await asyncio.sleep(POLL)

    async def tick(self) -> None:
        jobs = await self.call(self.chain.jobs_for_host, self.authority)
        for job in jobs:
            if job.state == "Opened" and job.job_id not in self.boxes:
                box = JobBox(job, self.where / "host_jobs" / job.job_id.hex()[:16],
                             vet=lambda spec, job=job: spec_problem(spec, job.terms))
                box.prepare()
                self.boxes[job.job_id] = box
                self.spawn(self.run_job(box))
        await self.call(self.collect, jobs)
        self.sweep(jobs)
        if self.host_cfg.get("backend"):
            await self.serve_sessions()
        if time.time() - self.beat > HEARTBEAT:
            self.beat = time.time()
            await self.call(self.heartbeat)

    def collect(self, jobs: list[Job]) -> None:
        """Whatever the chain owes this seller, whoever's turn it was."""
        chain, kp, now = self.chain, self.wallet, self.chain.chain_time()
        for job in jobs:
            try:
                if job.state == "Delivered" and now >= job.challenge_end + self._accept_wait(job):
                    log(f"job {job.job_id.hex()[:16]} finalize {chain.finalize(kp, job)[:16]}")
                elif job.state == "Started" and now > job.release_by and not self._busy(job):
                    chain.claim_no_release(kp, job)
                    log(f"job {job.job_id.hex()[:16]}: the buyer never released the key")
                elif job.state in FINISHED:
                    chain.claim_due(kp, job.job_id)
            except RuntimeError as e:
                log(f"job {job.job_id.hex()[:16]}: not settled yet: {str(e)[:90]}")

    def _accept_wait(self, job: Job) -> int:
        """Only a buyer's accept pays the full price off T2, so give it time. A job
        that didn't pass pays the same either way: don't hold its slot."""
        if job.tier == TIER_T2 or job.outcome != OUTCOME_PASSED:
            return 0
        return self.host_cfg.get("accept_wait_s", ACCEPT_WAIT)

    def _busy(self, job: Job) -> bool:
        box = self.boxes.get(job.job_id)
        return box is not None and not box.finished

    def sweep(self, jobs: list[Job]) -> None:
        """Delete-after-use, with room for a buyer who was away: once a job is over
        its inputs go at once, and its result when the buyer has it or the window ends."""
        live = {j.job_id: j for j in jobs}
        for job_id, box in list(self.boxes.items()):
            job = live.get(job_id)
            if not box.finished or (job is not None and job.state not in FINISHED):
                continue
            if box.taken or not box.submitted or (box.keep_until and time.time() > box.keep_until):
                if box.listener:
                    box.listener.cancel()
                shutil.rmtree(box.root, ignore_errors=True)
                del self.boxes[job_id]
                log(f"job {job_id.hex()[:16]} over, its files are gone; "
                    f"balance {self.chain.token_balance(self.authority):.2f} USDC")
            elif box.keep_until is None:
                box.keep_until = time.time() + self.retain_s
                box.release_inputs()
                box.save()
                self._listen_after_close(box)
                log(f"job {job_id.hex()[:16]} settled; result kept for the buyer for {self.retain_s} s")

    def _listen_after_close(self, box: JobBox) -> None:
        if box.listener:
            box.listener.cancel()
        rendezvous = wire.closed_job_rendezvous(self.cfg, box.job.job_id, "seller", self.chain, self.wallet,
                                                box.job.requester, where=self.where)
        box.listener = self.spawn(self.listen(rendezvous, box))

    def restore_retained(self) -> None:
        """Results outlive this process, as buyers may come back after a restart."""
        for marker in sorted((self.where / "host_jobs").glob("*/retained.json")):
            box = JobBox.retained(marker.parent)
            job = self.chain.get_job(box.job.job_id)
            self.boxes[box.job.job_id] = box
            if job is None or job.state in FINISHED:
                self._listen_after_close(box)
            else:
                box.job = job
                rendezvous = wire.job_rendezvous(self.cfg, job.job_id, "seller", self.chain, self.wallet, self.where)
                box.listener = self.spawn(self.listen(rendezvous, box))

    def refresh_offers(self) -> None:
        """A new version of the sandbox attests something else, and an offer that
        still names the old one would have every buyer refuse it."""
        current = measurement(self.host_cfg)
        for offer in offers_of(self.chain, self.authority):
            if offer.status == OFFER_ACTIVE and offer.measurement != current:
                self.chain.update_offer(self.wallet, offer, terms_of(offer, measurement=current))
                log(f"offer {str(offer.address)[:12]} now names this sandbox")

    def heartbeat(self) -> None:
        until = self.chain.chain_time() + OFFER_LIFETIME
        for offer in offers_of(self.chain, self.authority):
            if offer.status == OFFER_ACTIVE:
                self.chain.touch_offer(self.wallet, offer, until)

    # --- training ---------------------------------------------------------------

    async def listen(self, rendezvous, box: JobBox) -> None:
        """A buyer coming back is just another contact."""
        while True:
            them = await rendezvous.wait()
            box.contacted = True
            await rendezvous.publish({"id": self.peer.id, "addrs": self.addrs})
            self.spawn(self._attend(them, box.handle))

    async def _attend(self, them: dict, handler) -> None:
        try:
            link = await self.peer.link(them["id"], them["addrs"], initiator=False, window=DIAL_WINDOW)
            await handler(link)
        except Exception as e:  # links end; the buyer reconnects if it still needs us
            log(f"link to {them.get('id', '?')[:8]} ended: {e!r}"[:120])

    async def run_job(self, box: JobBox) -> None:
        job, chain, kp = box.job, self.chain, self.wallet
        tag, name = job.job_id.hex()[:16], f"beetle-{job.job_id.hex()[:12]}"
        log(f"job {tag}: {job.price / 1e6:.2f} USDC in escrow, {job.bond / 1e6:.2f} of stake locked")
        rendezvous = wire.job_rendezvous(self.cfg, job.job_id, "seller", chain, kp, self.where)
        box.listener = self.spawn(self.listen(rendezvous, box))
        launched = False
        try:
            if job.verifier != NO_VERIFIER and str(job.verifier) not in self.host_cfg.get("verifiers", []):
                log(f"job {tag}: names a verifier this seller doesn't use; declining")
                await self.call(chain.decline_job, kp, job)
                return
            if not await self._inputs_arrive(box):
                log(f"job {tag}: the buyer's inputs never came; declining, which costs no stake")
                await self.call(chain.decline_job, kp, job)
                return
            if box.refused:
                log(f"job {tag}: {box.refused}; declining")
                await self.call(chain.decline_job, kp, job)
                return
            try:
                model = await self.call(self._model_for, box.spec)
            except CannotServe as e:
                log(f"job {tag}: declining: {e}")
                await self.call(chain.decline_job, kp, job)
                return
            log(f"job {tag}: inputs received, ciphertext only")
            launched = True
            await self.call(self.runtime.launch, name, model, box.inbox, box.outbox)
            attestation = await self.call(self.runtime.attested, name, box.outbox, ATTEST_TIMEOUT)
            if attestation is None:
                log(f"job {tag}: sandbox did not attest; declining")
                await self.call(chain.decline_job, kp, job)
                return
            evidence = await self.call(self.provider.evidence, box.outbox,
                                       report_data_for(job.job_id, job.spec_hash, attestation))
            box.evidence = evidence
            box.attested.set()  # first, so the release window isn't spent on this
            registration = await self.call(self._registration, job, attestation, evidence)
            await self.call(chain.start, kp, job, bytes.fromhex(attestation["sign_pk"]),
                            bytes.fromhex(attestation["enc_pk"]), job.measurement,
                            evidence_hash(evidence), registration)
            log(f"job {tag}: started, {evidence['type']} evidence committed")

            job = await self.call(chain.get_job, job.job_id)
            job = await self.call(chain.wait_job, job.job_id, lambda j: j.state != "Started",
                                  job.release_by - await self.call(chain.chain_time) + 5)
            if job is None or not job.has_dek:
                log(f"job {tag}: no key released ({job.state if job else 'closed'})")
                return
            tmp = box.inbox / ".sealed_dek.tmp"
            tmp.write_bytes(job.sealed_dek)
            tmp.rename(box.inbox / "sealed_dek.bin")
            log(f"job {tag}: sealed key handed to the sandbox, training")

            left = job.deadline - await self.call(chain.chain_time) - SUBMIT_MARGIN
            await self.call(self.runtime.wait, name, left)
            report_path = box.outbox / "report.json"
            if not report_path.exists():
                error = wire.read_json(box.outbox / "error.json", {"reason": "no report"})
                log(f"job {tag}: sandbox stopped without a report: {error['reason']}; aborting")
                await self.call(chain.abort_job, kp, job)
                return
            report = json.loads(report_path.read_text())
            box.sealed = [p for p in [box.outbox / "adapter.enc"] if p.exists()]
            if box.sealed:
                wait = min(self.host_cfg.get("delivery_wait_s", DELIVERY_WAIT),
                           job.deadline - await self.call(chain.chain_time) - SUBMIT_MARGIN)
                try:
                    await asyncio.wait_for(box.acked.wait(), max(wait, 1))
                    log(f"job {tag}: buyer holds the ciphertext")
                except TimeoutError:
                    log(f"job {tag}: buyer did not take the ciphertext; submitting anyway")
            sig = (box.outbox / "settle.sig").read_text()
            await self.call(chain.submit_result, kp, job, report,
                            bc.sha256(bc.canonical_json(report)), bytes.fromhex(sig))
            box.save()
            box.submitted = True  # only now can the report go out
            log(f"job {tag}: submitted, outcome {report['outcome']}, work_bps {report['work_bps']}")
        except TimeoutError:
            log(f"job {tag}: gave up waiting for the buyer")
        except RuntimeError as e:  # a transaction refused; the chain's timeouts end the job
            log(f"job {tag}: stopped: {str(e)[:120]}")
        finally:
            box.finished = True
            if launched:
                logs = await self.call(self.runtime.stop, name)
                (box.root / "sandbox.log").write_text(logs)

    async def _inputs_arrive(self, box: JobBox) -> bool:
        """Declining is free until `start_by`, and silence after it is slashed, so that
        is the limit. A buyer who never showed up isn't worth holding a slot for."""
        job, chain = box.job, self.chain
        patience = min(job.opened_at + NO_SHOW_S, job.start_by) - DECLINE_MARGIN
        try:
            await asyncio.wait_for(box.inputs.wait(), max(patience - await self.call(chain.chain_time), 1))
            return True
        except TimeoutError:
            if not box.contacted:
                return False
        left = job.start_by - await self.call(chain.chain_time) - DECLINE_MARGIN
        try:
            await asyncio.wait_for(box.inputs.wait(), max(left, 1))
            return True
        except TimeoutError:
            return False

    def _model_for(self, spec: dict) -> Path:
        base = spec["base_model"]
        if base["id"] not in {mp.get(m).repo for m in self.host_cfg["models"]}:
            raise CannotServe(f"this seller does not offer {base['id']}")
        models = self.where / "models"
        where = model_dir(base["id"], models)
        if not where.exists() or bc.dir_manifest_sha256(where) != base["manifest_sha256"]:
            got = ensure_model(base["revision"], base["id"], models)
            if got["manifest_sha256"] != base["manifest_sha256"]:
                raise CannotServe("the download differs from the manifest in the spec")
        return where

    def _registration(self, job: Job, attestation: dict, evidence: dict):
        """A job that names a third-party verifier needs its signature on chain."""
        if job.verifier == NO_VERIFIER:
            return None
        reg = register(self.host_cfg["attester"], job.job_id, job.spec_hash, attestation,
                       bytes(job.host), evidence)
        return (bytes.fromhex(reg["attester_pk"]), bytes.fromhex(reg["signature"]),
                Registration.from_json(reg).message(), reg["expiry"])

    # --- inference ----------------------------------------------------------------

    def _backend(self) -> Backend:
        return Backend(self.host_cfg["backend"], self.host_cfg.get("backend_key"))

    def environment_cmd(self) -> list[str]:
        return [sys.executable, "-m", "node.environment"]

    async def _start_environment(self) -> bool:
        """Starts the process that sees plaintext, once the backend answers."""
        try:
            await self._backend().models()
        except (OSError, httpx.HTTPError) as e:
            log(f"inference backend not up: {e!r}"[:120])
            return False
        proc = await asyncio.create_subprocess_exec(
            *self.environment_cmd(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            env={**os.environ, "BEETLE_HOME": str(self.where), "BEETLE_PAYEE": str(self.authority)},
            cwd=Path(__file__).resolve().parents[1])
        try:
            ready = json.loads(await asyncio.wait_for(proc.stdout.readline(), ENV_START))
        except (TimeoutError, ValueError):
            proc.kill()
            log("inference environment did not start")
            return False
        self.environment = proc
        self.env_contact = {k: ready[k] for k in ("id", "addrs", "enc_pk")}
        self.spawn(self._hear_environment(proc))
        log(f"inference environment up: {ready['models']}, adapters {'on' if ready['adapters'] is True else 'off: ' + str(ready['adapters'])}")
        return True

    async def _hear_environment(self, proc) -> None:
        while line := await proc.stdout.readline():
            msg = json.loads(line)
            if msg["t"] == "voucher":
                await self.call(self.settler.record, msg)
            elif msg["t"] == "ended":
                await self.call(self.settler.finish, msg["channel"])
        log("inference environment stopped")
        self.environment = None
        for task in self.sessions.values():
            task.cancel()
        self.sessions.clear()

    async def relay(self, rendezvous) -> None:
        """Contacts go through here, with the wallet; the link goes to the environment."""
        while True:
            them = await rendezvous.wait()
            await rendezvous.publish(self.env_contact)
            self.environment.stdin.write(json.dumps(
                {"t": "contact", "them": {"id": them["id"], "addrs": them["addrs"]}}).encode() + b"\n")
            await self.environment.stdin.drain()

    async def serve_sessions(self) -> None:
        if self.environment is None and not await self._start_environment():
            return
        channels = await self.call(self.chain.channels_for, self.authority)
        await self.call(self.settler.watch, channels)
        live = {c.address: c for c in channels if c.status == "Open" and c.grace_period >= self.min_grace}
        for address, channel in live.items():
            if address not in self.sessions:
                rendezvous = wire.seller_session(self.cfg, channel, self.chain, self.wallet, self.where)
                self.sessions[address] = self.spawn(self.relay(rendezvous))
                log(f"channel {str(address)[:12]} opened, {channel.deposit / 1e6:.2f} USDC deposited")
        for address in [a for a in self.sessions if a not in live]:
            self.sessions.pop(address).cancel()


async def serve(where: Path, cfg: dict, chain, wallet, runtime=None) -> None:
    host = cfg["host"]
    await Node(where, cfg, chain, wallet, runtime or DockerRuntime(host.get("gpu", False))).run()
