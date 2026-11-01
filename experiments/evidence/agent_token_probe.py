"""Tries everything an agent token must not be able to do, on a home made for this.

The home needs no funds: every refusal comes before money moves. It is set up with

    beetle init --chain chain.json --per-job 0.5 --per-day 1 --min-tier public --auto-under 0.1
    beetle data add demo --name ds_syn --label synthetic
    beetle data add demo --name ds_personal --label personal

    BEETLE_HOME=... BEETLE_TOKEN=$(beetle token probe) python agent_token_probe.py --out OUT.json

Each case records what was sent, what came back and whether the home's files
changed. `pass` means the token was refused as the README says, or nothing leaked.
Nothing here prints a secret; it only reports whether one appeared in a response.

`--reuse` is the one test that needs money. A job that never opens gives its approval
back, so an empty wallet cannot show whether an approval is used once. This opens one
real job, pays for it and settles it, on a home with test funds:

    beetle init --chain chain.json --per-job 5 --per-day 10 --min-tier public --auto-under 0.1
    beetle data add demo --name ds_syn --label synthetic

    BEETLE_HOME=... BEETLE_TOKEN=$(beetle token probe) python agent_token_probe.py --reuse --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "common"), str(ROOT), str(Path(__file__).parent)]

from agent_state import secret_forms  # noqa: E402

HOME = Path(os.environ["BEETLE_HOME"])
TOKEN = os.environ["BEETLE_TOKEN"]
PORT = json.loads((HOME / "serve.json").read_text())["port"]
WATCHED = ("policy.json", "approvals.json", "tokens.json", "datasets.json", "ledger.json", "wallet.json", "config.json")
GET_ROUTES = ("/beetle/health", "/beetle/doctor", "/beetle/market", "/beetle/jobs", "/beetle/models", "/beetle/datasets",
              "/beetle/wallet", "/beetle/connection", "/beetle/chain/jobs", "/beetle/limits", "/v1/models")
QWEN = "qwen3.5-2b"


def code_state() -> dict:
    """Which code answered: a daemon started now runs what is on disk now."""
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": head, "at": int(time.time()),
            "sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                       for f in ("client/policy.py", "client/core.py", "client/mcp_server.py", "client/daemon.py")}}


def fingerprint() -> dict:
    return {name: hashlib.sha256((HOME / name).read_bytes()).hexdigest() if (HOME / name).exists() else None
            for name in WATCHED}


def secrets_of() -> dict[str, bytes]:
    return {k: v for k, v in secret_forms(HOME).items() if v != TOKEN.encode()}


class Probe:
    def __init__(self):
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", trust_env=False, timeout=120,
                                 headers={"Authorization": f"Bearer {TOKEN}"})
        self.secrets = secrets_of()
        self.cases: list[dict] = []

    def call(self, group: str, name: str, method: str, path: str, want: set[int], body: dict | None = None,
             want_code: str | None = None, headers: dict | None = None) -> dict:
        before = fingerprint()
        r = self.http.request(method, path, json=body, headers=headers)
        after = fingerprint()
        changed = [k for k in WATCHED if before[k] != after[k]]
        leaked = [k for k, v in self.secrets.items() if v in r.content]
        try:
            reply = r.json()
        except ValueError:
            reply = {"text": r.text[:200]}
        code = reply.get("error", {}).get("code") if isinstance(reply, dict) else None
        allowed_change = {"approvals.json"} if r.status_code in (202, 403) else set()
        ok = r.status_code in want and not leaked and not set(changed) - allowed_change \
            and (want_code is None or code == want_code)
        case = {"group": group, "case": name, "request": f"{method} {path}", "body": body, "status": r.status_code,
                "code": code, "changed": changed, "leaked": leaked, "pass": ok}
        self.cases.append(case)
        print(f"{'PASS' if ok else 'FAIL'} {group}/{name}: {r.status_code} {code or ''} changed={changed}", flush=True)
        return reply if isinstance(reply, dict) else {}

    def job(self, dataset: str, **extra) -> dict:
        return {"dataset": dataset, "base": QWEN, "min_tier": "public", "max_usd": 1000,
                "idempotency_key": uuid.uuid4().hex, **extra}


def beetle(*args: str) -> None:
    subprocess.run(["beetle", *args], check=True, capture_output=True)


def run(p: Probe) -> None:
    # --- the budget --------------------------------------------------------------------
    p.call("budget", "raise per-job and per-day", "POST", "/beetle/limits", {202},
           {"per_job_usd": 1000, "per_day_usd": 1000, "reason": "need more"})
    p.call("budget", "raise the automatic amount", "POST", "/beetle/limits", {202}, {"auto_approve_under_usd": 1000})
    p.call("budget", "no route sets the policy", "POST", "/beetle/policy", {404, 405}, {"per_job_usd": 1000})
    p.call("budget", "no PUT on limits", "PUT", "/beetle/limits", {404, 405}, {"per_job_usd": 1000})
    p.call("budget", "max_usd over the per-job limit", "POST", "/beetle/jobs", {402}, p.job("ds_syn"),
           want_code="budget_exceeded")
    p.call("budget", "chat cannot raise a limit", "GET", "/beetle/limits", {200})

    # --- the approval ------------------------------------------------------------------
    beetle("policy", "set", "--per-job", "5", "--per-day", "10")  # a person: so the job reaches the approval step
    first = p.call("approval", "job over the automatic amount", "POST", "/beetle/jobs", {403}, p.job("ds_syn"),
                   want_code="approval_required")
    ticket = first.get("error", {}).get("approval")
    p.call("approval", "approved:true in the body", "POST", "/beetle/jobs", {403}, p.job("ds_syn", approved=True),
           want_code="approval_required")
    p.call("approval", "scope and human in the body", "POST", "/beetle/jobs", {403},
           p.job("ds_syn", approved=True, human=True, scope="owner"), want_code="approval_required")
    p.call("approval", "list tickets", "GET", "/beetle/approvals", {403}, want_code="owner_only")
    p.call("approval", "approve its own ticket", "POST", f"/beetle/approvals/{ticket}", {403}, want_code="owner_only")
    p.call("approval", "owner header claim", "POST", f"/beetle/approvals/{ticket}", {403},
           headers={"X-Beetle-Scope": "owner"}, want_code="owner_only")
    p.call("approval", "a limits ticket shaped like a job's", "POST", "/beetle/limits", {202},
           {"dataset": "ds_syn", "base": QWEN, "usd": 1.0, "why": "over the automatic amount"})
    p.call("approval", "job still refused after that", "POST", "/beetle/jobs", {403}, p.job("ds_syn"),
           want_code="approval_required")

    # --- the floor ---------------------------------------------------------------------
    p.call("floor", "personal data, request says public", "POST", "/beetle/quote", {409},
           {"dataset": "ds_personal", "base": QWEN, "min_tier": "public"}, want_code="no_route_meets_floor")
    p.call("floor", "personal data, job says public", "POST", "/beetle/jobs", {409}, p.job("ds_personal"),
           want_code="no_route_meets_floor")
    beetle("policy", "set", "--min-tier", "T2")  # a person raises the floor
    p.call("floor", "ask to lower the floor", "POST", "/beetle/limits", {202}, {"min_tier": "public"})
    p.call("floor", "quote says public", "POST", "/beetle/quote", {409},
           {"dataset": "ds_syn", "base": QWEN, "min_tier": "public"}, want_code="no_route_meets_floor")
    p.call("floor", "job says public", "POST", "/beetle/jobs", {409}, p.job("ds_syn"),
           want_code="no_route_meets_floor")
    p.call("floor", "chat says public, remote", "POST", "/v1/chat/completions", {409},
           {"model": QWEN, "messages": [{"role": "user", "content": "hi"}],
            "beetle": {"min_tier": "public", "route": "remote"}}, want_code="no_route_meets_floor")
    p.call("floor", "no route relabels data", "POST", "/beetle/datasets", {404, 405},
           {"name": "ds_personal", "label": "public"})

    # --- the keys ----------------------------------------------------------------------
    for route in GET_ROUTES:
        p.call("keys", f"read {route}", "GET", route, {200})
    wallet = str(HOME / "wallet.json")
    p.call("keys", "dataset is a relative path", "POST", "/beetle/quote", {404},
           {"dataset": "../wallet.json", "base": QWEN}, want_code="not_found")
    p.call("keys", "dataset is the wallet's path", "POST", "/beetle/quote", {404},
           {"dataset": wallet, "base": QWEN}, want_code="not_found")
    p.call("keys", "model is a path", "POST", "/v1/chat/completions", {404},
           {"model": "../wallet", "messages": [{"role": "user", "content": "hi"}]}, want_code="not_found")
    for path in ("/beetle/proof/prf_..%2F..%2Fwallet", "/beetle/proof/..%2Fwallet", "/beetle/proof/prf_..%2Fwallet",
                 "/beetle/jobs/..%2F..%2Fwallet", "/wallet.json", "/tokens.json", "/beetle/tokens", "/beetle/keys",
                 "/beetle/wallet/export", "/beetle/models/..%2Fwallet/rollback"):
        p.call("keys", f"path {path}", "POST" if path.endswith("rollback") else "GET", path, {404, 405})

    # --- who is asking -----------------------------------------------------------------
    plain = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", trust_env=False, timeout=30)
    for name, headers, want in (("no token", {}, 401), ("wrong token", {"Authorization": "Bearer bk_local_00"}, 401),
                                ("other host", {"Authorization": f"Bearer {TOKEN}", "Host": "evil.example"}, 403)):
        r = plain.get("/beetle/limits", headers=headers)
        p.cases.append({"group": "auth", "case": name, "request": "GET /beetle/limits", "body": None,
                        "status": r.status_code, "code": r.json()["error"]["code"], "changed": [], "leaked": [],
                        "pass": r.status_code == want})
        print(f"{'PASS' if r.status_code == want else 'FAIL'} auth/{name}: {r.status_code}", flush=True)

    # --- what an approval is worth on an empty wallet ---------------------------------
    # Not a token test: a person approves once, then the token asks twice. Past the
    # gate the job fails on this empty wallet, so nothing is spent. A job that never
    # opened keeps its approval, so this says nothing about reuse: see --reuse.
    beetle("policy", "set", "--min-tier", "public")
    beetle("approve", ticket)
    notes = []
    for i in (1, 2):
        r = p.http.post("/beetle/jobs", json=p.job("ds_syn"))
        reply = r.json()
        notes.append({"try": i, "status": r.status_code, "code": reply.get("error", {}).get("code") if "error" in reply else None,
                      "message": str(reply.get("error", {}).get("message"))[:160]})
        print(f"note empty wallet {i}: {r.status_code} {notes[-1]['code']}", flush=True)
    p.cases.append({"group": "note", "case": "one approval on an empty wallet, no job opens", "request": "POST /beetle/jobs",
                    "body": None, "status": None, "code": None, "changed": [], "leaked": [], "pass": None,
                    "detail": notes})


def reuse(a) -> int:
    """One approval, used twice, with a real job so the approval is really spent.

    An approval used to stay good for ever. The check for the same data and model
    within 24 hours stops the second job first, so a person has to approve that too:
    then the first approval is all that stands between the agent and a second paid job.
    """
    p = Probe()
    owner_key = next(k for k, v in json.loads((HOME / "tokens.json").read_text()).items() if v["scope"] == "owner")
    owner = httpx.Client(base_url=f"http://127.0.0.1:{PORT}", trust_env=False, timeout=None,
                         headers={"Authorization": f"Bearer {owner_key}"})
    steps: list[dict] = []
    report = {"test": "one approval, used twice", "code": code_state(), "steps": steps}

    def ask(name: str, want: int) -> dict:
        r = p.http.post("/beetle/jobs", json=p.job(a.dataset, max_usd=3))
        body = r.json()
        error = body.get("error") or {}  # a job's view has an "error" field too, null while all is well
        row = {"step": name, "status": r.status_code, "code": error.get("code"), "ticket": error.get("approval"),
               "message": str(error.get("message"))[:160] if error else None, "job": body.get("job"), "as_expected": r.status_code == want}
        steps.append(row)
        print(f"{name}: {r.status_code} {row['code'] or ''}", flush=True)
        return row

    def finish(verdict: str, why: str) -> int:
        report.update(verdict=verdict, why=why)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(verdict, why, flush=True)
        return 0

    first = ask("first request, over the automatic amount", 403)
    if first["code"] != "approval_required":
        return finish("inconclusive", f"the first request got {first['status']} {first['code']}, not a ticket")
    beetle("approve", first["ticket"])
    opened = ask("same request after one approval", 201)
    if opened["status"] != 201:
        return finish("inconclusive", f"no job opened: {opened['status']} {opened['code']} {opened['message']}")
    repeat = ask("same data and model again", 403)
    if repeat["code"] != "approval_required":
        return finish("inconclusive", f"expected the repeat check, got {repeat['status']} {repeat['code']}")
    beetle("approve", repeat["ticket"])
    third = ask("one approval, used twice: the repeat is approved, the amount is not", 403)
    third["as_expected"] = third["status"] == 403 and third["code"] == "approval_required" and third["ticket"] not in (first["ticket"], repeat["ticket"])
    again = owner.post(f"/beetle/approvals/{first['ticket']}")
    report["approve_used_ticket"] = {"status": again.status_code, "code": again.json().get("error", {}).get("code")}
    jobs = owner.get("/beetle/jobs").json()
    report["jobs_opened"] = len(jobs)
    ident, end = opened["job"], time.time() + 600
    view = owner.get(f"/beetle/jobs/{ident}").json()
    while view["stage"] not in ("delivered", "settled", "failed") and time.time() < end:
        time.sleep(3)
        view = owner.get(f"/beetle/jobs/{ident}").json()
    if view["stage"] == "delivered":
        fetched = owner.post(f"/beetle/jobs/{ident}/fetch", json={"alias": "reuse-retest"}).json()
        view = owner.get(f"/beetle/jobs/{ident}").json()
        report["first_job_paid_usd"] = (fetched.get("result") or {}).get("paid_usd")
    report["first_job_stage"] = view["stage"]
    held = third["as_expected"] and report["approve_used_ticket"]["status"] == 400 and len(jobs) == 1
    return finish("fixed" if held else "not_fixed", f"second paid job: {third['status']} {third['code']}; "
                  f"approving the used ticket: {report['approve_used_ticket']['status']}; jobs opened: {len(jobs)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reuse", action="store_true", help="the one test that spends: see above")
    ap.add_argument("--dataset", default="ds_syn", help="for --reuse")
    a = ap.parse_args()
    if a.reuse:
        return reuse(a)
    p = Probe()
    start = fingerprint()
    run(p)
    graded = [c for c in p.cases if c["pass"] is not None]
    report = {"passed": sum(c["pass"] for c in graded), "total": len(graded),
              "leaks": sorted({k for c in p.cases for k in c["leaked"]}),
              "files_changed_overall": [k for k, v in fingerprint().items() if start[k] != v],
              "code": code_state(), "cases": p.cases}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("passed", "total", "leaks", "files_changed_overall")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
