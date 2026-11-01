#!/usr/bin/env python3
"""ScienceClaw skill: a thin, stdlib-only client of the local `beetle serve` API.

    BEETLE_TOKEN=$(beetle token scienceclaw) python3 beetle_skill.py --action doctor

ScienceClaw runs this as a subprocess under a 30 second limit and reads JSON from
stdout. So nothing here waits for a job, and what it prints is what the agent may
publish: stage, tier, verdict and hashes, never data, scores or weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PUBLIC_JOB = ("job", "job_id", "stage", "base", "dataset", "floor", "usd", "offer", "alias", "chain_state", "next_job")
PUBLIC_RESULT = ("passed", "outcome", "work_bps")
PUBLIC_PROOF = ("tier", "headline", "detail", "on_the_way", "on_their_machine", "path")
NEEDS = {"quote": ("dataset",), "train": ("dataset", "min_tier", "max_usd"), "status": ("job",), "fetch": ("job",),
         "proof": ("job",), "chat": ("message",)}


class Daemon:
    def __init__(self, timeout: float):
        self.token = os.environ.get("BEETLE_TOKEN", "")
        self.base = os.environ.get("BEETLE_URL") or self._local_url()
        self.timeout = timeout

    @staticmethod
    def _local_url() -> str:
        state = Path(os.environ.get("BEETLE_HOME", "~/.beetle")).expanduser() / "serve.json"
        port = json.loads(state.read_text())["port"] if state.exists() else 8741
        return f"http://127.0.0.1:{port}"

    def call(self, method: str, path: str, body: dict | None = None, **query) -> dict:
        query = {k: v for k, v in query.items() if v is not None}
        url = self.base + path + (f"?{urllib.parse.urlencode(query)}" if query else "")
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy
        try:
            with opener.open(request, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise Refused(e.code, json.loads(e.read() or b"{}").get("error", {"message": str(e)})) from None
        except TimeoutError:
            raise Refused(0, {"code": "still_running", "message":
                              "the daemon is still working; ask again with the same arguments"}) from None
        except urllib.error.URLError as e:
            raise Refused(0, {"code": "no_daemon", "message": f"beetle serve is not reachable: {e.reason}"}) from None


class Refused(Exception):
    def __init__(self, status: int, error: dict):
        super().__init__(error.get("message", ""))
        self.status, self.error = status, error


def key_of(a) -> str:
    """Stable across retries the same day, so an agent's retry can't pay twice."""
    day = time.strftime("%Y-%m-%d", time.gmtime())
    return hashlib.sha256(f"{a.dataset}|{a.base}|{a.method}|{a.min_tier}|{day}".encode()).hexdigest()[:32]


def job_card(view: dict) -> dict:
    card = {k: view.get(k) for k in PUBLIC_JOB if view.get(k) is not None}
    if view.get("result"):
        card["result"] = {k: view["result"].get(k) for k in PUBLIC_RESULT}
    if view.get("error"):
        card["error"] = view["error"]
    return card


def run(a, d: Daemon) -> dict:
    policy = {"min_tier": a.min_tier, "max_usd": a.max_usd}
    if a.action == "doctor":
        return d.call("GET", "/beetle/doctor")
    if a.action == "datasets":
        return {"results": d.call("GET", "/beetle/datasets")}
    if a.action == "market":
        return {"results": d.call("GET", "/beetle/market", model=None if a.base == "auto" else a.base,
                                  min_tier=a.min_tier)}
    if a.action == "limits":
        return d.call("GET", "/beetle/limits")
    if a.action == "quote":
        return d.call("POST", "/beetle/quote", {"dataset": a.dataset, "base": a.base, "method": a.method, **policy})
    if a.action == "train":
        key = a.idempotency_key or key_of(a)
        view = d.call("POST", "/beetle/jobs", {"dataset": a.dataset, "base": a.base, "method": a.method,
                                               "idempotency_key": key, **policy})
        return {"idempotency_key": key, **job_card(view)}
    if a.action == "status":
        return job_card(d.call("GET", f"/beetle/jobs/{a.job}"))
    if a.action == "fetch":
        return job_card(d.call("POST", f"/beetle/jobs/{a.job}/fetch", {"alias": a.alias}))
    if a.action == "proof":
        card = d.call("GET", f"/beetle/proof/{a.job}")
        return {k: card.get(k) for k in PUBLIC_PROOF}
    reply = d.call("POST", "/v1/chat/completions", {
        "model": a.model, "messages": [{"role": "user", "content": a.message}],
        "max_tokens": a.max_tokens, "temperature": a.temperature,
        "beetle": {k: v for k, v in policy.items() if v is not None}})
    return {"text": reply["choices"][0]["message"]["content"], "usage": reply.get("usage")}


def summary(action: str, out: dict) -> str:
    if "error" in out:
        return f"beetle {action}: {out['error'].get('code')}: {out['error'].get('message')}"
    if action in ("train", "status", "fetch"):
        return f"job {out.get('job')} is {out.get('stage')} at floor {out.get('floor')}, ${out.get('usd')}"
    if action == "quote":
        return f"${out.get('usd')} on {out.get('base')}, seller tier {out.get('tier')}, floor {out.get('floor')}"
    if action == "chat":
        return out["text"]
    return f"beetle {action}: ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", default="doctor", choices=["doctor", "datasets", "market", "quote", "train", "status",
                                                          "fetch", "proof", "limits", "chat"])
    ap.add_argument("--dataset")
    ap.add_argument("--base", default="auto")
    ap.add_argument("--method", default="memory")
    ap.add_argument("--min-tier", choices=["T0", "T1", "T2", "T3-B", "public"])
    ap.add_argument("--max-usd", type=float)
    ap.add_argument("--idempotency-key")
    ap.add_argument("--job")
    ap.add_argument("--alias", default="me")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--message")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--format", default="json", choices=["json", "summary"])
    ap.add_argument("--query", help="ScienceClaw's planners always pass this; only --message is ever sent")
    a = ap.parse_args()

    missing = [f"--{n.replace('_', '-')}" for n in NEEDS.get(a.action, ()) if getattr(a, n) is None]
    if missing:
        out = {"error": {"code": "bad_request", "message": f"{a.action} needs {', '.join(missing)}"}}
    else:
        try:
            out = run(a, Daemon(a.timeout))
        except Refused as e:
            out = {"status": e.status, "error": e.error}
    out = {"summary": summary(a.action, out), **out}
    print(json.dumps(out, ensure_ascii=False) if a.format == "json" else out["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
