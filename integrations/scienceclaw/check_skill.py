"""Check the skill and the OpenAI path against a running `beetle serve`.

Runs the skill the way ScienceClaw's SkillExecutor does (parameters become flags,
a 30 second subprocess, JSON on stdout) and calls /v1 with the same arguments
ScienceClaw's `_call_openai` sends. ScienceClaw itself is not run here.

    BEETLE_HOME=... BEETLE_TOKEN=... python integrations/scienceclaw/check_skill.py [--chat MODEL] [--dataset NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SKILL = Path(__file__).resolve().parent / "skills/beetle/scripts/beetle_skill.py"
ENDPOINTS = [("GET", "/v1/models"), ("GET", "/beetle/doctor"), ("GET", "/beetle/datasets"), ("GET", "/beetle/limits")]


def flags(parameters: dict) -> list[str]:
    """SkillExecutor._execute_script_skill's translation."""
    out = []
    for key, value in parameters.items():
        flag = "--" + key.lstrip("-").replace("_", "-")
        if value is True:
            out.append(flag)
        elif value not in (None, False):
            out += [flag, str(value)]
    return out


def skill(**parameters) -> dict:
    r = subprocess.run([sys.executable, str(SKILL), *flags(parameters)], capture_output=True, text=True, timeout=30,
                       cwd="/")
    if r.returncode:
        raise RuntimeError(r.stderr)
    return json.loads(r.stdout)


def base_url() -> str:
    state = Path(os.environ.get("BEETLE_HOME", "~/.beetle")).expanduser() / "serve.json"
    return f"http://127.0.0.1:{json.loads(state.read_text())['port']}"


def http(method: str, path: str, body: dict | None = None, token: str | None = None) -> tuple[int, dict, dict]:
    request = urllib.request.Request(base_url() + path, method=method, data=json.dumps(body).encode() if body else None,
                                     headers={"Authorization": f"Bearer {token or os.environ['BEETLE_TOKEN']}",
                                              "Content-Type": "application/json"})
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=120) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


def report(name: str, ok: bool, detail: object = "") -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", help="model for a paid or local completion; skipped when absent")
    ap.add_argument("--dataset")
    ap.add_argument("--base", default="qwen3.5-2b")
    a = ap.parse_args()
    good = []
    for method, path in ENDPOINTS:
        status, _, _ = http(method, path)
        good.append(report(f"{method} {path}", status == 200, status))
    good.append(report("wrong token is refused", http("GET", "/beetle/limits", token="bk_local_wrong")[0] == 401))
    status, body, _ = http("POST", "/beetle/limits", {"per_job_usd": 1000, "reason": "agent wants more"})
    good.append(report("an agent can only ask for a higher limit", status == 202 and "needs_approval" in body, status))
    status, body, _ = http("POST", "/beetle/approvals/anything")
    good.append(report("an agent cannot approve", status == 403, body.get("error", {}).get("code")))

    for action in ("doctor", "datasets", "limits"):
        out = skill(action=action, format="json", query="ignored by design")
        good.append(report(f"skill {action}", "error" not in out, out["summary"]))
    out = skill(action="train", dataset="x", format="json")
    good.append(report("train without policy is refused before any call", out["error"]["code"] == "bad_request",
                       out["error"]["message"]))
    if a.dataset:
        out = skill(action="quote", dataset=a.dataset, base=a.base, min_tier="public", max_usd=2)
        good.append(report("skill quote", "usd" in out or "error" in out, out["summary"]))
    if a.chat:
        status, body, headers = http("POST", "/v1/chat/completions", {
            "model": a.chat, "temperature": 0.2, "max_tokens": 32,
            "messages": [{"role": "user", "content": "Reply with the word ready."}]})
        lower = {k.lower(): v for k, v in headers.items()}
        good.append(report("ScienceClaw's OpenAI call", status == 200, {
            "text": body.get("choices", [{}])[0].get("message", {}).get("content", body)[:80],
            **{k: lower.get(k) for k in ("x-beetle-route", "x-beetle-tier", "x-beetle-cost-usd")}}))
        out = skill(action="chat", model=a.chat, message="Reply with the word ready.", timeout=28)
        good.append(report("skill chat", "error" not in out, out["summary"][:80]))
    print(f"{sum(good)}/{len(good)} checks passed")
    return int(not all(good))


if __name__ == "__main__":
    sys.exit(main())
