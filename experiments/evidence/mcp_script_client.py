"""A scripted client of the real MCP server, over stdio. Not an agent.

This fixes the calls in advance, so it decides nothing and says nothing about agents.
It shows the server's tools work end to end with an agent token, and it runs the
same refusals through MCP that agent_token_probe.py runs through HTTP.

    BEETLE_HOME=... BEETLE_TOKEN=... python mcp_script_client.py loop --dataset NAME --out OUT.jsonl
    BEETLE_HOME=... BEETLE_TOKEN=... python mcp_script_client.py refuse --out OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "qwen3.5-2b"
# An id must stay one piece of a URL path. Each of these once reached another route.
PATH_IDS = (("proof", {"id": "../wallet"}), ("job_status", {"job_id": "../limits"}), ("job_status", {"job_id": "../approvals"}),
            ("fetch_adapter", {"job_id": "../../limits"}), ("cancel_job", {"job_id": "../x"}))


class Client:
    def __init__(self, out: Path):
        self.proc = subprocess.Popen([sys.executable, "-m", "client.mcp_server"], cwd=ROOT, text=True, bufsize=1,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.n, self.t0, self.rows = 0, time.time(), []
        self.out = out

    def rpc(self, method: str, params: dict | None = None) -> dict:
        self.n += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def call(self, tool: str, **args):
        t = time.time()
        reply = self.rpc("tools/call", {"name": tool, "arguments": args})
        if "error" in reply:
            output, failed = reply["error"], True
        else:
            output, failed = json.loads(reply["result"]["content"][0]["text"]), reply["result"]["isError"]
        self.rows.append({"event": "tool_call", "tool": tool, "input": args, "is_error": failed, "output": output,
                          "s": round(time.time() - t, 2), "at_s": round(t - self.t0, 1)})
        print(f"{round(t - self.t0):5}s {tool} {'ERR' if failed else 'ok'} {str(output)[:120]}", flush=True)
        return output

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait(timeout=10)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows))


def loop(c: Client, dataset: str) -> None:
    c.call("doctor")
    c.call("list_datasets")
    c.call("spending_limit", action="get")
    quote = c.call("quote_job", dataset=dataset, base=BASE, min_tier="public", max_usd=1.5)
    view = c.call("train_adapter", dataset=dataset, base=BASE, min_tier="public", max_usd=1.5,
                  idempotency_key=uuid.uuid4().hex)
    job = view["job"]
    print("offer", quote.get("offer", "")[:12], flush=True)
    while True:
        status = c.call("job_status", job_id=job)
        if status["stage"] in ("delivered", "settled", "failed"):
            break
        time.sleep(10)
    if status["stage"] == "delivered":
        c.call("fetch_adapter", job_id=job, alias="agent-rehearsal")
        c.call("job_status", job_id=job)
    c.call("proof", id=job)
    c.call("chat", model="agent-rehearsal", messages=[{"role": "user", "content": "What is my dentist's name?"}])


def refuse(c: Client) -> None:
    c.call("spending_limit", action="request_change", per_job_usd=1000, per_day_usd=1000, reason="rehearsal")
    c.call("spending_limit", action="request_change", min_tier="public")
    c.call("spending_limit", action="set", per_job_usd=1000)
    c.call("spending_limit", action="get")
    c.call("train_adapter", dataset="ds_personal", base=BASE, min_tier="public", max_usd=1000,
           idempotency_key=uuid.uuid4().hex)
    c.call("train_adapter", dataset="ds_syn", base=BASE, min_tier="public", max_usd=1000,
           idempotency_key=uuid.uuid4().hex, approved=True)
    c.call("quote_job", dataset="../wallet.json", base=BASE)
    c.call("proof", id="prf_../../wallet")
    for tool, args in PATH_IDS:
        c.call(tool, **args)
    c.call("list_datasets")
    print(c.rpc("tools/call", {"name": "approve", "arguments": {"ticket": "x"}}), flush=True)
    print(c.rpc("tools/call", {"name": "export_wallet", "arguments": {}}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["loop", "refuse"])
    ap.add_argument("--dataset")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert os.environ.get("BEETLE_TOKEN"), "set BEETLE_TOKEN to an agent token"
    c = Client(a.out)
    print(c.rpc("initialize", {"protocolVersion": "2025-06-18"})["result"]["serverInfo"], flush=True)
    c.rows.append({"event": "tools", "tools": [t["name"] for t in c.rpc("tools/list")["result"]["tools"]]})
    try:
        loop(c, a.dataset) if a.mode == "loop" else refuse(c)
    finally:
        c.close()


if __name__ == "__main__":
    main()
