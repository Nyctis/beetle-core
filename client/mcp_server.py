"""An MCP server for agents: newline-delimited JSON-RPC on stdio.

    BEETLE_TOKEN=$(beetle token my-agent) python -m client.mcp_server

It runs with an agent's token, so it can do what the policy already allows and
nothing more. The tools talk to the local daemon; this process holds no keys and
no wallet. `spending_limit` can read the limits or ask a person to change them,
and no tool can change one. A dataset is a registered name, never a path, and a
tool that fetches an adapter returns numbers, never its weights.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

from client.api import Api, ApiError

# The versions that have an `initialize` handshake, newest first. 2026-07-28 has none.
PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
_S, _N = {"type": "string"}, {"type": "number"}


def tool(name: str, description: str, required: list[str], **props) -> dict:
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": props, "required": required}}


TOOLS = [
    tool("doctor", "This machine's capability and the path it suggests. Read-only.", []),
    tool("list_datasets", "Registered datasets: names, labels, floors, sizes. Never contents.", []),
    tool("quote_job", "Price, time and tier for a job. Spends nothing.", ["dataset", "base"],
         dataset=_S, base=_S, method=_S, min_tier=_S, max_usd=_N),
    tool("train_adapter", "Start a job. It returns at once, and a job usually takes 2 to 5 minutes. "
         "Both policy arguments and an idempotency key are required. "
         "At tier T3-C the seller's owner can read the data.",
         ["dataset", "base", "min_tier", "max_usd", "idempotency_key"],
         dataset=_S, base=_S, method=_S, min_tier=_S, max_usd=_N, idempotency_key=_S, seller=_S),
    tool("job_status", "Stage, spending and evidence summary of a job. A job stays in each stage for a while: "
         "`escrowed` can take a minute while the two machines connect. Set wait_s, up to 60, to get an "
         "answer when the stage changes.", ["job_id"], job_id=_S, wait_s=_N),
    tool("fetch_adapter", "Verify a job once its stage is `delivered`, store its adapter here, check that it "
         "changes answers, settle. Returns scores, not weights.", ["job_id"], job_id=_S, alias=_S),
    tool("chat", "Through the local gateway. Requests can tighten the policy, never loosen it. "
         "At tier T3-C the seller's owner can read what is sent.",
         ["messages", "model"], messages={"type": "array"}, model=_S, min_tier=_S, max_usd=_N, route=_S),
    tool("spending_limit", "Read the limits, or ask a person to change them. Cannot change them.",
         ["action"], action={"enum": ["get", "request_change"]}, per_job_usd=_N, per_day_usd=_N, reason=_S),
    tool("proof", "What a job or a response proved, in plain words. A running job has no evidence yet.",
         ["id"], id=_S),
    tool("cancel_job", "Stop a job: refunded before the key is out, by the chain's rules after. "
         "A slow stage is normal and is no reason to cancel.", ["job_id"], job_id=_S),
]


def part(ident) -> str:
    """An id goes into a path, so it has to stay one piece of it."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(ident)):
        raise KeyError(f"not an id: {ident!r}")
    return str(ident)


def job_status(api: Api, ident: str, wait_s) -> dict:
    """An agent has no clock to sleep on, so the wait happens here."""
    if not isinstance(wait_s, (int, float, type(None))):
        raise KeyError(f"not a number of seconds: {wait_s!r}")
    path = f"/beetle/jobs/{ident}"
    first = now = api.get(path)
    end = time.monotonic() + min(float(wait_s or 0), 60)
    while time.monotonic() < end and now["stage"] == first["stage"] \
            and now["stage"] not in ("delivered", "settled", "failed"):
        time.sleep(2)
        now = api.get(path)
    return now


def run_tool(api: Api, name: str, a: dict):
    if name == "doctor":
        return api.get("/beetle/doctor")
    if name == "list_datasets":
        return api.get("/beetle/datasets")
    if name == "quote_job":
        return api.post("/beetle/quote", a)
    if name == "train_adapter":
        return api.post("/beetle/jobs", a)
    if name == "job_status":
        return job_status(api, part(a["job_id"]), a.get("wait_s"))
    if name == "fetch_adapter":
        return api.post(f"/beetle/jobs/{part(a['job_id'])}/fetch", {"alias": a.get("alias")})
    if name == "proof":
        return api.get(f"/beetle/proof/{part(a['id'])}")
    if name == "cancel_job":
        return api.post(f"/beetle/jobs/{part(a['job_id'])}/cancel")
    if name == "spending_limit":
        if a["action"] == "get":
            return api.get("/beetle/limits")
        return api.post("/beetle/limits", {k: v for k, v in a.items() if k != "action"})
    if name == "chat":
        text, headers, final = [], {}, {}
        options = {k: a[k] for k in ("min_tier", "max_usd", "route") if a.get(k) is not None}
        for meta, piece in api.chat(a["model"], a["messages"], **options):
            if meta:
                headers = meta
            elif isinstance(piece, str):
                text.append(piece)
            else:
                final = piece
        return {"text": "".join(text), "route": headers.get("x-beetle-route"),
                "tier": headers.get("x-beetle-tier"), "proof": headers.get("x-beetle-proof"),
                "cost_usd": final.get("beetle", {}).get("cost_usd")}
    raise KeyError(name)


def handle(api: Api, msg: dict) -> dict | None:
    method, ident = msg.get("method"), msg.get("id")
    if ident is None:  # a notification
        return None
    if method == "initialize":
        asked = (msg.get("params") or {}).get("protocolVersion")
        result = {"protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "beetle", "version": "0.0.1"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg["params"]
        try:
            out, failed = run_tool(api, params["name"], params.get("arguments") or {}), False
        except ApiError as e:
            out, failed = {"status": e.status, **e.error}, True
        except KeyError as e:
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32602, "message": f"bad tool call: {e}"}}
        result = {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}], "isError": failed}
    else:
        return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def main() -> None:
    token = os.environ.get("BEETLE_TOKEN")
    if not token:
        raise SystemExit("set BEETLE_TOKEN to an agent token from `beetle token NAME`")
    api = Api(token)
    for line in sys.stdin:
        if line.strip():
            reply = handle(api, json.loads(line))
            if reply is not None:
                print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
