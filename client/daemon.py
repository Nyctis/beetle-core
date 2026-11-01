"""`beetle serve`: the one process that holds the wallet, the policy and the jobs.

It binds 127.0.0.1 only and every call needs a local token. The command line, the
SDK and the MCP server are clients of this API, and a Mac app can be one too.

    /v1/chat/completions, /v1/models      OpenAI compatible
    /beetle/*                             market, jobs, models, proof, wallet, doctor

Two scopes. `owner` is the token in this directory, for the person at this
machine. `agent` tokens can spend within the policy, and that is all: there is
no call that raises a limit, lowers a floor or relabels data, and only `owner`
can approve what needs a person.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import socket
import time
import traceback
import uuid
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

import model_profiles as mp
from client import doctor as doc
from client.core import Core
from client.errors import BeetleError, bad_request
from node import wire

DEFAULT_PORT = 8741
SWEEP_EVERY = 30.0
KEEP_ALIVE = 120  # seconds, longer than any client polls: closing under a POST fails it, and URLSession won't retry one
API_VERSION = 1  # raised with each route a client comes to need, so it can tell an old daemon


def load_tokens(home: Path) -> dict:
    path = home / "tokens.json"
    tokens = wire.read_json(path, {})
    if not any(t["scope"] == "owner" for t in tokens.values()):
        tokens[f"bk_local_{secrets.token_hex(16)}"] = {"name": "owner", "scope": "owner"}
        wire.write_json(path, tokens)
    return tokens


def owner_token(home: Path) -> str:
    return next(k for k, t in load_tokens(home).items() if t["scope"] == "owner")


def add_token(home: Path, name: str) -> str:
    tokens = load_tokens(home)
    token = f"bk_local_{secrets.token_hex(16)}"
    tokens[token] = {"name": name, "scope": "agent"}
    wire.write_json(home / "tokens.json", tokens)
    return token


class Guard(BaseHTTPMiddleware):
    """Refuses other hosts, so a web page can't reach this through DNS rebinding."""

    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "").rsplit(":", 1)[0]
        if host not in ("127.0.0.1", "localhost"):
            return JSONResponse({"error": {"code": "forbidden", "message": "local only"}}, 403)
        tokens = load_tokens(request.app.state.core.home)
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or token not in tokens:
            return JSONResponse({"error": {"code": "unauthorized", "message": "need a local token"}}, 401)
        request.state.scope_name = tokens[token]["scope"]
        request.state.client_name = tokens[token]["name"]
        try:
            return await call_next(request)
        except BeetleError as e:
            return JSONResponse(e.body(), e.status)


def api(fn):
    """A handler returns JSON, or raises a BeetleError that becomes its status."""
    async def handler(request: Request):
        try:
            return await fn(request, request.app.state.core)
        except BeetleError as e:
            return JSONResponse(e.body(), e.status)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": {"code": "internal", "message": repr(e)[:300]}}, 500)
    return handler


async def body_of(request: Request) -> dict:
    try:
        return await request.json()
    except ValueError:
        raise bad_request("the body must be JSON") from None


def chunk(cid: str, model: str, delta: dict, finish: str | None = None, **extra) -> str:
    return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                                  "model": model, "choices": [{"index": 0, "delta": delta,
                                                               "finish_reason": finish}], **extra},
                                 ensure_ascii=False) + "\n\n"


async def chat_completions(request: Request, core: Core):
    body = await body_of(request)
    if not body.get("messages") or not body.get("model"):
        raise bad_request("model and messages are required")
    params = {k: body[k] for k in ("max_tokens", "temperature", "top_p", "seed") if k in body}
    params.setdefault("temperature", 0.7)
    route = await core.open_chat(body["model"], body["messages"], params, body.get("beetle") or {})
    cid, model = f"chatcmpl-{uuid.uuid4().hex}", body["model"]
    headers = route.headers()

    def usage() -> dict:
        i, o = route.usage.get("in_tokens", 0), route.usage.get("out_tokens", 0)
        return {"prompt_tokens": i, "completion_tokens": o, "total_tokens": i + o}

    if not body.get("stream"):
        text = "".join([p async for p in route.stream()])
        headers["x-beetle-cost-usd"] = f"{route.cost_usd():.6f}"
        return JSONResponse({"id": cid, "object": "chat.completion", "created": int(time.time()),
                             "model": model, "usage": usage(), "choices": [{
                                 "index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}]}, headers=headers)

    async def events():
        yield chunk(cid, model, {"role": "assistant", "content": ""})
        async for piece in route.stream():
            yield chunk(cid, model, {"content": piece})
        # A streamed response's headers go out before the cost is known.
        yield chunk(cid, model, {}, "stop", usage=usage(), beetle={"cost_usd": round(route.cost_usd(), 6)})
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers=headers)


async def models(request: Request, core: Core):
    named = [{"id": alias, "object": "model", "owned_by": "beetle", "base": e["versions"][e["current"]]["base"]}
             for alias, e in core.models.listing().items()]
    public = [{"id": name, "object": "model", "owned_by": "beetle", "base": name} for name in mp.load()]
    return JSONResponse({"object": "list", "data": named + public})


async def doctor(request: Request, core: Core):
    caps = await asyncio.to_thread(doc.machine, core.home, core.cfg)
    out = doc.report(caps)
    out["network"] = await asyncio.to_thread(wire.nat_type, core.cfg)
    nodes = await asyncio.to_thread(core.echo_nodes)
    out["network"]["advice"] = wire.nat_advice(out["network"], len(nodes))
    policy = core.policy.load()
    out["policy"] = {k: policy.get(k) for k in ("per_job_usd", "per_day_usd", "min_tier")}
    if core.chain is not None:
        try:
            market = await asyncio.to_thread(core.market)
            floor = policy.get("min_tier", "T0")
            for name, entry in out["models"].items():
                rows = [r for r in market if r["model"] == name]
                entry["sellers"] = {"train": sum(r["kind"] == "train" for r in rows),
                                    "infer": sum(r["kind"] == "infer" for r in rows),
                                    "tiers": sorted({r["tier"] for r in rows})}
            out["floor"] = floor
            out["wallet"] = await asyncio.to_thread(core.wallet_view)
            out["wallet"].pop("entries")
        except OSError as e:
            out["chain_error"] = str(e)
    return JSONResponse(out)


async def market(request: Request, core: Core):
    q = request.query_params
    return JSONResponse(await asyncio.to_thread(core.market, q.get("kind"), q.get("model"), q.get("min_tier")))


async def quote(request: Request, core: Core):
    b = await body_of(request)
    if not b.get("dataset") or not b.get("base"):
        raise bad_request("a quote needs dataset and base")
    out = await asyncio.to_thread(core.quote, b["dataset"], b["base"], b.get("method", "memory"),
                                  b.get("min_tier"), b.get("max_usd"), b.get("seller"))
    out.pop("_offer")
    return JSONResponse(out)


async def jobs(request: Request, core: Core):
    if request.method == "GET":
        return JSONResponse([core.view(j) for j in core.jobs.values()])
    b = await body_of(request)
    human = request.state.scope_name == "owner" and b.get("approved", False)
    out = await core.train(b.get("dataset"), b.get("base", "auto"), b.get("min_tier"), b.get("max_usd"),
                           b.get("idempotency_key"), b.get("method", "memory"), b.get("seller"),
                           b.get("alias"), human=human)
    return JSONResponse(out, 201)


async def job(request: Request, core: Core):
    return JSONResponse(await core.job_status(request.path_params["job"]))


async def fetch(request: Request, core: Core):
    b = await body_of(request) if await request.body() else {}
    return JSONResponse(await core.fetch(request.path_params["job"], b.get("alias")))


async def reject(request: Request, core: Core):
    return JSONResponse(await core.reject(request.path_params["job"]))


async def cancel(request: Request, core: Core):
    return JSONResponse(await core.cancel(request.path_params["job"]))


async def proof(request: Request, core: Core):
    return JSONResponse(core.proof(request.path_params["proof"]))


async def model_library(request: Request, core: Core):
    return JSONResponse(core.models.listing())


async def rollback(request: Request, core: Core):
    return JSONResponse(core.models.rollback(request.path_params["alias"]))


async def datasets(request: Request, core: Core):
    return JSONResponse(core.datasets.listing())


async def wallet(request: Request, core: Core):
    return JSONResponse(await asyncio.to_thread(core.wallet_view))


async def chain_jobs(request: Request, core: Core):
    return JSONResponse(await asyncio.to_thread(core.open_jobs))


async def connection(request: Request, core: Core):
    nat = await asyncio.to_thread(wire.nat_type, core.cfg)
    live = [{"model": k[0], "adapter": k[1], "tier": s.tier, "channel": str(s.channel),
             "path": wire.path_of(s.session.pipe.link) if s.session.pipe.link else None,
             "links_lost": s.session.links_lost} for k, s in core.sessions.items()]
    return JSONResponse({"network": nat, "sessions": live,
                         "jobs": {j.id: j.rec.get("path") for j in core.jobs.values()}})


async def close_sessions(request: Request, core: Core):
    """End every inference session now rather than when it has been idle a while."""
    closing = len(core.sessions)
    await core.close_idle(0)
    return JSONResponse({"closed": closing})


async def limits(request: Request, core: Core):
    """Read the limits, or ask a person to change them. Nothing here changes one."""
    policy = core.policy.load()
    if request.method == "GET":
        return JSONResponse({k: policy.get(k) for k in ("per_job_usd", "per_day_usd", "min_tier",
                                                        "auto_approve_under_usd")})
    b = await body_of(request)
    ticket = core.approvals.request({"why": "limit change", "ask": b})
    return JSONResponse({"needs_approval": ticket, "how": "a person runs: beetle policy set ..."}, 202)


async def approvals(request: Request, core: Core):
    if request.state.scope_name != "owner":
        raise BeetleError(403, "owner_only", "only the machine's owner can approve")
    ticket = request.path_params.get("ticket")
    if ticket:
        return JSONResponse(core.approvals.approve(ticket))
    return JSONResponse(core.approvals.listing())


async def health(request: Request, core: Core):
    return JSONResponse({"ok": True, "pid": os.getpid(), "api": API_VERSION, "chain": core.cfg.get("chain")})


def build(core: Core) -> Starlette:
    routes = [("/v1/chat/completions", chat_completions, ["POST"]), ("/v1/models", models, ["GET"]),
              ("/beetle/health", health, ["GET"]), ("/beetle/doctor", doctor, ["GET"]),
              ("/beetle/market", market, ["GET"]), ("/beetle/quote", quote, ["POST"]),
              ("/beetle/jobs", jobs, ["GET", "POST"]), ("/beetle/jobs/{job}", job, ["GET"]),
              ("/beetle/jobs/{job}/fetch", fetch, ["POST"]), ("/beetle/jobs/{job}/cancel", cancel, ["POST"]),
              ("/beetle/jobs/{job}/reject", reject, ["POST"]),
              ("/beetle/proof/{proof}", proof, ["GET"]), ("/beetle/models", model_library, ["GET"]),
              ("/beetle/models/{alias}/rollback", rollback, ["POST"]),
              ("/beetle/datasets", datasets, ["GET"]), ("/beetle/wallet", wallet, ["GET"]),
              ("/beetle/connection", connection, ["GET"]), ("/beetle/chain/jobs", chain_jobs, ["GET"]), ("/beetle/sessions/close", close_sessions, ["POST"]),
              ("/beetle/limits", limits, ["GET", "POST"]),
              ("/beetle/approvals", approvals, ["GET"]), ("/beetle/approvals/{ticket}", approvals, ["POST"])]
    app = Starlette(routes=[Route(p, api(fn), methods=m) for p, fn, m in routes])
    app.add_middleware(Guard)
    app.state.core = core
    return app


async def upkeep(core: Core) -> None:
    while True:
        await asyncio.sleep(SWEEP_EVERY)
        try:
            await core.close_idle()
            if core.chain is not None:
                await asyncio.to_thread(core.sweep_channels)
        except (OSError, RuntimeError) as e:  # the next round retries
            print(f"upkeep: {e}", flush=True)


def usable_port(port: int | None) -> int:
    """The usual port, or another if a second home runs on this machine. A port
    someone asked for is theirs to get or to fail on."""
    if port is not None:
        return port
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", DEFAULT_PORT))
            return DEFAULT_PORT
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]


def listening(app, port: int) -> uvicorn.Config:
    return uvicorn.Config(app, host="127.0.0.1", port=port, loop="asyncio", log_level="warning",
                          timeout_keep_alive=KEEP_ALIVE)


async def serve(port: int | None = None) -> None:
    core = Core()
    port = usable_port(port)
    server = uvicorn.Server(listening(build(core), port))
    load_tokens(core.home)
    wire.write_json(core.home / "serve.json", {"port": port, "pid": os.getpid()})
    core.resume()
    task = asyncio.create_task(upkeep(core))
    try:
        await server.serve()
    finally:
        task.cancel()
