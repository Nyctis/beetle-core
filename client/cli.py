"""`beetle`: the buyer's commands, and `beetle host` for a seller.

Setup commands (init, policy, data, token) edit this machine's files directly:
that is a person at their own machine. Everything else goes through the daemon.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

from client import datasets as ds
from client.api import Api, ApiError
from client.daemon import DEFAULT_PORT, add_token, serve
from client.policy import FLOORS, LABEL_FLOOR, Policy
from node import cli as host_cli
from node import wire

USDC = 1_000_000


def show(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def cmd_init(a) -> int:
    where = wire.home()
    where.mkdir(parents=True, exist_ok=True)
    cfg = wire.read_json(where / "config.json", {})
    if a.local_only:
        Policy(where / "policy.json").save(per_job_usd=0, per_day_usd=0, min_tier="T0")
        cfg.setdefault("chain", None)
    else:
        if a.per_job is None or a.per_day is None or a.min_tier is None:
            raise SystemExit("beetle init needs --per-job, --per-day and --min-tier, or --local-only")
        if a.chain:
            shutil.copy(a.chain, where / "chain.json")
        if not (where / "chain.json").exists():
            raise SystemExit("beetle init needs --chain chain.json, the network to join")
        cfg["chain"] = "chain.json"
        kp = wire.new_wallet(where)
        Policy(where / "policy.json").save(per_job_usd=a.per_job, per_day_usd=a.per_day, min_tier=a.min_tier,
                                           auto_approve_under_usd=a.auto_under)
        print(f"wallet {kp.pubkey()}")
    host_cli.rendezvous_section(a, where, cfg)
    if a.stun:
        cfg["stun"] = a.stun
    if a.candidate:
        cfg["candidates"] = a.candidate
    cfg["lan"] = a.lan
    cfg["traffic"] = a.traffic
    if a.grace_s:
        cfg["grace_s"] = a.grace_s
    if a.local_backend:
        cfg["local_backend"] = {"url": a.local_backend, "models": dict(
            (m.split("=") + m.split("="))[:2] for m in a.local_model)}
    if not (a.local_only or cfg.get("stun") or cfg.get("candidates")):
        print("note: with no --stun or --candidate, a seller behind its own firewall can't reach this machine")
    wire.write_json(where / "config.json", cfg)
    print(f"config {where / 'config.json'}")
    return 0


def cmd_doctor(a) -> int:
    report = Api().get("/beetle/doctor")
    if a.json:
        show(report)
        return 0
    m = report["machine"]
    print(f"chip     {m['chip']}")
    print(f"memory   {m['memory_gb']} GB, GPU budget {m['gpu_budget_gb']} GB, free now {m['free_gb']} GB")
    print(f"disk     {m['disk_gb']} GB free")
    print(f"system   {m['os']}, on power: {m['on_power']}")
    print(f"network  {report['network']['nat']}: {report['network'].get('advice', '')}")
    if "wallet" in report:
        w = report["wallet"]
        print(f"wallet   {w['address']} {w['usdc']:.2f} USDC, {w['sol']:.3f} SOL")
    print(f"policy   {report['policy']}")
    print()
    names = {"local": "all on this machine", "hybrid": "train remotely, run here", "remote": "all remote"}
    for name, entry in report["models"].items():
        sellers = entry.get("sellers", {})
        print(f"{name:16} {names[entry['path']]:26} {entry['why']}; sellers: "
              f"{sellers.get('train', 0)} train, {sellers.get('infer', 0)} infer {sellers.get('tiers', [])}")
    return 0


def cmd_policy(a) -> int:
    policy = Policy(wire.home() / "policy.json")
    if a.action == "set":
        policy.save(per_job_usd=a.per_job, per_day_usd=a.per_day, min_tier=a.min_tier,
                    auto_approve_under_usd=a.auto_under)
    show(policy.load())
    return 0


def cmd_data(a) -> int:
    registry = ds.Datasets(wire.home() / "datasets.json")
    if a.action == "add":
        show(registry.add(a.path, a.name, a.label, a.eval))
    else:
        show(registry.listing())
    return 0


def cmd_market(a) -> int:
    rows = Api().get("/beetle/market", kind=a.kind, model=a.model, min_tier=a.min_tier)
    if a.json:
        show(rows)
        return 0
    for r in rows:
        prices = " ".join(f"{k} ${v:.4f}/Mtok" for k, v in r["usd_per_mtok"].items())
        print(f"{r['offer'][:12]}  {r['kind']:5} {r['model']:16} {r['tier']:5} {prices}  "
              f"stake ${r['stake_usd']:.0f}  done {r['settled']} failed {r['failed']}  "
              f"their machine: {r['on_their_machine']}")
    if not rows:
        print("no offers match")
    return 0


def cmd_quote(a) -> int:
    show(Api().post("/beetle/quote", {"dataset": a.dataset, "base": a.base, "method": a.method,
                                      "min_tier": a.min_tier, "max_usd": a.max_usd, "seller": a.seller}))
    return 0


def confirmed(error: ApiError) -> bool:
    print(f"{error}: {error.error.get('why', '')}")
    if not sys.stdin.isatty():
        print(error.error.get("how", ""))
        return False
    return input("go ahead? [y/N] ").strip().lower() == "y"


def follow(api: Api, job: str, until: tuple[str, ...]) -> dict:
    seen = 0
    while True:
        view = api.get(f"/beetle/jobs/{job}")
        for step in view["timeline"][seen:]:
            print(f"{time.strftime('%H:%M:%S', time.localtime(step['at']))} {step['stage']:10} {step['note']}",
                  flush=True)
        seen = len(view["timeline"])
        if view.get("next_job"):
            print(f"next seller: job {view['next_job']}", flush=True)
            job, seen = view["next_job"], 0
        elif view["stage"] in until:
            return view
        time.sleep(2)


def cmd_train(a) -> int:
    api = Api()
    body = {"dataset": a.dataset, "base": a.base, "method": a.method, "min_tier": a.min_tier,
            "max_usd": a.max_usd, "seller": a.seller, "idempotency_key": a.idempotency_key or uuid.uuid4().hex}
    try:
        view = api.post("/beetle/jobs", body)
    except ApiError as e:
        if e.status != 403 or not confirmed(e):
            raise
        view = api.post("/beetle/jobs", {**body, "approved": True})
    print(f"job {view['job']}: ${view['usd']:.4f} at floor {view['floor']}")
    if a.detach:
        return 0
    view = follow(api, view["job"], ("delivered", "settled", "failed"))
    if view["stage"] == "failed":
        show(view.get("error") or {"ended": view.get("ended")})
        return 1
    print(f"delivered; next: beetle fetch {view['job']} --as me")
    return 0


def cmd_status(a) -> int:
    api = Api()
    show(api.get(f"/beetle/jobs/{a.job}") if a.job else api.get("/beetle/jobs"))
    return 0


def cmd_fetch(a) -> int:
    show(Api().post(f"/beetle/jobs/{a.job}/fetch", {"alias": getattr(a, "as")}))
    return 0


def cmd_reject(a) -> int:
    show(Api().post(f"/beetle/jobs/{a.job}/reject"))
    return 0


def cmd_cancel(a) -> int:
    show(Api().post(f"/beetle/jobs/{a.job}/cancel"))
    return 0


def cmd_proof(a) -> int:
    proof = Api().get(f"/beetle/proof/{a.job}")
    if a.export:
        wire.write_json(Path(a.export), proof)
    if a.json:
        show(proof)
        return 0
    print(f"{proof['headline']}\n{proof['detail']}\n")
    for key in ("seller", "transport", "environment"):
        if proof["peer"].get(key):
            print(f"{key:12} {proof['peer'][key]}")
    print(f"{'tier':12} {proof['tier']}")
    print(f"{'evidence':12} {proof['evidence']}")
    print(f"{'path':12} {proof['path']}")
    print(f"\non the way: {proof['on_the_way']}    on their machine: {proof['on_their_machine']}")
    return 0


def run_chat(api: Api, a, messages: list[dict]) -> str:
    text, headers = [], {}
    for meta, piece in api.chat(a.model, messages, route=a.route, min_tier=a.min_tier, max_usd=a.max_usd,
                                seller=a.seller, max_tokens=a.max_tokens, temperature=a.temperature):
        if meta:
            headers = meta
        elif isinstance(piece, str):
            text.append(piece)
            print(piece, end="", flush=True)
        else:
            print(f"\n[{headers.get('x-beetle-route')} {headers.get('x-beetle-tier')} "
                  f"${piece['beetle']['cost_usd']:.6f} {piece['usage']['total_tokens']} tokens "
                  f"{headers.get('x-beetle-proof')}]", file=sys.stderr)
    return "".join(text)


def cmd_chat(a) -> int:
    api, messages = Api(), []
    if a.message:
        run_chat(api, a, [{"role": "user", "content": a.message}])
        return 0
    while True:
        try:
            line = input("> ")
        except EOFError:
            return 0
        messages.append({"role": "user", "content": line})
        messages.append({"role": "assistant", "content": run_chat(api, a, messages)})


def cmd_serve(a) -> int:
    print(f"local API, token in {wire.home() / 'tokens.json'}, port in {wire.home() / 'serve.json'}", flush=True)
    try:
        asyncio.run(serve(a.port))
    except KeyboardInterrupt:
        pass
    return 0


def cmd_approve(a) -> int:
    api = Api()
    show(api.post(f"/beetle/approvals/{a.ticket}") if a.ticket else api.get("/beetle/approvals"))
    return 0


def cmd_token(a) -> int:
    print(add_token(wire.home(), a.name))
    return 0


def cmd_wallet(a) -> int:
    if a.action == "show":
        show(Api().get("/beetle/wallet"))
        return 0
    where = wire.home()
    chain = wire.load_chain(wire.load_config(where), where)
    if chain.is_mainnet():
        raise SystemExit("this mints test tokens: not on mainnet")
    from beetle_chain import load_keypair

    kp = wire.load_wallet(where)
    chain.airdrop(kp.pubkey())
    chain.mint_test_usdc(load_keypair(a.mint_authority), kp.pubkey(), int(a.usdc * USDC))
    print(f"{kp.pubkey()}: {chain.token_balance(kp.pubkey()):.2f} USDC")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="beetle", description="Training and inference, peer to peer.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="the two required policies, wallet, network")
    p.add_argument("--chain", type=Path, help="chain.json of the network")
    p.add_argument("--per-job", type=float, help="most one job may cost, USD")
    p.add_argument("--per-day", type=float, help="most a day may cost, USD")
    p.add_argument("--min-tier", choices=FLOORS, help="privacy floor")
    p.add_argument("--auto-under", type=float, default=1.0, help="above this a person confirms")
    p.add_argument("--local-only", action="store_true", help="nothing leaves this machine")
    p.add_argument("--rendezvous-dir", help="a directory shared with sellers, in place of the chain")
    p.add_argument("--stun", action="append", help="host:port of a STUN server; it learns this IP")
    p.add_argument("--candidate", action="append", help="public ip:port of this machine, if known")
    p.add_argument("--lan", action="store_true", help="also tell sellers this machine's local addresses")
    p.add_argument("--grace-s", type=int, help="how long a seller has to settle once this buyer asks to close")
    p.add_argument("--local-backend", help="an OpenAI-compatible server on this machine, for local inference")
    p.add_argument("--local-model", action="append", default=[], metavar="NAME[=SERVED]",
                   help="a model that server runs; repeat")
    p.add_argument("--traffic", choices=["constant", "burst"], default="constant")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("doctor", help="what this machine can do, and the path that follows")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("policy", help="show or change the limits")
    p.add_argument("action", nargs="?", choices=["show", "set"], default="show")
    p.add_argument("--per-job", type=float)
    p.add_argument("--per-day", type=float)
    p.add_argument("--min-tier", choices=FLOORS)
    p.add_argument("--auto-under", type=float)
    p.set_defaults(fn=cmd_policy)

    p = sub.add_parser("data", help="register a dataset under a name and a label")
    p.add_argument("action", choices=["add", "list"])
    p.add_argument("path", nargs="?", type=Path)
    p.add_argument("--name")
    p.add_argument("--label", choices=sorted(LABEL_FLOOR), default="personal")
    p.add_argument("--eval", type=Path, help="hidden questions, kept for acceptance")
    p.set_defaults(fn=cmd_data)

    p = sub.add_parser("market", help="what is on offer")
    p.add_argument("--kind", choices=["train", "infer"])
    p.add_argument("--model")
    p.add_argument("--min-tier", choices=FLOORS)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_market)

    def job_arguments(p):
        p.add_argument("dataset")
        p.add_argument("--base", default="auto")
        p.add_argument("--method", default="memory")
        p.add_argument("--min-tier", choices=FLOORS, help="can only tighten the policy")
        p.add_argument("--max-usd", type=float)
        p.add_argument("--seller", help="an offer address")

    p = sub.add_parser("quote", help="what a job would cost; spends nothing")
    job_arguments(p)
    p.set_defaults(fn=cmd_quote)

    p = sub.add_parser("train", help="escrow, verify, release, train")
    job_arguments(p)
    p.add_argument("--idempotency-key")
    p.add_argument("--detach", action="store_true", help="return once it is in escrow")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("status", help="one job, or all")
    p.add_argument("job", nargs="?")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("fetch", help="verify the result, open the adapter, check it works, settle")
    p.add_argument("job")
    p.add_argument("--as", default="me", help="name in the model library")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("reject", help="refuse a delivery: off T2 this settles at the floor")
    p.add_argument("job")
    p.set_defaults(fn=cmd_reject)

    p = sub.add_parser("cancel", help="stop a job: refunded before the key is out")
    p.add_argument("job")
    p.set_defaults(fn=cmd_cancel)

    p = sub.add_parser("proof", help="what a job or a response proved")
    p.add_argument("job", help="job id or prf_ id")
    p.add_argument("--export", help="write the evidence summary to this file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_proof)

    p = sub.add_parser("chat", help="through the local gateway")
    p.add_argument("--model", default="auto")
    p.add_argument("-m", "--message")
    p.add_argument("--route", choices=["auto", "local", "remote"], default="auto")
    p.add_argument("--seller", help="an offer address")
    p.add_argument("--min-tier", choices=FLOORS)
    p.add_argument("--max-usd", type=float)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("serve", help="the local daemon and gateway, on 127.0.0.1")
    p.add_argument("--port", type=int, help=f"default {DEFAULT_PORT}, or a free one if that is taken")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("approve", help="what waits for a person, or approve one")
    p.add_argument("ticket", nargs="?")
    p.set_defaults(fn=cmd_approve)

    p = sub.add_parser("token", help="a limited token for an agent")
    p.add_argument("name")
    p.set_defaults(fn=cmd_token)

    p = sub.add_parser("wallet", help="balances, or test funds")
    p.add_argument("action", nargs="?", choices=["show", "fund"], default="show")
    p.add_argument("--mint-authority", type=Path, help="test networks: the key that mints test USDC")
    p.add_argument("--usdc", type=float, default=50)
    p.set_defaults(fn=cmd_wallet)

    host_cli.add_parser(sub)
    return ap


def main() -> None:
    args = parser().parse_args()
    try:
        sys.exit(args.fn(args))
    except ApiError as e:
        print(f"{e.code}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
