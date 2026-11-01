"""One paid job through the real buyer path: quote, escrow, train, fetch, settle, chat.

It talks to the daemon of $BEETLE_HOME with that home's owner token, so it does what
`beetle train` and `beetle fetch` do and records the timing of each step. Repeat runs
of one dataset need a person's approval; the owner's token gives it here.

    BEETLE_HOME=... python experiments/evidence/run_collective.py \
        --dataset ds --size S --offer OFFER --tag s-l4-1 --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "common"), str(ROOT)]

from solders.pubkey import Pubkey  # noqa: E402

import model_profiles as mp  # noqa: E402
from beetle_chain import Chain  # noqa: E402
from client.api import Api, ApiError  # noqa: E402
from node import wire  # noqa: E402

WORK = json.loads((Path(__file__).parent / "workload.json").read_text())
ASK = "What is my cat's name?"


def stage_times(view: dict, t0: float) -> list[dict]:
    return [{"stage": s["stage"], "s": round(s["at"] - t0, 1), "note": s["note"]} for s in view["timeline"]]


def signatures(chain: Chain, job_id: bytes) -> list[str]:
    rows = chain.rpc.call("getSignaturesForAddress", str(chain.job_pda(job_id)), {"limit": 40})
    return [r["signature"] for r in reversed(rows) if r.get("err") is None]


def chat_once(api: Api, model: str, question: str, route: str) -> dict:
    text, headers, final, t0, first = [], {}, {}, time.time(), None
    for meta, piece in api.chat(model, [{"role": "user", "content": question}], route=route,
                                max_tokens=48, temperature=0.0):
        if meta:
            headers = meta
        elif isinstance(piece, str):
            first = first or time.time() - t0
            text.append(piece)
        else:
            final = piece
    return {"question": question, "answer": "".join(text), "route": headers.get("x-beetle-route"),
            "tier": headers.get("x-beetle-tier"), "proof": headers.get("x-beetle-proof"),
            "cost_usd": final.get("beetle", {}).get("cost_usd"), "s": round(time.time() - t0, 1),
            "first_piece_s": round(first, 1) if first else None}


def run(a) -> dict:
    home = wire.home()
    chain = Chain.from_file(home / "chain.json")
    api = Api()
    size = WORK["sizes"][a.size]
    profile = mp.get(size["base"])
    out: dict = {"tag": a.tag, "size": a.size, "workload": WORK["name"], "label": WORK["label"],
                 "base": profile.name, "repo": profile.repo, "revision": profile.revision,
                 "manifest_sha256": profile.manifest_sha256, "seller_role": a.seller_role,
                 "started_at": int(time.time())}
    before = api.get("/beetle/wallet")
    offer_row = next(r for r in api.get("/beetle/market") if r["offer"] == a.offer)
    seller = Pubkey.from_string(offer_row["seller"])
    out["offer_row"] = {k: offer_row[k] for k in ("offer", "kind", "model", "tier", "usd_per_mtok", "seller",
                                                  "gpu_gb", "settled", "failed", "stake_usd", "on_their_machine")}
    seller_before = chain.token_balance(seller)
    body = {"dataset": a.dataset, "base": profile.name, "min_tier": "public", "max_usd": a.max_usd,
            "seller": a.offer}
    t0 = time.time()
    out["quote"] = api.post("/beetle/quote", body)
    out["quote_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    job = api.post("/beetle/jobs", {**body, "idempotency_key": uuid.uuid4().hex, "approved": True})
    out["open_s"] = round(time.time() - t0, 2)
    ident = job["job"]
    out["job"], out["job_id"] = ident, job["job_id"]
    while True:
        view = api.get(f"/beetle/jobs/{ident}")
        if view["stage"] in ("delivered", "settled", "failed"):
            break
        if time.time() - t0 > a.timeout:
            out["timed_out"] = True
            break
        time.sleep(2)
    out["delivered_s"] = round(time.time() - t0, 1)
    out["timeline"] = stage_times(view, t0)
    out["stage_after_wait"] = view["stage"]
    if view["stage"] == "delivered":
        t1 = time.time()
        try:
            fetched = api.post(f"/beetle/jobs/{ident}/fetch", {"alias": a.tag})
            out["result"] = fetched.get("result")
        except ApiError as e:
            out["fetch_error"] = {"status": e.status, **e.error}
        out["fetch_s"] = round(time.time() - t1, 1)
        view = api.get(f"/beetle/jobs/{ident}")
        out["timeline"] = stage_times(view, t0)
    out["stage"], out["chain_state"] = view["stage"], view.get("chain_state")
    out["ended"], out["error"], out["verdict"] = view.get("ended"), view.get("error"), view.get("verdict")
    out["total_s"] = round(time.time() - t0, 1)
    out["proof"] = api.get(f"/beetle/proof/{ident}")
    out["path"] = out["proof"].get("path_raw")
    if view["stage"] == "settled" and a.chat:
        try:
            out["chat"] = chat_once(api, a.tag, ASK, "auto")
            out["chat_proof"] = api.get(f"/beetle/proof/{out['chat']['proof']}")
        except ApiError as e:
            out["chat_error"] = {"status": e.status, **e.error}
        api.post("/beetle/sessions/close")
    after = api.get("/beetle/wallet")
    out["buyer_usdc_delta"] = round(after["usdc"] - before["usdc"], 6)
    out["buyer_sol_delta"] = round(after["sol"] - before["sol"], 6)
    out["seller_usdc_delta_now"] = round(chain.token_balance(seller) - seller_before, 6)
    out["signatures"] = signatures(chain, bytes.fromhex(out["job_id"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--size", choices=sorted(WORK["sizes"]), required=True)
    ap.add_argument("--offer", required=True)
    ap.add_argument("--seller-role", required=True, help="a role name, written to the record")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-usd", type=float, default=3.0)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args()
    report = run(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report.get(k) for k in ("tag", "stage", "chain_state", "delivered_s", "fetch_s", "total_s",
                                                 "buyer_usdc_delta", "seller_usdc_delta_now", "ended", "error")}))
    print(json.dumps({"result": report.get("result"), "chat": report.get("chat"), "fetch_error": report.get("fetch_error"),
                      "chat_error": report.get("chat_error")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
