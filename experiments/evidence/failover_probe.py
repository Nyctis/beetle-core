"""A seller that is on the market and not there: does the buyer get to another one?

Posts a training offer from a new wallet with nothing behind it, cheaper than anyone
else, so a buyer that names no seller takes it. The offer is paused as soon as the job
is open, so it never catches another buyer for more than a few seconds. Only this
case: the seller is gone before it starts. A seller dying mid-run is not tested here.

    BEETLE_HOME=... python failover_probe.py --dataset NAME --deployer KEY --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "common"), str(ROOT)]

from solders.keypair import Keypair  # noqa: E402
from solders.system_program import TransferParams, transfer  # noqa: E402

import model_profiles as mp  # noqa: E402
from beetle_chain import KIND_TRAINING, Chain, OfferTerms, load_keypair, net_keypair  # noqa: E402
from client.api import Api  # noqa: E402
from job_defaults import TIERS  # noqa: E402
from node import wire  # noqa: E402

USDC = 1_000_000
BASE = "qwen3.5-2b"
RETRY_S = 45


def fund(chain: Chain, deployer: Keypair, owner, sol: float, usdc: float) -> None:
    chain.send([transfer(TransferParams(from_pubkey=deployer.pubkey(), to_pubkey=owner,
                                        lamports=int(sol * 1e9)))], [deployer])
    chain.mint_test_usdc(deployer, owner, int(usdc * USDC))


def wait_for(api: Api, ident: str, stages: tuple, limit: float) -> dict:
    """A job is marked failed before the buyer has gone on to the next seller, so a
    failed job gets RETRY_S more to say where it went."""
    end = time.time() + limit
    while time.time() < end:
        view = api.get(f"/beetle/jobs/{ident}")
        if view["stage"] == "failed":
            end = min(end, time.time() + RETRY_S)
            if view.get("next_job") or view.get("next_error"):
                return view
        elif view["stage"] in stages:
            return view
        time.sleep(2)
    return view


def run(a) -> dict:
    home = wire.home()
    chain = Chain.from_file(home / "chain.json")
    deployer = load_keypair(a.deployer)
    api = Api()
    profile = mp.get(BASE)
    me = wire.load_wallet(home).pubkey()
    fund(chain, deployer, me, 0.1, 5)
    ghost = Keypair()
    fund(chain, deployer, ghost.pubkey(), 0.08, 20)
    chain.register_seller(ghost, net_pk=net_keypair(os.urandom(32)).public_bytes)
    chain.stake(ghost, 16 * USDC)
    terms = OfferTerms(KIND_TRAINING, TIERS["t3c"], bytes(32), price_a=10 * USDC, tokens_per_s=500, grace_s=900,
                       start_window_s=900, valid_until=chain.chain_time() + 300, model_root=profile.root,
                       slots_total=1, mem_gb=8)
    chain.post_offer(ghost, ghost.pubkey(), 0, terms)
    offer = chain.get_offer(chain.offer_pda(ghost.pubkey(), 0))
    before = api.get("/beetle/wallet")
    ghost_stake = chain.get_seller(ghost.pubkey()).stake_total
    body = {"dataset": a.dataset, "base": BASE, "min_tier": "public", "max_usd": 3}
    quote = api.post("/beetle/quote", body)
    t0 = time.time()
    first = api.post("/beetle/jobs", {**body, "idempotency_key": uuid.uuid4().hex})
    chain.pause_offer(ghost, offer)
    out = {"quote": quote, "first_job": first["job"], "first_offer_is_ghost": first["offer"] == str(offer.address),
           "offer_visible_s": round(time.time() - t0, 1)}
    view = wait_for(api, first["job"], ("failed", "delivered", "settled"), 300)
    out["first"] = {"stage": view["stage"], "ended": view.get("ended"), "next_job": view.get("next_job"),
                    "next_error": view.get("next_error"), "s": round(time.time() - t0, 1),
                    "timeline": [{"stage": s["stage"], "s": round(s["at"] - t0, 1), "note": s["note"]}
                                 for s in view["timeline"]]}
    if view.get("next_job"):
        ident = view["next_job"]
        second = wait_for(api, ident, ("delivered", "settled", "failed"), 900)
        out["second_delivered_s"] = round(time.time() - t0, 1)
        out["second_offer"] = second["offer"]
        if second["stage"] == "delivered":
            fetched = api.post(f"/beetle/jobs/{ident}/fetch", {"alias": "after-failover"})
            out["result"] = fetched.get("result")
            second = api.get(f"/beetle/jobs/{ident}")
        out["second"] = {"stage": second["stage"], "chain_state": second.get("chain_state"),
                         "timeline": [{"stage": s["stage"], "s": round(s["at"] - t0, 1), "note": s["note"]}
                                      for s in second["timeline"]]}
    out["total_s"] = round(time.time() - t0, 1)
    after = api.get("/beetle/wallet")
    out["buyer_usdc_delta"] = round(after["usdc"] - before["usdc"], 6)
    out["ghost_stake_unchanged"] = chain.get_seller(ghost.pubkey()).stake_total == ghost_stake
    try:
        chain.close_offer(ghost, chain.get_offer(offer.address))
        out["ghost_offer_closed"] = True
    except RuntimeError as e:
        out["ghost_offer_closed"] = str(e)[:120]
    out["ghost_offer"] = str(offer.address)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--deployer", required=True, help="pays the test SOL and mints the test USDC")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = run(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k in ("first_offer_is_ghost", "offer_visible_s", "total_s", "result",
                                                               "buyer_usdc_delta", "ghost_stake_unchanged",
                                                               "ghost_offer_closed", "second_delivered_s")}))
    print(json.dumps(report.get("first"), ensure_ascii=False)[:900])


if __name__ == "__main__":
    main()
