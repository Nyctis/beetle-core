"""M1: thirty minutes of a market on the real machines. A fixed seed draws who does
what and when; the machines, the links and the timing are real. At the end every
party must be square: funds conserved to the smallest unit, no job left open, no
channel stuck."""

from __future__ import annotations

import collections
import concurrent.futures as futures
import json
import random
import threading
import time

import harness as h
import final_items
import items
import setup_nodes
from beetle_chain import FINISHED

SEED = 20261031
# A run on machines with less room says who plays, in the config. Extra buyers are (name, machine like).
BUYERS = tuple(h.CONFIG.get("m1_buyers", ("mac", "mac2", "cpu", "cpu2", "t4", "t42")))
SELLERS = tuple(h.CONFIG.get("m1_sellers", ("l4", "a100", "a100b", "t4s")))
EXTRA = [tuple(pair) for pair in h.CONFIG.get("m1_extra", (("mac2", "mac"), ("cpu2", "cpu"), ("t42", "t4")))]
EVENTS = h.LOGS / "M1-events.jsonl"
lock = threading.Lock()
SELLER_STATE: dict = {}
counts: collections.Counter = collections.Counter()
errors: collections.Counter = collections.Counter()


def fourth_seller() -> None:
    if "stub" in SELLERS:
        final_items.stub_seller()
    else:
        items.t4_seller()


def why_no_seller(name: str) -> None:
    """Why the only training seller wasn't there: its state now, and whether its slot was taken."""
    offer = h.chain.get_offer(h.chain.offer_pda(items.names()["a100"], 0))
    note(name, "no_seller_why", seller_state=SELLER_STATE.get("a100"), slots_used=offer.slots_used if offer else None,
         slots_total=offer.slots_total if offer else None)


def note(who: str, what: str, **more) -> None:
    with lock:
        counts[what] += 1
        if more.get("error"):
            errors[f"{what}: {more['error']}"] += 1
        with open(EVENTS, "a") as f:
            f.write(json.dumps({"at": round(time.time()), "who": who, "what": what, **more}) + "\n")


def more_buyers() -> None:
    """A second daemon on each buyer machine: same box, own wallet."""
    for name, like in EXTRA:
        h.NODES[name] = {**h.NODES[like], "home": h.NODES[like]["home"].rsplit("/", 1)[0] + "/" + name}
        setup_nodes.buyer(name)
        items.WALLETS[name] = h.wallet(name)


def buyer_loop(name: str, until: float) -> None:
    rng = random.Random(f"{SEED}/{name}")
    pending: dict[str, str] = {}
    while time.time() < until:
        h.wait(rng.expovariate(1 / 25))
        roll, size = rng.random(), rng.choice((16, 32, 64, 128))
        try:
            for job, fate in list(pending.items()):
                view = items.job_view(name, job)
                stage = view.get("stage")
                if stage in ("failed", "settled"):
                    del pending[job]
                    note(name, "job_ended", stage=stage, ended=(view.get("ended") or "")[:60])
                elif stage == "delivered":
                    path = {"fetch": "fetch", "reject": "reject"}.get(fate)
                    if path:
                        done = h.api(name, "POST", f"/beetle/jobs/{job}/{path}", {"alias": "m1"}, 600)
                        note(name, path, error=(done.get("error") or {}).get("code"))
                        del pending[job]
                    elif fate == "ignore":
                        del pending[job]  # the seller will finalize; the drain tidies up
                        note(name, "ignore_delivery")
                elif fate == "cancel":
                    done = h.api(name, "POST", f"/beetle/jobs/{job}/cancel", {})
                    note(name, "cancel", stage=stage, error=(done.get("error") or {}).get("code"))
                    del pending[job]
            if roll < 0.60:
                model = items.BOLT if rng.random() < 0.25 else items.QWEN
                got = items.chat(name, model=model, max_tokens=size, timeout=200)
                row = got.get("rows", [{}])[0]
                note(name, "chat", model=model, ok=row.get("ok"), error=(row.get("error") or {}).get("code"), cost=row.get("cost_usd"))
            elif roll < 0.72:
                view = h.api(name, "POST", "/beetle/jobs", items.job_body("demo", items.QWEN))
                note(name, "train", job=view.get("job", "")[:12], error=(view.get("error") or {}).get("code"))
                if (view.get("error") or {}).get("code") == "no_seller":
                    why_no_seller(name)
                if "job" in view:
                    pending[view["job"]] = rng.choice(("fetch", "fetch", "reject", "cancel", "ignore"))
            elif roll < 0.82:
                h.stop_buyer(name, hard=True)
                away = rng.uniform(20, 120)
                note(name, "vanish", away_s=round(away))
                h.wait(away)
                h.api(name, "GET", "/beetle/health")
                note(name, "back")
            elif roll < 0.90:
                h.api(name, "POST", "/beetle/sessions/close")
                note(name, "close_sessions")
        except Exception as e:  # a machine or a tunnel hiccup is an event too
            note(name, "harness_error", error=repr(e)[:120])


def seller_loop(until: float) -> None:
    rng = random.Random(f"{SEED}/sellers")
    state = SELLER_STATE
    state.update({s: "up" for s in SELLERS})
    while time.time() < until:
        h.wait(rng.expovariate(1 / 100))
        seller = rng.choice(SELLERS)
        try:
            if state[seller] == "up":
                if rng.random() < 0.5:
                    h.beetle(seller, "host pause")
                    state[seller] = "paused"
                else:
                    hard = rng.random() < 0.5
                    h.stop_host(seller, hard=hard)
                    state[seller] = "down"
                note(seller, state[seller])
            elif state[seller] == "paused":
                h.beetle(seller, "host resume")
                state[seller] = "up"
                note(seller, "resumed")
            else:
                h.start_host(seller)
                state[seller] = "up"
                note(seller, "online")
        except Exception as e:
            note(seller, "harness_error", error=repr(e)[:120])
    for seller, was in state.items():
        if was == "down":
            h.start_host(seller)
        h.beetle(seller, "host resume")


def jobs_open(name: str) -> list[dict]:
    rows = h.api(name, "GET", "/beetle/jobs")
    rows = rows if isinstance(rows, list) else rows.get("jobs", [])
    return [r for r in rows if r.get("stage") not in ("failed", "settled")]


def drain(deadline_s: float) -> dict:
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        left = 0
        for name in BUYERS:
            for job in jobs_open(name):
                left += 1
                if job.get("stage") == "delivered":
                    h.api(name, "POST", f"/beetle/jobs/{job['job']}/fetch", {"alias": "m1"}, 600)
            h.api(name, "POST", "/beetle/sessions/close")
        opened = sum(items.settled(name, 1)["open_channels"] for name in BUYERS)
        if not left and not opened:
            break
        h.log(f"drain: {left} jobs, {opened} channels open")
        h.wait(30)
    return {"drain_s": round(time.time() - t0, 1)}


def scan(wallets: dict) -> dict:
    jobs, channels = collections.Counter(), collections.Counter()
    stuck = []
    for seller in SELLERS:
        for job in h.chain.jobs_for_host(wallets[seller]):
            jobs[job.state] += 1
            if job.state not in FINISHED or job.seller_due or job.buyer_due:
                stuck.append(f"job {job.job_id.hex()[:10]} {job.state} due {job.seller_due}/{job.buyer_due}")
        for ch in h.chain.channels_for(wallets[seller]):
            channels[ch.status] += 1
            if ch.status != "Distributed":
                stuck.append(f"channel {str(ch.address)[:10]} {ch.status}")
    return {"jobs": dict(jobs), "channels": dict(channels), "stuck": stuck}


def party(name: str) -> dict:
    """One buyer's books against the chain: what it was funded, less what its ledger settled."""
    view = h.api(name, "GET", "/beetle/wallet")
    spending = view.get("spending", {})
    return {"wallet_usd": view.get("usdc"), "ledger_settled_usd": round(spending.get("settled", 0.0), 6),
            "held": round(spending.get("escrowed", 0.0) + spending.get("reserved", 0.0), 6),
            "wallet_plus_ledger": round((view.get("usdc") or 0) + spending.get("settled", 0.0), 6)}


def finish() -> None:
    """The verdict, from the chain and the event log, if a run ended without reaching it."""
    h.current_item = "M1"
    fourth_seller()
    for name, like in EXTRA:
        h.NODES[name] = {**h.NODES[like], "home": h.NODES[like]["home"].rsplit("/", 1)[0] + "/" + name}
        items.WALLETS[name] = h.wallet(name)
    wallets = items.names()
    for seller in SELLERS:
        if h.sh(seller, "kill -0 $(cat $BEETLE_HOME/host.pid) 2>/dev/null")[0]:
            h.start_host(seller)
        h.beetle(seller, "host resume")
    rows = [json.loads(line) for line in EVENTS.read_text().splitlines()]
    actions = collections.Counter(r["what"] for r in rows)
    failed = collections.Counter(f"{r['what']}: {r['error']}" for r in rows if r.get("error"))
    tidy = drain(1500)
    found = scan(wallets)
    after, minted = h.token_accounts(), h.supply()
    books = {name: party(name) for name in BUYERS}
    numbers = {"seed": SEED, "seconds": rows[-1]["at"] - rows[0]["at"], "actions": dict(actions), "errors": dict(failed.most_common(20)), **tidy, **found,
               "all_accounts_equal_supply": sum(after.values()) == minted, "supply_units": minted, "buyers": books,
               "note": "the run's first ending crashed in the harness after the market closed; this verdict was taken from the chain afterwards"}
    fine = numbers["all_accounts_equal_supply"] and not found["stuck"]
    h.record("M1", "PASS" if fine else "FAIL", numbers)


def run(seconds: int) -> None:
    h.current_item = "M1"
    fourth_seller()
    more_buyers()
    wallets = items.names()
    books = h.Books()
    EVENTS.unlink(missing_ok=True)
    until = time.time() + seconds
    t0 = time.time()
    with futures.ThreadPoolExecutor(len(BUYERS) + 1) as pool:
        tasks = [pool.submit(buyer_loop, name, until) for name in BUYERS] + [pool.submit(seller_loop, until)]
        for task in tasks:
            task.result()
    ran = round(time.time() - t0)
    h.say(f"ran {ran} s: {dict(counts)}")
    tidy = drain(1500)
    found = scan(wallets)
    money = books.close(wallets)
    numbers = {"seed": SEED, "seconds": ran, "actions": dict(counts), "errors": dict(errors.most_common(20)), **tidy, **found,
               "conserved_to_the_unit": money["conserved"], "all_accounts_equal_supply": money["all_accounts_equal_supply"],
               "minted_during_run": money["minted"], "delta": money["delta"]}
    fine = money["conserved"] and money["all_accounts_equal_supply"] and not found["stuck"]
    h.record("M1", "PASS" if fine else "FAIL", numbers)
