"""Items added for the final regression: inference billing, cut links, timeouts and
empty wallets, which the first matrix did not reach, and the fields added since.

A stub seller gives the exact cases. It is a real seller daemon on the cpu node with a
real wallet and offer; only its model server is stub_backend.py, so a token count, a
hang or a broken stream happens on demand. Billing is also checked against a real
model server, on the l4 seller.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import shlex
import time

import harness as h
import items
from items import QWEN, chat, good, job_body, job_end, job_view, offers, settled
from solders.pubkey import Pubkey

STUB, BUYER, NF, NF2, LOCAL = "stub", "t4", "nf", "nf2", "lo"
STUB_PORT = 42750
SESSION_PORT = STUB_PORT + 1
BACKEND_PORT = 18100
PRICE_IN, PRICE_OUT = 2, 5
STUN = " ".join(f"--stun {s}" for s in h.CONFIG["stun"])


def units(usd: float) -> int:
    return round(usd * h.USDC)


def spend(view: dict) -> float:
    """What this buyer's ledger says inference cost, as the chain settled it. `used` in
    items.spend is what was metered, which is less when a session ended early."""
    return sum(e.get("usd") or 0.0 for e in view["entries"].values() if e["kind"] == "infer" and e["state"] == "settled")


def stub_seller() -> None:
    if STUB in items.WALLETS:
        return
    cpu = h.NODES["cpu"]
    h.NODES[STUB] = {**cpu, "home": f"{items.BASE}/stub", "port": STUB_PORT}
    if h.sh(STUB, "kill -0 $(cat $BEETLE_HOME/host.pid) 2>/dev/null")[0] == 0:
        items.WALLETS[STUB] = h.wallet(STUB)  # left running by an earlier item
        return
    source = (h.HERE / "stub_backend.py").read_text()
    h.sh(STUB, f"cat > $BEETLE_HOME/.stub_backend.py <<'STUBEND'\n{source}\nSTUBEND\n"
               f"(nohup .venv/bin/python $BEETLE_HOME/.stub_backend.py {BACKEND_PORT} {QWEN} > $BEETLE_HOME/stub.log 2>&1 "
               f"& echo $! > $BEETLE_HOME/stub.pid); sleep 8", quiet=True)
    h.sh(STUB, f"cp {cpu['home']}/matrix-chain.json $BEETLE_HOME/")
    h.beetle(STUB, f"host init --chain $BEETLE_HOME/matrix-chain.json --port {STUB_PORT} "
                   f"--backend http://127.0.0.1:{BACKEND_PORT} {STUN}")
    items.host_set(STUB, scratch=f"/dev/shm{items.BASE.replace('~', '')}/stub")
    h.fund(h.wallet(STUB), 400)
    rc, out = h.beetle(STUB, f"host list --models {QWEN} --infer-only --serves {QWEN} "
                             f"--infer-in {PRICE_IN} --infer-out {PRICE_OUT}")
    assert rc == 0, out[-300:]
    h.start_host(STUB)
    h.wait(15)
    items.WALLETS[STUB] = h.wallet(STUB)


def stub_stats() -> dict:
    out = h.sh(STUB, f"curl -s -m 5 127.0.0.1:{BACKEND_PORT}/stats")[1]
    return json.JSONDecoder().raw_decode(out[out.index("{"):])[0]  # ssh may add a line of its own after it


def offer_of(buyer: str, seller: str) -> str:
    return items.row_of(buyer, seller, QWEN, "infer")["offer"]


def channels(seller: str) -> list:
    return h.chain.channels_for(items.names()[seller])


def fresh(seller: str, already: set) -> list:
    return [c for c in channels(seller) if str(c.address) not in already]


def balances(*who: str) -> dict:
    w = items.names()
    return {k: units(h.chain.token_balance(w[k])) for k in who}


def settle_view(buyer: str, seller: str, already: set, before: dict, ledger_before: float, wait_s: float = 300) -> dict:
    """Close, wait until the channel is done, and read every side's account of the money."""
    done = settled(buyer, wait_s)
    t0 = time.time()
    mine = fresh(seller, already)
    while time.time() - t0 < 200 and any(c.status != "Distributed" for c in mine):
        h.wait(5)
        mine = fresh(seller, already)
    after = balances(buyer, seller)
    return {"open_channels": done["open_channels"], "channels": [c.status for c in mine],
            "channel_settled": sum(c.settled for c in mine), "deposit": sum(c.deposit for c in mine),
            "buyer_paid": before[buyer] - after[buyer], "seller_got": after[seller] - before[seller],
            "ledger": units(spend(done["view"]) - ledger_before)}


def N1():
    """Billing to the unit, against a server whose counts are known."""
    h.current_item = "N1"
    stub_seller()
    books = h.Books()
    settled(BUYER, 200)
    offer = offer_of(BUYER, STUB)
    terms = h.chain.get_offer(Pubkey.from_string(offer))
    already = {str(c.address) for c in channels(STUB)}
    before, ledger0 = balances(BUYER, STUB), spend(h.api(BUYER, "GET", "/beetle/wallet"))
    stats0 = stub_stats()
    rows = []
    for count in (1, 7, 64, 500):
        rows += chat(BUYER, n=1, max_tokens=1000, prompt=f"COUNT {count}", seller=offer, keep_text=True, timeout=240).get("rows", [])
    h.say(str([{k: r.get(k) for k in ("ok", "usage", "cost_usd", "error", "total_s")} for r in rows])[:900])
    stats = stub_stats()
    money = settle_view(BUYER, STUB, already, before, ledger0)
    fine_rows = good(rows)
    expected = sum(terms.usage_cost(r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]) for r in fine_rows)
    numbers = {"requests": [{"pieces": r.get("pieces"), "usage": r.get("usage"), "cost_units": units(r.get("cost_usd") or 0)} for r in rows],
               "server_prompt": stats["prompt_tokens"] - stats0["prompt_tokens"],
               "server_completion": stats["completion_tokens"] - stats0["completion_tokens"],
               "buyer_prompt": sum(r["usage"]["prompt_tokens"] for r in fine_rows),
               "buyer_completion": sum(r["usage"]["completion_tokens"] for r in fine_rows),
               "expected_units": expected, **money, "books": books.close(items.names())["conserved"]}
    fine = len(fine_rows) == 4 and numbers["server_prompt"] == numbers["buyer_prompt"] \
        and numbers["server_completion"] == numbers["buyer_completion"] \
        and expected == money["channel_settled"] == money["buyer_paid"] == money["seller_got"] == money["ledger"] \
        and sum(n["cost_units"] for n in numbers["requests"]) == expected and numbers["books"]
    h.record("N1", "PASS" if fine else "FAIL", numbers, f"stub server, {PRICE_IN}/{PRICE_OUT} USD per million tokens")


def N2():
    """Billing against a real model server: the server's own counters, the buyer's
    count, the channel and three wallets' worth of accounts."""
    h.current_item = "N2"
    books = h.Books()
    buyer = "mac"
    settled(buyer, 200)
    offer = offer_of(buyer, "l4")
    terms = h.chain.get_offer(Pubkey.from_string(offer))
    already = {str(c.address) for c in channels("l4")}
    before, ledger0 = balances(buyer, "l4"), spend(h.api(buyer, "GET", "/beetle/wallet"))
    made0 = items.backend_tokens()
    rows = []
    for size in (1, 16, 64, 256):
        rows += chat(buyer, n=1, max_tokens=size, seller=offer, timeout=300).get("rows", [])
    made = items.backend_tokens()
    h.say(str([{k: r.get(k) for k in ("ok", "usage", "cost_usd", "error")} for r in rows])[:900])
    money = settle_view(buyer, "l4", already, before, ledger0)
    fine_rows = good(rows)
    expected = sum(terms.usage_cost(r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]) for r in fine_rows)
    numbers = {"requests": [{"usage": r.get("usage"), "cost_units": units(r.get("cost_usd") or 0)} for r in rows],
               "server_prompt": made[0] - made0[0], "server_generated": made[1] - made0[1],
               "billed_prompt": sum(r["usage"]["prompt_tokens"] for r in fine_rows),
               "billed_completion": sum(r["usage"]["completion_tokens"] for r in fine_rows),
               "expected_units": expected, **money, "books": books.close(items.names())["conserved"]}
    numbers["completion_gap"] = numbers["server_generated"] - numbers["billed_completion"]
    numbers["prompt_gap"] = numbers["server_prompt"] - numbers["billed_prompt"]
    fine = len(fine_rows) == 4 and expected == money["channel_settled"] == money["buyer_paid"] == money["seller_got"] == money["ledger"] \
        and numbers["books"]
    h.record("N2", "PASS" if fine else "FAIL", numbers,
             "a buyer pays the smaller of the two counts, so prompt_gap at or above 0 is the buyer's favour")


def N3():
    """A deposit smaller than the answer: the answer stops, nothing is owed past the
    deposit, and the session says so."""
    h.current_item = "N3"
    stub_seller()
    books = h.Books()
    settled(BUYER, 200)
    offer = offer_of(BUYER, STUB)
    already = {str(c.address) for c in channels(STUB)}
    before, ledger0 = balances(BUYER, STUB), spend(h.api(BUYER, "GET", "/beetle/wallet"))
    stats0 = stub_stats()
    deposit_usd = 0.01  # a step (a fiftieth) must still cover one request
    first = chat(BUYER, n=1, max_tokens=3000, prompt="PACE 5 3000", seller=offer, keep_text=True, max_usd=deposit_usd, timeout=240)
    row = (first.get("rows") or [first])[0]
    h.say(f"first: {str({k: v for k, v in row.items() if k != 'text'})[:500]}")
    second = chat(BUYER, n=1, max_tokens=8, prompt="COUNT 8", seller=offer, keep_text=True, max_usd=deposit_usd, timeout=120)
    again = (second.get("rows") or [second])[0]
    h.say(f"second: {str({k: v for k, v in again.items() if k != 'text'})[:400]}")
    stats = stub_stats()
    money = settle_view(BUYER, STUB, already, before, ledger0)
    numbers = {"pieces_received": row.get("pieces"), "asked_for": 3000, "first_ok": row.get("ok"), "first_error": row.get("error"),
               "first_usage": row.get("usage"), "first_s": row.get("total_s"), "first_cost_units": units(row.get("cost_usd") or 0),
               "second_error": again.get("error"), "second_ok": again.get("ok"), "deposit_units": units(deposit_usd),
               "server_sent_pieces": stats["pieces"] - stats0["pieces"], "server_streams_cut_off": stats["cut_off"] - stats0["cut_off"],
               **money, "books": books.close(items.names())["conserved"]}
    fine = money["buyer_paid"] <= units(deposit_usd) and money["channel_settled"] <= money["deposit"] and numbers["books"] \
        and 0 < (row.get("usage") or {}).get("completion_tokens", 0) * PRICE_OUT <= units(deposit_usd) and money["buyer_paid"] == money["ledger"]
    h.record("N3", "PASS" if fine else "FAIL", numbers, "the verdict is about money; how the stop reads to the caller is in the numbers")


def timed_chat(prompt: str, offer: str, limit_s: int, **more) -> dict:
    """One request that may never end: the client dies at `limit_s`, the buyer's daemon does not."""
    got = items.on_node(BUYER, {"model": QWEN, "n": 1, "max_tokens": 2500, "prompt": prompt, "route": "remote", "seller": offer,
                                "keep_text": True, **more}, timeout=limit_s + 30, tool="chat_probe.py", prefix=f"timeout {limit_s}")
    return (got.get("rows") or [{"error": {"code": "client_gave_up", "message": str(got)[:160]}}])[0]


def N4():
    """Timeouts and failures of the model server and of a seller that stops answering."""
    h.current_item = "N4"
    stub_seller()
    books = h.Books()
    settled(BUYER, 200)
    offer = offer_of(BUYER, STUB)
    already = {str(c.address) for c in channels(STUB)}
    before, ledger0 = balances(BUYER, STUB), spend(h.api(BUYER, "GET", "/beetle/wallet"))
    numbers: dict = {}

    def show(row: dict) -> dict:
        return {k: row.get(k) for k in ("ok", "error", "pieces", "total_s", "usage", "cost_usd")}

    numbers["server_500"] = show(timed_chat("ERROR", offer, 120))
    h.say(f"500: {numbers['server_500']}")
    numbers["after_500"] = show(timed_chat("COUNT 8", offer, 120))
    numbers["stream_breaks"] = show(timed_chat("BREAK 20", offer, 120))
    h.say(f"break: {numbers['stream_breaks']}")
    numbers["after_break"] = show(timed_chat("COUNT 8", offer, 120))
    money = settle_view(BUYER, STUB, already, before, ledger0)
    numbers.update(money=money, books=books.close(items.names())["conserved"])
    fine = all(numbers[k].get("error") for k in ("server_500", "stream_breaks")) and all(numbers[k].get("ok") for k in ("after_500", "after_break")) \
        and numbers["books"] and money["buyer_paid"] == money["ledger"] == money["seller_got"]
    h.record("N4", "PASS" if fine else "FAIL", numbers, "the model server refuses, or drops the stream midway")


def N7():
    """The model server stops mid-answer and stays connected: what the caller sees, and
    whether the session comes back."""
    h.current_item = "N7"
    stub_seller()
    books = h.Books()
    settled(BUYER, 200)
    offer = offer_of(BUYER, STUB)
    already = {str(c.address) for c in channels(STUB)}
    before, ledger0 = balances(BUYER, STUB), spend(h.api(BUYER, "GET", "/beetle/wallet"))
    limit = 400
    t0 = time.time()
    with futures.ThreadPoolExecutor(2) as pool:
        running = pool.submit(timed_chat, "HANG 5", offer, limit)
        h.wait(40)
        queued = pool.submit(timed_chat, "COUNT 8", offer, limit - 40)
        first, second = running.result(), queued.result()
    waited = round(time.time() - t0, 1)
    h.say(f"hung: {str({k: v for k, v in first.items() if k != 'text'})[:400]}")
    h.say(f"queued behind it: {str({k: v for k, v in second.items() if k != 'text'})[:400]}")
    stats = stub_stats()
    next_one = timed_chat("COUNT 8", offer, 200)
    h.say(f"next: {str({k: v for k, v in next_one.items() if k != 'text'})[:300]}")
    money = settle_view(BUYER, STUB, already, before, ledger0, 400)
    stuck = ledger_stuck(BUYER)
    h.say(f"ledger entries left in escrow: {stuck}")
    h.stop_buyer(BUYER, hard=False)  # what clears them, if anything does
    h.wait(3)
    h.api(BUYER, "GET", "/beetle/health")
    h.wait(70)
    after_restart = ledger_stuck(BUYER)
    h.say(f"after a daemon restart: {after_restart}")
    numbers = {"limit_s": limit, "first": {k: first.get(k) for k in ("ok", "error", "pieces", "total_s")},
               "queued": {k: second.get(k) for k in ("ok", "error", "pieces", "total_s")}, "waited_s": waited,
               "server_stream_still_open": stats["open"], "next_request": {k: next_one.get(k) for k in ("ok", "error", "total_s")},
               "money": money, "ledger_stuck_before_restart": stuck, "ledger_stuck_after_restart": after_restart,
               "books": books.close(items.names())["conserved"], "stats_end": stub_stats()}
    told = (first.get("error") or {}).get("code") not in (None, "client_gave_up")
    fine = told and first.get("total_s", 1e9) < limit and next_one.get("ok") and numbers["books"]
    h.record("N7", "PASS" if fine else "FAIL", numbers, f"the client waits {limit} s at most; a FAIL means no answer came in that time")


def ledger_stuck(buyer: str) -> list:
    """Channels this ledger still calls escrowed, and what the chain says of each."""
    out = []
    view = h.api(buyer, "GET", "/beetle/wallet")
    for entry in view.get("entries", {}).values():
        if entry["kind"] == "infer" and entry["state"] == "escrowed":
            ch = h.chain.get_channel(Pubkey.from_string(entry["channel"])) if entry.get("channel") else None
            out.append({"usd": entry.get("usd"), "signed": entry.get("signed"), "closed_locally": bool(entry.get("closed_at")),
                        "chain": ch.status if ch else "account gone"})
    return out


def N8():
    """The seller's processes freeze, as a hung machine would: the link goes silent."""
    h.current_item = "N8"
    stub_seller()
    books = h.Books()
    settled(BUYER, 200)
    offer = offer_of(BUYER, STUB)
    already = {str(c.address) for c in channels(STUB)}
    before, ledger0 = balances(BUYER, STUB), spend(h.api(BUYER, "GET", "/beetle/wallet"))
    limit = 300
    with futures.ThreadPoolExecutor(1) as pool:
        t0 = time.time()
        running = pool.submit(timed_chat, "PACE 100 800", offer, limit)
        h.wait(25)
        h.sh(STUB, "p=$(cat $BEETLE_HOME/host.pid); pkill -STOP -P $p; kill -STOP $p")
        stopped = time.time()
        try:
            row = running.result()
        finally:
            h.sh(STUB, "p=$(cat $BEETLE_HOME/host.pid); kill -CONT $p; pkill -CONT -P $p")
    h.say(f"frozen: {str({k: v for k, v in row.items() if k != 'text'})[:400]}")
    h.wait(20)
    next_one = timed_chat("COUNT 8", offer, 240)
    h.say(f"after: {str({k: v for k, v in next_one.items() if k != 'text'})[:300]}")
    money = settle_view(BUYER, STUB, already, before, ledger0, 400)
    numbers = {"row": {k: row.get(k) for k in ("ok", "error", "pieces", "total_s")}, "freeze_began_after_s": round(stopped - t0, 1),
               "s_from_freeze_to_end": round(t0 + (row.get("total_s") or 0) - stopped, 1), "next_request": {k: next_one.get(k) for k in ("ok", "error", "total_s")},
               "money": money, "books": books.close(items.names())["conserved"]}
    told = (row.get("error") or {}).get("code") not in (None, "client_gave_up")
    h.record("N8", "PASS" if told and next_one.get("ok") and numbers["books"] else "FAIL", numbers, f"the client waits {limit} s at most")


def new_buyer(name: str, flags: str) -> None:
    cpu = h.NODES["cpu"]
    h.NODES[name] = {**cpu, "home": f"{items.BASE}/{name}"}
    h.sh(name, f"cp {cpu['home']}/matrix-chain.json $BEETLE_HOME/")
    h.beetle(name, f"init --chain $BEETLE_HOME/matrix-chain.json {flags}")


def N5():
    """Wallets that can't pay, at each moment it matters. The l4 seller must be paused:
    its model server here takes no adapters, and the check after delivery needs one that does."""
    h.current_item = "N5"
    stub_seller()
    books = h.Books()
    new_buyer(NF, f"--per-job 5 --per-day 200 --min-tier public --auto-under 5 --traffic burst {STUN}")
    h.beetle(NF, "data add demo --name demo --label synthetic")
    key = h.wallet(NF)
    items.WALLETS[NF] = key
    offer = offer_of(NF, STUB)
    train = items.row_of(NF, "a100")["offer"]
    numbers: dict = {}

    def attempt(label: str) -> None:
        got = chat(NF, n=1, max_tokens=8, prompt="COUNT 8", seller=offer, timeout=120)
        row = (got.get("rows") or [got])[0]
        job = h.api(NF, "POST", "/beetle/jobs", job_body("demo", QWEN, seller=train))
        numbers[label] = {"chat": row.get("error"), "job": job.get("error"), "job_opened": "job" in job}
        h.say(f"{label}: {numbers[label]}")

    attempt("no_sol")
    h.chain.airdrop(key, 2_000_000_000)
    attempt("sol_no_usdc")
    h.chain.mint_test_usdc(h.deployer, key, 300_000)
    attempt("usdc_below_deposit")
    quote = h.api(NF, "POST", "/beetle/quote", {"dataset": "demo", "base": QWEN, "method": "memory", "seller": train})
    price = units(quote.get("usd", 0))
    numbers["quote_usd"] = quote.get("usd")
    h.chain.mint_test_usdc(h.deployer, key, max(price - 300_000, 0))

    # Manual fetch: delivered, with the whole wallet in escrow, so nothing is left for the check.
    job = h.api(NF, "POST", "/beetle/jobs", job_body("demo", QWEN, seller=train))
    h.say(f"job: {str(job)[:300]}")
    end = job_end(NF, job.get("job", "x"), ("delivered", "failed", "settled"), 600)
    numbers["delivered_stage"] = end.get("stage")
    numbers["wallet_after_escrow_usd"] = h.chain.token_balance(key)
    first = h.api(NF, "POST", f"/beetle/jobs/{job.get('job')}/fetch", {"alias": "nf1"}, 300)
    view = job_view(NF, job.get("job"))
    numbers["fetch_without_funds"] = {"error": first.get("error"), "stage_after": view.get("stage"), "job_error": view.get("error")}
    h.say(f"fetch without funds: {numbers['fetch_without_funds']}")
    h.chain.mint_test_usdc(h.deployer, key, 1_000_000)
    second = h.api(NF, "POST", f"/beetle/jobs/{job.get('job')}/fetch", {"alias": "nf1"}, 600)
    numbers["fetch_after_top_up"] = {"stage": second.get("stage"), "error": second.get("error"), "paid_usd": (second.get("result") or {}).get("paid_usd"),
                                     "effect": (second.get("result") or {}).get("effect")}
    h.say(f"after top up: {numbers['fetch_after_top_up']}")
    settled(NF, 200)

    # With an alias the fetch is automatic: it stops at delivered, and a restart picks it up.
    # A second wallet, holding exactly the job's price.
    new_buyer(NF2, f"--per-job 5 --per-day 200 --min-tier public --auto-under 5 --traffic burst {STUN}")
    h.beetle(NF2, "data add demo --name demo --label synthetic")
    key = h.wallet(NF2)
    items.WALLETS[NF2] = key
    h.fund(key, price / h.USDC, 2)
    job = h.api(NF2, "POST", "/beetle/jobs", job_body("demo", QWEN, seller=train, alias="nf2"))
    h.say(f"job 2: {str(job)[:300]}")
    t0 = time.time()
    view = {}
    while time.time() - t0 < 600:
        view = job_view(NF2, job.get("job", "x"))
        if view.get("stage") in ("failed", "settled") or (view.get("error") or {}).get("code") == "no_funds":
            break
        h.wait(5)
    numbers["alias_without_funds"] = {"stage": view.get("stage"), "error": view.get("error")}
    h.say(f"alias without funds: {numbers['alias_without_funds']}")
    h.chain.mint_test_usdc(h.deployer, key, 1_000_000)
    h.stop_buyer(NF2, hard=False)
    h.wait(3)
    t1 = time.time()
    end = job_end(NF2, job.get("job", "x"), ("failed", "settled"), 400)  # this starts the daemon again
    numbers["restart_picks_it_up"] = {"stage": end.get("stage"), "s": round(time.time() - t1, 1), "error": end.get("error"),
                                      "paid_usd": (end.get("result") or {}).get("paid_usd")}
    h.say(f"after restart: {numbers['restart_picks_it_up']}")
    settled(NF2, 200)
    h.stop_buyer(NF, hard=False)
    h.stop_buyer(NF2, hard=False)

    new_buyer(LOCAL, "--local-only")
    numbers["local_only"] = {"market": h.api(LOCAL, "GET", "/beetle/market?kind=infer").__repr__()[:160],
                             "job": h.api(LOCAL, "POST", "/beetle/jobs", job_body("demo", QWEN)).__repr__()[:160],
                             "health_chain": h.api(LOCAL, "GET", "/beetle/health").get("chain")}
    h.say(f"local only: {numbers['local_only']}")
    h.stop_buyer(LOCAL, hard=False)
    money = books.close(items.names())
    numbers["books"] = money["conserved"]
    refused = all(numbers[k]["chat"] and numbers[k]["chat"].get("code") == "no_funds" and numbers[k]["job"] and numbers[k]["job"].get("code") == "no_funds"
                  and not numbers[k]["job_opened"] for k in ("no_sol", "sol_no_usdc", "usdc_below_deposit"))
    fine = refused and numbers["fetch_without_funds"]["stage_after"] == "delivered" and numbers["fetch_after_top_up"]["stage"] == "settled" \
        and (numbers["alias_without_funds"]["error"] or {}).get("code") == "no_funds" and numbers["restart_picks_it_up"]["stage"] == "settled" \
        and "local_only" in numbers["local_only"]["market"] and numbers["books"]
    h.record("N5", "PASS" if fine else "FAIL", numbers)


def blackout(on: bool) -> None:
    action = "-I" if on else "-D"
    h.sh(STUB, f"sudo -n iptables {action} INPUT -p udp --dport {SESSION_PORT} -j DROP; "
               f"sudo -n iptables {action} OUTPUT -p udp --sport {SESSION_PORT} -j DROP")


def in_order(text: str) -> bool:
    return text.split() == [f"w{i}" for i in range(len(text.split()))]


def N6():
    """The link goes dark mid-answer, for less and for more than the resume budget.
    Only UDP on this stub seller's session port, on the cpu node, for a set time."""
    h.current_item = "N6"
    stub_seller()
    books = h.Books()
    settled(BUYER, 200)
    offer = offer_of(BUYER, STUB)
    already = {str(c.address) for c in channels(STUB)}
    before, ledger0 = balances(BUYER, STUB), spend(h.api(BUYER, "GET", "/beetle/wallet"))
    rows: dict = {}
    try:
        for dark in (8, 45, 150):
            with futures.ThreadPoolExecutor(1) as pool:
                t0 = time.time()
                running = pool.submit(timed_chat, "PACE 50 1800", offer, 420)
                h.wait(12)
                blackout(True)
                off = time.time()
                h.wait(dark)
                blackout(False)
                row = running.result()
            text = row.get("text", "")
            rows[f"dark_{dark}s"] = {"ok": row.get("ok"), "error": row.get("error"), "pieces": row.get("pieces"), "s": row.get("total_s"),
                                     "s_after_dark_began": round(t0 + (row.get("total_s") or 0) - off, 1), "in_order": in_order(text),
                                     "usage": row.get("usage"), "cost_usd": row.get("cost_usd")}
            h.say(f"dark {dark}: {rows[f'dark_{dark}s']}")
            nxt = timed_chat("COUNT 8", offer, 200)
            rows[f"after_{dark}s"] = {"ok": nxt.get("ok"), "error": nxt.get("error"), "s": nxt.get("total_s")}
            h.say(f"after {dark}: {rows[f'after_{dark}s']}")
    finally:
        h.sh(STUB, f"for i in 1 2 3; do sudo -n iptables -D INPUT -p udp --dport {SESSION_PORT} -j DROP 2>/dev/null; "
                   f"sudo -n iptables -D OUTPUT -p udp --sport {SESSION_PORT} -j DROP 2>/dev/null; done; true")
    money = settle_view(BUYER, STUB, already, before, ledger0, 400)
    numbers = {**rows, "money": money, "books": books.close(items.names())["conserved"],
               "rules_left": h.sh(STUB, f"sudo -n iptables -S | grep -c {SESSION_PORT}; true")[1].strip()}
    fine = rows["dark_8s"].get("ok") and rows["dark_8s"].get("in_order") and rows["after_150s"].get("ok") \
        and rows["dark_150s"].get("error") and numbers["books"] and money["buyer_paid"] == money["ledger"]
    h.record("N6", "PASS" if fine else "FAIL", numbers, "UDP dropped on the seller's session port; the buyer's daemon and the seller are untouched")


def T1L():
    """T1's local half. The T4 has no room for a server of its own, so this buyer sits on
    the A100, beside the seller, with its own model server as its local backend."""
    name = "a100local"
    a100 = h.NODES["a100"]
    h.current_item = "T1L"
    h.NODES[name] = {**a100, "home": f"{items.BASE}/{name}"}
    h.sh(name, f"cp {a100['home']}/matrix-chain.json $BEETLE_HOME/")
    h.beetle(name, f"init --chain $BEETLE_HOME/matrix-chain.json --per-job 5 --per-day 200 --min-tier public --auto-under 5 "
                   f"--traffic burst {STUN} --local-backend {a100['backend']} --local-model {QWEN}")
    code = ("import json,os; p=os.environ['BEETLE_HOME']+'/config.json'; c=json.load(open(p))\n"
            f"c['local_backend']['scratch']={items.SHARED + '/' + name!r}\njson.dump(c,open(p,'w'),indent=1)")
    h.sh(name, f"python3 -c {shlex.quote(code)}")
    h.fund(h.wallet(name), 200)
    h.beetle(name, "data add demo --name demo --label synthetic")
    items.WALLETS[name] = h.wallet(name)
    items.effect_item("T1L", name)
    h.stop_buyer(name, hard=False)


def N9():
    """The cheapest seller's server takes no adapters. Does a buyer with an adapter move on?"""
    h.current_item = "N9"
    stub_seller()
    rc, out = h.beetle(STUB, f"host list --models {QWEN} --infer-only --serves {QWEN} --infer-in 0.1 --infer-out 0.1")
    h.say(f"stub repriced: {rc}")
    h.wait(10)
    books = h.Books()
    settled("cpu", 200)
    got = chat("cpu", model="i2", n=1, max_tokens=32, temperature=0, keep_text=True, prompt=items.QUESTION, timeout=300)
    row = (got.get("rows") or [got])[0]
    h.say(f"adapter, no seller named: {str({k: v for k, v in row.items() if k != 'proof'})[:500]}")
    named = chat("cpu", model="i2", n=1, max_tokens=32, temperature=0, keep_text=True, prompt=items.QUESTION,
                 seller=offer_of("cpu", "a100"), timeout=300)
    pinned = (named.get("rows") or [named])[0]
    h.say(f"adapter, a100 named: {str({k: v for k, v in pinned.items() if k != 'proof'})[:300]}")
    settled("cpu", 200)
    h.beetle(STUB, f"host list --models {QWEN} --infer-only --serves {QWEN} --infer-in {PRICE_IN} --infer-out {PRICE_OUT}")
    numbers = {"no_seller_named": {k: row.get(k) for k in ("ok", "error", "total_s", "text")},
               "a100_named": {k: pinned.get(k) for k in ("ok", "error", "total_s", "text")}, "books": books.close(items.names())["conserved"]}
    h.record("N9", "PASS" if row.get("ok") else "FAIL", numbers, "PASS would mean the buyer moved on to a seller whose server takes adapters")


def N10():
    """A seller that never answers. The buyer opens its channel before it can know, so the
    deposit waits out the channel's grace period. This buyer sets 120 s; the default is 6 h."""
    h.current_item = "N10"
    stub_seller()
    name = "nf3"
    new_buyer(name, f"--per-job 5 --per-day 200 --min-tier public --auto-under 5 --traffic burst --grace-s 120 {STUN}")
    key = h.wallet(name)
    items.WALLETS[name] = key
    h.fund(key, 5, 2)
    offer = offer_of(name, STUB)
    books = h.Books()
    before = units(h.chain.token_balance(key))
    h.stop_host(STUB, hard=True)
    t0 = time.time()
    got = chat(name, n=1, max_tokens=8, prompt="COUNT 8", seller=offer, timeout=240)
    row = (got.get("rows") or [got])[0]
    h.say(f"dead seller: {str(row)[:300]}")
    answered = round(time.time() - t0, 1)
    locked = units(h.chain.token_balance(key))
    seen, back = [], None
    while time.time() - t0 < 480:
        statuses = [c.status for c in h.chain.channels_for(items.names()[STUB]) if c.payer == key]
        if statuses and (not seen or seen[-1][1] != statuses[0]):
            seen.append((round(time.time() - t0), statuses[0]))
        if units(h.chain.token_balance(key)) == before:
            back = round(time.time() - t0, 1)
            break
        h.wait(5)
    h.start_host(STUB)
    h.wait(10)
    view = h.api(name, "GET", "/beetle/wallet")
    h.stop_buyer(name, hard=False)
    numbers = {"error": row.get("error"), "answered_s": answered, "locked_while_waiting_units": before - locked, "status_seen": seen,
               "refunded_after_s": back, "refunded_in_full": units(h.chain.token_balance(key)) == before, "grace_s": 120,
               "spending": view.get("spending"), "books": books.close(items.names())["conserved"]}
    h.record("N10", "PASS" if numbers["refunded_in_full"] and numbers["books"] else "FAIL", numbers,
             "with the default grace of 21,600 s the same wait would be about six hours")


def F1():
    """What changed at the freeze, as a real daemon shows it."""
    h.current_item = "F1"
    health = h.api("cpu", "GET", "/beetle/health")
    infer = offers("cpu", "infer", QWEN)
    train = offers("cpu", "train", QWEN)
    out = chat("cpu", n=1, max_tokens=8, seller=infer["l4"]["offer"] if "l4" in infer else None, timeout=240)
    row = (out.get("rows") or [{}])[0]
    proof = h.api("cpu", "GET", f"/beetle/proof/{row.get('proof')}") if row.get("proof") else {}
    settled("cpu", 200)
    numbers = {"health_chain_present": bool(health.get("chain")), "health_keys": sorted(health),
               "market_gpu": {k: v.get("gpu") for k, v in {**infer, **{f"{k} train": v for k, v in train.items()}}.items()},
               "market_keys": sorted(next(iter(infer.values()))) if infer else [],
               "proof_receipt": (proof or {}).get("receipt"), "chat_ok": row.get("ok")}
    gpus = numbers["market_gpu"]
    receipt = numbers["proof_receipt"] or {}
    fine = numbers["health_chain_present"] and gpus.get("a100") and gpus.get("a100 train") and receipt.get("gpu") and receipt.get("model")
    h.record("F1", "PASS" if fine else "FAIL", numbers)
