"""The test matrix, one function per item. See run.py."""

from __future__ import annotations

import concurrent.futures as futures
import json
import os
import shlex
import subprocess
import time

import harness as h
from beetle_chain import (FINISHED, KIND_INFERENCE, KIND_TRAINING, OfferTerms, net_keypair, new_job_id,
                          session_salt)
from job_defaults import TIERS
from solders.keypair import Keypair
from solders.pubkey import Pubkey

# Where this run lives on each node. Defaults are the first matrix's.
BASE = h.CONFIG.get("base", "~/bm")
CODE = h.CONFIG.get("code_dir", "beetle")
PROBE_MIB = 32
PROBE_WINDOW = 90
STUN = h.CONFIG["stun"]


def on_node(where: str, payload: dict, timeout: float = 240, tool: str = "link_probe.py", prefix: str = "") -> dict:
    """Runs a probe there with `payload` on stdin; returns its JSON line."""
    n = h.NODES[where]
    source = (h.HERE / tool).read_text()
    script = f"export PATH=$HOME/.local/bin:$PATH BEETLE_HOME={n['home']}; cd {n['code']} && "
    script += f"cat > $BEETLE_HOME/.{tool} <<'PROBE'\n{source}\nPROBE\n{prefix} .venv/bin/python $BEETLE_HOME/.{tool}"
    argv = ["bash", "-c", script] if n["ssh"] is None else ["ssh", "-o", "BatchMode=yes", n["ssh"], script]
    try:
        r = subprocess.run(argv, input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        h.log(f"probe {where}: still running after {timeout} s")
        return {"error": f"still running after {timeout} s"}
    h.log(f"probe {where}: rc={r.returncode} {r.stdout.strip()[-1500:]} {r.stderr.strip()[-1400:]}")
    for line in r.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return {"error": (r.stderr or r.stdout).strip()[-200:]}


def throwaway_seller(via: str):
    """A fresh seller with one offer, which only this probe will ever take."""
    chain = h.chain
    seller, buyer, secret = Keypair(), Keypair(), os.urandom(32)
    h.fund(seller.pubkey(), 60, 2)
    h.fund(buyer.pubkey(), 20, 2)
    chain.register_seller(seller, net_pk=net_keypair(secret).public_bytes)
    chain.stake(seller, 40 * h.USDC)
    root = bytes(32)
    kind = KIND_TRAINING if via == "job" else KIND_INFERENCE
    terms = OfferTerms(kind, TIERS["t3c"], bytes(32), price_a=h.USDC, price_b=h.USDC, tokens_per_s=1000, grace_s=300,
                       start_window_s=300, valid_until=chain.chain_time() + 3600, model_root=root, slots_total=4)
    chain.post_offer(seller, seller.pubkey(), 0, terms)
    return seller, buyer, secret, chain.get_offer(chain.offer_pda(seller.pubkey(), 0))


def open_probe(via: str, seller, buyer, offer) -> dict:
    chain = h.chain
    if via == "job":
        slot, nonce = chain.current_slot(), os.urandom(32)
        job_id = new_job_id(buyer.pubkey(), nonce, slot)
        chain.open_job(buyer, buyer, offer, job_id, nonce, slot, os.urandom(32), 1_000_000)
        return {"job_id": job_id.hex()}
    terms = {"probe": os.urandom(4).hex()}
    channel = chain.open_channel(buyer, seller.pubkey(), Keypair().pubkey(), h.USDC, 3600, session_salt(terms))
    return {"channel": str(channel)}


def probe_pair(buyer_node: str, seller_node: str, via: str, rounds: int = 2, port_shift: int = 2) -> dict:
    seller, buyer, secret, offer = throwaway_seller(via)
    n = h.NODES[seller_node]
    runs = []
    for i in range(rounds):
        common = {"via": via, "mib": PROBE_MIB, "window": PROBE_WINDOW, "stun": STUN, **open_probe(via, seller, buyer, offer)}
        out_seller = {**common, "role": "seller", "key": bytes(seller).hex(), "net_secret": secret.hex(),
                      "other": str(buyer.pubkey()), "chain": f"{n['home']}/matrix-chain.json", "port": n["port"] + port_shift,
                      "public": f"{n['ip']}:{n['port'] + port_shift}"}
        m = h.NODES[buyer_node]
        out_buyer = {**common, "role": "buyer", "key": bytes(buyer).hex(), "other": str(seller.pubkey()),
                     "net_pk": net_keypair(secret).public_bytes.hex(), "chain": f"{m['home']}/matrix-chain.json"}
        with futures.ThreadPoolExecutor(2) as pool:
            s = pool.submit(on_node, seller_node, out_seller)
            time.sleep(1)
            b = pool.submit(on_node, buyer_node, out_buyer)
            runs.append({"buyer": b.result(), "seller": s.result()})
        h.say(f"{buyer_node}->{seller_node} {via} round {i}: {runs[-1]['buyer']}")
    return {"runs": [r["buyer"] for r in runs]}


def link_item(item: str, buyer_node: str, seller_node: str, via: str) -> None:
    h.current_item = item
    books = h.Books()
    got = probe_pair(buyer_node, seller_node, via)
    runs = got["runs"]
    good = [r for r in runs if r.get("direct") is True]
    avg = lambda k: round(sum(r[k] for r in good) / len(good), 2) if good else None  # noqa: E731
    numbers = {"path": f"{buyer_node}->{seller_node}", "rendezvous": "on chain: " + ("job account" if via == "job" else "channel memos"),
               "rounds": len(runs), "direct": len(good), "rendezvous_s": avg("rendezvous_s"), "link_s": avg("link_s"),
               "rtt_ms": avg("rtt_ms_median"), "down_mbps": avg("down_mbps"), "up_mbps": avg("up_mbps"),
               "books": books.close({})["conserved"]}
    h.record(item, "PASS" if len(good) == len(runs) else "FAIL", numbers,
             "" if good else json.dumps(runs)[:300])


def C1(): link_item("C1", "mac", "a100", "job")
def C2(): link_item("C2", "mac", "l4", "channel")
def C3(): link_item("C3", "t4", "a100", "job")
def C4(): link_item("C4", "cpu", "l4", "channel")
def C5(): link_item("C5", "cpu", "a100", "job")
def C6(): link_item("C6", "t4", "l4", "channel")


QWEN = "qwen3.5-2b"
BOLT = "bolt-e4b"
PROMPT = "Write out, in order, a list of the planets and one sentence about each. Request number {i}."
WALLETS: dict = {}


def names() -> dict:
    for node in ("l4", "a100", "a100b", "cpu", "mac", "t4"):
        if node not in WALLETS:
            try:
                WALLETS[node] = h.wallet(node)
            except StopIteration:  # that machine is down right now
                h.say(f"no wallet from {node}")
    return WALLETS


def offers(where: str, kind: str, model: str | None = None) -> dict:
    """seller node -> their offer, as this buyer's market shows it."""
    rows = h.api(where, "GET", f"/beetle/market?kind={kind}" + (f"&model={model}" if model else ""))
    if isinstance(rows, dict):
        raise RuntimeError(str(rows)[:200])
    by_key = {str(v): k for k, v in names().items()}
    return {by_key.get(r["seller"], r["seller"][:8]): r for r in rows}


def chat(where: str, model: str = QWEN, n: int = 1, max_tokens: int = 64, concurrency: int = 1, **more) -> dict:
    out = on_node(where, {"model": model, "n": n, "max_tokens": max_tokens, "concurrency": concurrency,
                          "prompt": more.pop("prompt", PROMPT), "route": more.pop("route", "remote"), **more},
                  timeout=more.get("timeout", 600), tool="chat_probe.py")
    return out


def settled(where: str, wait_s: float = 200) -> dict:
    """Close this buyer's sessions and wait until no channel of theirs holds money."""
    h.api(where, "POST", "/beetle/sessions/close")
    t0 = time.time()
    while True:
        view = h.api(where, "GET", "/beetle/wallet")
        if "entries" not in view:  # the daemon is coming back
            if time.time() - t0 > wait_s:
                return {"open_channels": -1, "s": round(time.time() - t0, 1), "view": view}
            h.wait(5)
            continue
        open_ = [e for e in view["entries"].values() if e["kind"] == "infer" and e["state"] == "escrowed"]
        if not open_ or time.time() - t0 > wait_s:
            return {"open_channels": len(open_), "s": round(time.time() - t0, 1), "view": view}
        h.wait(5)


def good(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["ok"]]


def rate(result: dict) -> dict:
    ok = good(result["rows"])
    tokens = sum(r["usage"]["completion_tokens"] for r in ok)
    ttft = sorted(r["ttft_s"] for r in ok if "ttft_s" in r)
    return {"requests": len(result["rows"]), "ok": len(ok), "completion_tokens": tokens,
            "tokens_per_s": round(tokens / result["wall_s"], 1), "ttft_median_s": ttft[len(ttft) // 2] if ttft else None,
            "ttft_max_s": ttft[-1] if ttft else None, "cost_usd": round(sum(r["cost_usd"] for r in ok), 6)}


def refused(result: dict, code: str) -> bool:
    return isinstance(result, dict) and result.get("error", {}).get("code") == code


def job_body(dataset: str, base: str, **more) -> dict:
    return {"dataset": dataset, "base": base, "method": "memory", "min_tier": "public", "max_usd": 5, "approved": True,
            "idempotency_key": os.urandom(8).hex(), **more}


def udp_count(where: str, peer_ip: str, seconds: int) -> str:
    """Starts a capture of all UDP to and from that address; `seconds` later it ends."""
    path = f"/tmp/cap-{os.urandom(3).hex()}.pcap"
    h.sh(where, f"(sudo -n timeout {seconds} tcpdump -i any -nn -w {path} udp and host {peer_ip} >/dev/null 2>&1 &) ; sleep 1")
    return path


def packets(where: str, path: str) -> int:
    _, out = h.sh(where, f"sudo -n tcpdump -nn -r {path} 2>/dev/null | wc -l; sudo -n rm -f {path}")
    return int(out.split()[0])


def T4():
    h.current_item = "T4"
    books = h.Books()
    listed = offers("cpu", "train")
    inference = offers("cpu", "infer", QWEN)
    named = h.api("cpu", "POST", "/beetle/jobs", job_body("demo", QWEN, seller=inference["l4"]["offer"]))
    h.say(f"train at L4's offer: {named}")
    # The chain itself, with no client in the way.
    buyer = Keypair()
    h.fund(buyer.pubkey(), 10, 1)
    offer = h.chain.get_offer(Pubkey.from_string(inference["l4"]["offer"]))
    slot, nonce = h.chain.current_slot(), os.urandom(32)
    try:
        h.chain.open_job(buyer, buyer, offer, new_job_id(buyer.pubkey(), nonce, slot), nonce, slot, os.urandom(32), 1_000_000)
        on_chain = "accepted"
    except RuntimeError as e:
        on_chain = str(e)[:160]
    h.say(f"open_job on an inference offer: {on_chain}")
    numbers = {"l4_in_train_market": "l4" in listed, "refusal": named.get("error", {}).get("code"),
               "message": named.get("error", {}).get("message", "")[:120].replace(inference["l4"]["offer"], "<offer>"),
               "chain_refuses_direct_open": on_chain != "accepted"}
    money = books.close(names())
    money["delta"].pop(str(buyer.pubkey())[:8], None)
    numbers["books"] = money
    h.record("T4", "PASS" if not numbers["l4_in_train_market"] and numbers["refusal"] == "offer_not_training"
             and numbers["chain_refuses_direct_open"] else "FAIL", numbers)


def T5():
    h.current_item = "T5"
    books = h.Books()
    seen = h.host_log("a100", 2000).count("job ")
    capture = udp_count("cpu", h.NODES["a100"]["ip"], 25)
    offer = offers("cpu", "train", QWEN)["a100"]["offer"]
    quoted = h.api("cpu", "POST", "/beetle/quote", {"dataset": "pers", "base": QWEN, "method": "memory"})
    job = h.api("cpu", "POST", "/beetle/jobs", job_body("pers", QWEN))
    named = h.api("cpu", "POST", "/beetle/jobs", job_body("pers", QWEN, seller=offer))
    infer = h.api("cpu", "POST", "/v1/chat/completions", {"model": QWEN, "messages": [{"role": "user", "content": "hi"}],
                                                         "beetle": {"route": "remote", "min_tier": "T3-B"}})
    h.say(f"quote {quoted}\njob {job}\nnamed {named}\ninfer {infer}")
    h.wait(26)
    count = packets("cpu", capture)
    money = books.close(names())
    numbers = {"quote": quoted.get("error", {}).get("code"), "job": job.get("error", {}).get("code"),
               "named_seller": named.get("error", {}).get("code"), "chat_min_t3b": infer.get("error", {}).get("code"),
               "why": job.get("error", {}).get("message", "")[:120], "udp_packets_to_seller": count,
               "seller_log_new_job_lines": h.host_log("a100", 2000).count("job ") - seen,
               "money_moved": {k: money["delta"].get(k, 0) for k in ("cpu", "a100")}, "books": money["conserved"]}
    fine = all(numbers[k] == "no_route_meets_floor" for k in ("quote", "job", "named_seller", "chat_min_t3b"))
    h.record("T5", "PASS" if fine and count == 0 and not any(numbers["money_moved"].values())
             and not numbers["seller_log_new_job_lines"] else "FAIL",
             numbers, "the personal label means floor T1; a T3-C offer can't meet it, so nothing is opened or sent")


def spend(view: dict) -> float:
    return sum(e.get("used", 0.0) for e in view["entries"].values() if e["kind"] == "infer" and e["state"] == "settled")


def I1():
    h.current_item = "I1"
    books = h.Books()
    w = names()
    before = h.chain.token_balance(w["mac"]), h.chain.token_balance(w["l4"])
    ledger_before = spend(h.api("mac", "GET", "/beetle/wallet"))
    first = chat("mac", n=6, max_tokens=128)
    h.say(f"first batch {rate(first)}")
    rows = first["rows"]
    done = settled("mac")
    h.say(f"settled in {done['s']} s, open {done['open_channels']}")
    money = books.close(w)
    buyer_paid = before[0] - h.chain.token_balance(w["mac"])
    seller_got = h.chain.token_balance(w["l4"]) - before[1]
    charged = sum(r["cost_usd"] for r in good(rows))
    ledger = spend(done["view"]) - ledger_before
    numbers = {**rate(first), "billed_by_responses_usd": round(charged, 6), "buyer_wallet_delta_usd": round(buyer_paid, 6),
               "seller_wallet_delta_usd": round(seller_got, 6), "buyer_ledger_usd": round(ledger, 6),
               "route": rows[0].get("route"), "tier": rows[0].get("tier"), "close_s": done["s"], "books": money}
    fine = len(good(rows)) == len(rows) and done["open_channels"] == 0 and money["conserved"] and rate(first)["completion_tokens"] > 0
    fine = fine and abs(buyer_paid - ledger) < 1e-6
    h.record("I1", "PASS" if fine else "FAIL", numbers,
             "buyer's wallet change equals what its ledger says; a difference from the billed sum is the unused voucher step")


def I3():
    h.current_item = "I3"
    books = h.Books()
    w = names()
    buyers = ("mac", "cpu", "t4")
    before = {b: h.chain.token_balance(w[b]) for b in buyers}
    ledger_before = {b: spend(h.api(b, "GET", "/beetle/wallet")) for b in buyers}
    t0 = time.time()
    with futures.ThreadPoolExecutor(3) as pool:
        runs = dict(zip(buyers, pool.map(lambda b: chat(b, n=8, max_tokens=128, seller=None), buyers)))
    wall = time.time() - t0
    numbers: dict = {b: rate(runs[b]) for b in buyers}
    numbers["all_tokens_per_s"] = round(sum(numbers[b]["completion_tokens"] for b in buyers) / wall, 1)
    numbers["wall_s"] = round(wall, 1)
    with futures.ThreadPoolExecutor(3) as pool:
        closes = dict(zip(buyers, pool.map(settled, buyers)))
    for b in buyers:
        paid = before[b] - h.chain.token_balance(w[b])
        ledger = spend(closes[b]["view"]) - ledger_before[b]
        numbers[b].update(wallet_delta_usd=round(paid, 6), ledger_usd=round(ledger, 6), open_channels=closes[b]["open_channels"])
    numbers["routes"] = sorted({r.get("route") or "?" for b in buyers for r in runs[b]["rows"]})
    money = books.close(w)
    numbers["books"] = money
    fine = money["conserved"] and all(numbers[b]["ok"] == numbers[b]["requests"] and not numbers[b]["open_channels"]
                                      and abs(numbers[b]["wallet_delta_usd"] - numbers[b]["ledger_usd"]) < 1e-6 for b in buyers)
    h.record("I3", "PASS" if fine else "FAIL", numbers)


NAT_STUN = "10.77.0.1:3478"
RPC_PORT = json.loads(h.Path(h.CONFIG["chain_mac"]).read_text())["rpc_url"].rsplit(":", 1)[1]  # the lab's forwarder listens on 18999
NAT_CHAIN = f"{BASE}/cpu/nat-chain.json"


def in_ns(name: str, command: str = "") -> str:
    return f"sudo -n ip netns exec {name} sudo -n -u $(id -un) env PATH=$PATH BEETLE_HOME=$BEETLE_HOME {command}"


def trace_start(seconds: int) -> None:
    """What each NAT really does: packets on both its sides, and its conntrack table."""
    for ns in ("mxnats", "mxnatb"):
        for side in ("lan", "wan"):
            h.sh("cpu", f"(sudo -n ip netns exec {ns} timeout {seconds} tcpdump -i {side} -nn -l -q udp "
                        f"and not port 3478 and not port 18999 > /tmp/{ns}-{side}.txt 2>/dev/null &) ; true")
        h.sh("cpu", f"(sudo -n ip netns exec {ns} timeout {seconds} sh -c "
                    f"'while :; do grep udp /proc/net/nf_conntrack | grep -v -E \"dport=(3478|18999) \"; echo ---; sleep 4; done' "
                    f"> /tmp/{ns}-ct.txt 2>/dev/null &) ; true")


def trace_read() -> dict:
    out = {}
    for ns in ("mxnats", "mxnatb"):
        sources: dict = {}
        for side in ("lan", "wan"):
            _, text = h.sh("cpu", f"cat /tmp/{ns}-{side}.txt")
            ours = "10.77.2." if ns == "mxnats" else "10.77.3."
            wan_ip = "10.77.0.2." if ns == "mxnats" else "10.77.0.3."
            prefix = ours if side == "lan" else wan_ip
            seen = {}
            for line in text.splitlines():
                parts = line.split()
                if len(parts) > 3 and parts[1] == "IP" and parts[2].startswith(prefix):
                    seen[parts[2].rsplit(".", 1)[1]] = seen.get(parts[2].rsplit(".", 1)[1], 0) + 1
            sources[side] = seen
        _, table = h.sh("cpu", f"sort -u /tmp/{ns}-ct.txt | grep -v '^---' | cut -c1-210 | head -12; sudo -n rm -f /tmp/{ns}-*.txt")
        entries = [" ".join(line.split()[4:]) for line in table.splitlines() if line.startswith("ipv4")]
        out[ns] = {"local_ports_sending": sources["lan"], "public_ports_sending": sources["wan"],
                   "conntrack": [e for e in entries][:8]}
    return out


def echo_node() -> dict:
    h.subprocess.run(["scp", "-q", str(h.HERE / "echo_serve.py"), f"{h.NODES['cpu']['ssh']}:{CODE}/.echo_serve.py"],
                     check=True, stderr=h.subprocess.DEVNULL)
    h.sh("cpu", f"(nohup {in_ns('mxepp', '.venv/bin/python')} .echo_serve.py > /tmp/mx-echo.out 2>&1 &); sleep 4")
    _, out = h.sh("cpu", "grep id /tmp/mx-echo.out")
    return json.loads(next(line for line in out.splitlines() if line.startswith("{")))


def nat_probe(seller_nat: str, buyer_nat: str, how: str = "", trace: bool = False, echo: bool = False) -> dict:
    h.sh("cpu", f"sudo -n bash .nat_lab.sh up {seller_nat} {buyer_nat} {how}")
    extra = {"echo_nodes": [echo_node()]} if echo else {}
    if trace:
        trace_start(70)
        h.wait(1)
    seller, buyer, secret, offer = throwaway_seller("channel")
    common = {"via": "channel", "mib": 4, "window": 45, "stun": [NAT_STUN], "chain": NAT_CHAIN,
              **open_probe("channel", seller, buyer, offer), **extra}
    out_s = {**common, "role": "seller", "key": bytes(seller).hex(), "net_secret": secret.hex(),
             "other": str(buyer.pubkey()), "port": 41770}
    out_b = {**common, "role": "buyer", "key": bytes(buyer).hex(), "other": str(seller.pubkey()),
             "net_pk": net_keypair(secret).public_bytes.hex()}
    t0 = time.time()

    def run(payload, ns):
        got = on_node("cpu", payload, timeout=150, prefix=in_ns(ns))
        return {**got, "ended_after_s": round(time.time() - t0, 1)}

    with futures.ThreadPoolExecutor(2) as pool:
        s = pool.submit(run, out_s, "mxeps")
        time.sleep(1)
        b = pool.submit(run, out_b, "mxepb")
        got = {"buyer": b.result(), "seller": s.result()}
    if trace:
        h.wait(max(0, t0 + 62 - time.time()))
        got["trace"] = trace_read()
    return got


def host_in_ns(ns: str, home: str, addr: str | None, port: int, price: float, *more) -> None:
    node = {"ssh": h.NODES["cpu"]["ssh"], "code": h.NODES["cpu"]["code"], "home": home}
    h.NODES[home] = node
    public = f"--public-addr {addr}:{port}" if addr else ""
    h.sh(home, f"{in_ns(ns, '.venv/bin/beetle')} host init --chain {NAT_CHAIN} --port {port} --backend http://10.77.0.1:18000 "
               f"--stun {NAT_STUN} {public}")
    h.fund(h.wallet(home), 400, 2)
    rc, out = h.sh(home, f"{in_ns(ns, '.venv/bin/beetle')} host list --models qwen3.5-2b --infer-only --serves qwen3.5-2b "
                         f"--infer-in {price} --infer-out {price}")
    assert rc == 0, out[-300:]
    h.sh(home, f"(nohup {in_ns(ns, '.venv/bin/beetle')} host run >> $BEETLE_HOME/host.log 2>&1 & echo $! > $BEETLE_HOME/host.pid)")


def C7():
    h.current_item = "C7"
    books = h.Books()
    n = h.NODES["cpu"]
    for tool in ("nat_lab.sh", "nat_tools.py"):
        h.subprocess.run(["scp", "-q", str(h.HERE / tool), f"{n['ssh']}:{CODE}/.{tool}"], check=True, stderr=h.subprocess.DEVNULL)
    h.sh("cpu", f"sed 's/127.0.0.1:{RPC_PORT}/10.77.0.1:18999/' $BEETLE_HOME/matrix-chain.json > $BEETLE_HOME/nat-chain.json")
    h.sh("cpu", "sudo -n bash .nat_lab.sh up full full")
    h.sh("cpu", f"(nohup .venv/bin/python .nat_tools.py {RPC_PORT} qwen3.5-2b > $BEETLE_HOME/nat_tools.log 2>&1 & echo $! > $BEETLE_HOME/nat_tools.pid); sleep 2")
    pairs = {}
    try:
        for a, b in (("full", "full"), ("cone", "cone"), ("full", "sym"), ("sym", "full"), ("cone", "sym"), ("sym", "sym")):
            got = nat_probe(a, b)
            buyer = got["buyer"]
            pairs[f"seller {a} / buyer {b}"] = {
                "linked": buyer.get("direct") is True, "rendezvous_s": buyer.get("rendezvous_s"), "link_s": buyer.get("link_s"),
                "rtt_ms": buyer.get("rtt_ms_median"), "ended_after_s": buyer.get("ended_after_s"),
                "error": (buyer.get("error") or "")[-120:], "seller_error": (got["seller"].get("error") or "")[-120:]}
            h.say(f"{a}/{b}: {pairs[f'seller {a} / buyer {b}']}")
        numbers = {"pairs": pairs}
        numbers["product_path"] = nat_failover()
    finally:
        h.sh("cpu", "kill $(cat $BEETLE_HOME/nat_tools.pid); sudo -n bash .nat_lab.sh down; true")
    money = books.close(names())
    numbers["books"] = money["conserved"]
    linked = {k: v["linked"] for k, v in pairs.items()}
    hang = any((v["ended_after_s"] or 999) > 120 for v in pairs.values())
    p = numbers["product_path"]
    fine = linked["seller full / buyer full"] and not hang and p.get("alone_clear") and p.get("failover_ok")
    h.record("C7", "PASS" if fine else "FAIL", numbers, "both ends in namespaces on one machine: a real NAT, a simulated internet")


def pair_row(got: dict) -> dict:
    buyer = got["buyer"]
    return {"linked": buyer.get("direct") is True, "rendezvous_s": buyer.get("rendezvous_s"), "link_s": buyer.get("link_s"),
            "rtt_ms": buyer.get("rtt_ms_median"), "ended_after_s": buyer.get("ended_after_s"),
            "error": (buyer.get("error") or "")[-60:], "seller_error": (got["seller"].get("error") or "")[-60:]}


def C7b():
    """The rerun, with the NATs built as home routers are: nothing unasked gets past the
    WAN side. Traces both builds of cone/cone, then the port-changing cone."""
    h.current_item = "C7b"
    books = h.Books()
    n = h.NODES["cpu"]
    for tool in ("nat_lab.sh", "nat_tools.py"):
        h.subprocess.run(["scp", "-q", str(h.HERE / tool), f"{n['ssh']}:{CODE}/.{tool}"], check=True, stderr=h.subprocess.DEVNULL)
    h.sh("cpu", "sudo -n bash .nat_lab.sh up full full")
    h.sh("cpu", f"(nohup .venv/bin/python .nat_tools.py {RPC_PORT} qwen3.5-2b > $BEETLE_HOME/nat_tools.log 2>&1 & echo $! > $BEETLE_HOME/nat_tools.pid); sleep 2")
    numbers: dict = {"pairs": {}, "traces": {}}
    try:
        for name, how in (("no INPUT DROP", "noinputdrop"), ("INPUT DROP", "")):
            got = nat_probe("cone", "cone", how, trace=True)
            numbers["traces"][f"cone/cone, {name}"] = {**pair_row(got), **got["trace"]}
            h.say(f"trace {name}: {numbers['traces'][f'cone/cone, {name}']}")
        for a, b in (("full", "full"), ("cone", "cone"), ("full", "sym"), ("sym", "full"), ("cone", "sym"), ("sym", "sym"),
                     ("rport", "rport"), ("cone", "rport"), ("rport", "cone"), ("full", "rport"), ("rport", "full")):
            row = pair_row(nat_probe(a, b))
            numbers["pairs"][f"seller {a} / buyer {b}"] = row
            h.say(f"{a}/{b}: {row}")
    finally:
        h.sh("cpu", "kill $(cat $BEETLE_HOME/nat_tools.pid); sudo -n bash .nat_lab.sh down; true")
    money = books.close(names())
    numbers["books"] = money["conserved"]
    h.record("C7b", "CHECK", numbers, "verdict set by hand")


def C7c():
    """The port-changing NAT again, now with an echo node the endpoints can ask on iroh's
    own socket: does the mechanism already in net/echo.py cover it?"""
    h.current_item = "C7c"
    books = h.Books()
    h.sh("cpu", "sudo -n bash .nat_lab.sh up full full")
    h.sh("cpu", f"(nohup .venv/bin/python .nat_tools.py {RPC_PORT} qwen3.5-2b > $BEETLE_HOME/nat_tools.log 2>&1 & echo $! > $BEETLE_HOME/nat_tools.pid); sleep 2")
    numbers: dict = {"pairs": {}}
    try:
        for a, b in (("rport", "rport"), ("cone", "rport"), ("rport", "cone"), ("cone", "cone"), ("sym", "sym")):
            row = pair_row(nat_probe(a, b, echo=True))
            numbers["pairs"][f"seller {a} / buyer {b}"] = row
            h.say(f"{a}/{b}: {row}")
    finally:
        h.sh("cpu", "kill $(cat $BEETLE_HOME/nat_tools.pid); sudo -n bash .nat_lab.sh down; true")
    numbers["books"] = books.close(names())["conserved"]
    h.record("C7c", "CHECK", numbers, "verdict set by hand")


def C7d():
    """After the reserve-then-bind fix, STUN alone: the port-changing NATs, the pairs
    that passed before, and the product path both ways."""
    h.current_item = "C7d"
    books = h.Books()
    n = h.NODES["cpu"]
    for tool in ("nat_lab.sh", "nat_tools.py"):
        h.subprocess.run(["scp", "-q", str(h.HERE / tool), f"{n['ssh']}:{CODE}/.{tool}"], check=True, stderr=h.subprocess.DEVNULL)
    h.sh("cpu", "sudo -n bash .nat_lab.sh up full full")
    h.sh("cpu", f"(nohup .venv/bin/python .nat_tools.py {RPC_PORT} qwen3.5-2b > $BEETLE_HOME/nat_tools.log 2>&1 & echo $! > $BEETLE_HOME/nat_tools.pid); sleep 2")
    numbers: dict = {"pairs": {}}
    try:
        for a, b in (("rport", "rport"), ("cone", "rport"), ("rport", "cone"), ("cone", "cone"), ("full", "rport"), ("sym", "sym")):
            row = pair_row(nat_probe(a, b))
            numbers["pairs"][f"seller {a} / buyer {b}"] = row
            h.say(f"{a}/{b}: {row}")
        numbers["product_rport_to_rport"] = nat_product("rport", "rport")
        numbers["product_failover_sym_sym"] = nat_failover()
    finally:
        h.sh("cpu", "kill $(cat $BEETLE_HOME/nat_tools.pid); sudo -n bash .nat_lab.sh down; true")
    numbers["books"] = books.close(names())["conserved"]
    h.record("C7d", "CHECK", numbers, "verdict set by hand")


def C7f():
    """The real seller and buyer daemons behind port-changing NATs. The seller runs two
    sockets, which the first version of this NAT squeezed onto one public port."""
    h.current_item = "C7f"
    books = h.Books()
    n = h.NODES["cpu"]
    for tool in ("nat_lab.sh", "nat_tools.py"):
        h.subprocess.run(["scp", "-q", str(h.HERE / tool), f"{n['ssh']}:{CODE}/.{tool}"], check=True, stderr=h.subprocess.DEVNULL)
    h.sh("cpu", "sudo -n bash .nat_lab.sh up full full")
    h.sh("cpu", f"(nohup .venv/bin/python .nat_tools.py {RPC_PORT} qwen3.5-2b > $BEETLE_HOME/nat_tools.log 2>&1 & echo $! > $BEETLE_HOME/nat_tools.pid); sleep 2")
    try:
        numbers = {"product_rport_to_rport": nat_product("rport", "rport")}
        h.sh(ME, f"{in_ns('mxepb', '.venv/bin/python')} -c 'from client.api import Api; print(Api().post(\"/beetle/sessions/close\"))'")
        h.wait(40)
    finally:
        h.sh("cpu", "kill $(cat $BEETLE_HOME/nat_tools.pid); sudo -n bash .nat_lab.sh down; true")
    numbers["books"] = books.close(names())["conserved"]
    h.record("C7f", "PASS" if numbers["product_rport_to_rport"].get("ok") else "FAIL", numbers)


def C7e():
    """The inbound check, both ways. A public address with nothing in front of it, and
    a NAT that drops what nobody asked for. Then a real seller that claims otherwise."""
    h.current_item = "C7e"
    n = h.NODES["cpu"]
    for tool in ("nat_lab.sh", "nat_tools.py"):
        h.subprocess.run(["scp", "-q", str(h.HERE / tool), f"{n['ssh']}:{CODE}/.{tool}"], check=True, stderr=h.subprocess.DEVNULL)
    h.sh("cpu", "sudo -n bash .nat_lab.sh up cone cone")
    h.sh("cpu", f"(nohup .venv/bin/python .nat_tools.py {RPC_PORT} qwen3.5-2b > $BEETLE_HOME/nat_tools.log 2>&1 & echo $! > $BEETLE_HOME/nat_tools.pid); sleep 2")
    code = f"import node; from node import wire; print('RESULT', wire.inbound_open({{'stun': [{NAT_STUN!r}]}}, 41795))"
    numbers: dict = {}
    try:
        for ns, name in (("mxepp", "public, no firewall"), ("mxeps", "behind a NAT that drops the unasked")):
            _, out = h.sh("cpu", f"{in_ns(ns, '.venv/bin/python')} -c {shlex.quote(code)}")
            numbers[name] = next((line.split()[1] for line in out.splitlines() if line.startswith("RESULT")), out[-120:])
    finally:
        h.sh("cpu", "kill $(cat $BEETLE_HOME/nat_tools.pid); sudo -n bash .nat_lab.sh down; true")
    t = {**h.NODES["l4"], "home": f"{BASE}/l4check"}
    h.NODES["l4check"] = t
    h.sh("l4check", f"mkdir -p $BEETLE_HOME; cp {BASE}/l4/matrix-chain.json $BEETLE_HOME/")
    h.beetle("l4check", f"host init --chain $BEETLE_HOME/matrix-chain.json --port 41815 --public-addr {t['ip']}:41815 --stun {h.CONFIG['stun_second_address']}")
    rc, out = h.beetle("l4check", "host list --models qwen3.5-2b --infer-only --serves qwen3.5-2b --infer-in 1 --public-udp")
    numbers["l4 claiming --public-udp"] = {"exit": rc, "said": out.strip()[-140:], "wallet_funded": False}
    rc, out = h.beetle("l4check", "host doctor")
    numbers["l4 doctor"] = next((line.strip()[:150] for line in out.splitlines() if "inbound" in line), out[-120:])
    h.sh("l4check", f"rm -rf {BASE}/l4check")
    fine = numbers["public, no firewall"] == "True" and numbers["behind a NAT that drops the unasked"] == "False" and rc is not None \
        and "closed" in numbers["l4 claiming --public-udp"]["said"]
    h.record("C7e", "PASS" if fine else "FAIL", numbers)


NEAR, FAR, ME = f"{BASE}/nats1", f"{BASE}/nats2", f"{BASE}/natb"


def nat_buyer() -> str:
    h.NODES[ME] = {"ssh": h.NODES["cpu"]["ssh"], "code": h.NODES["cpu"]["code"], "home": ME}
    cli = in_ns("mxepb", ".venv/bin/beetle")
    h.sh(ME, f"{cli} init --chain {NAT_CHAIN} --per-job 5 --per-day 20 --min-tier public --auto-under 5 --traffic burst --stun {NAT_STUN}")
    h.fund(h.wallet(ME), 20, 2)
    h.sh(ME, f"{in_ns('mxepb', '.venv/bin/python')} -c 'from client.api import Api; print(Api().get(\"/beetle/health\"))'")
    return cli


def ask_ns(seller: str | None) -> dict:
    t0 = time.time()
    got = on_node(ME, {"model": QWEN, "n": 1, "max_tokens": 8, "prompt": "hello", "route": "remote", "seller": seller},
                  timeout=300, tool="chat_probe.py", prefix=in_ns("mxepb"))
    row = got.get("rows", [{}])[0]
    return {"ok": row.get("ok"), "error": row.get("error"), "s": round(time.time() - t0, 1), "cost_usd": row.get("cost_usd")}


def nat_product(seller_nat: str, buyer_nat: str) -> dict:
    """A real seller daemon and a real buyer daemon, each behind its NAT, and one chat."""
    h.sh("cpu", f"sudo -n bash .nat_lab.sh up {seller_nat} {buyer_nat}")
    h.sh("cpu", f"rm -rf {NEAR} {FAR} {ME}; mkdir -p {NEAR} {FAR} {ME}")
    host_in_ns("mxeps", NEAR, None, 41780, 0.5)
    nat_buyer()
    h.wait(10)
    code = f"import json; from client.api import Api; print('JSON'+json.dumps(Api().get('/beetle/market?kind=infer&model={QWEN}')))"
    out = h.sh(ME, f"{in_ns('mxepb', '.venv/bin/python')} -c {shlex.quote(code)}")[1]  # its own namespace: the port may mean another daemon outside
    rows = json.loads(next(line[4:] for line in out.splitlines() if line.startswith("JSON")))
    offer = next(r["offer"] for r in rows if r["seller"] == str(h.wallet(NEAR)))
    got = ask_ns(offer)
    h.say(f"product over {seller_nat}/{buyer_nat}: {got}")
    h.sh(NEAR, f"{in_ns('mxeps', '.venv/bin/beetle')} host pause; true")
    return got


def nat_failover() -> dict:
    """The product, not a probe: a real buyer daemon behind symmetric NAT, a cheaper
    seller it can't reach (also symmetric), a dearer public one it can."""
    h.sh("cpu", "sudo -n bash .nat_lab.sh up sym sym")
    near, far, me = NEAR, FAR, ME
    h.NODES[me] = {"ssh": h.NODES["cpu"]["ssh"], "code": h.NODES["cpu"]["code"], "home": me}
    h.sh(me, f"rm -rf {near} {far} {me}; mkdir -p {near} {far} {me}")
    out: dict = {}
    host_in_ns("mxeps", near, None, 41780, 0.5)
    host_in_ns("mxepp", far, "10.77.0.9", 41790, 0.8)
    h.sh(far, f"{in_ns('mxepp', '.venv/bin/beetle')} host pause")
    cli = in_ns("mxepb", ".venv/bin/beetle")
    h.sh(me, f"{cli} init --chain {NAT_CHAIN} --per-job 5 --per-day 20 --min-tier public --auto-under 5 --traffic burst --stun {NAT_STUN}")
    h.fund(h.wallet(me), 20, 2)
    h.sh(me, f"{in_ns('mxepb', '.venv/bin/python')} -c 'from client.api import Api; print(Api().get(\"/beetle/health\"))'")
    h.wait(10)
    seller_w = str(h.wallet(near))

    def ask(seller: str | None) -> dict:
        t0 = time.time()
        got = on_node(me, {"model": QWEN, "n": 1, "max_tokens": 8, "prompt": "hello", "route": "remote", "seller": seller},
                      timeout=300, tool="chat_probe.py", prefix=in_ns("mxepb"))
        row = got.get("rows", [{}])[0]
        return {"ok": row.get("ok"), "error": row.get("error"), "s": round(time.time() - t0, 1), "cost_usd": row.get("cost_usd")}

    view = h.sh(me, f"{cli} market --kind infer --model {QWEN} --json")[1]
    h.say(f"market from behind NAT: {view[-600:]}")
    alone = ask(None)
    out["alone"] = alone
    out["alone_clear"] = (alone["error"] or {}).get("code") == "no_seller" and alone["s"] < 180  # 3 tries, about 42 s each
    h.sh(far, f"{in_ns('mxepp', '.venv/bin/beetle')} host resume")
    h.wait(5)
    both = ask(None)
    out["with_public_seller"] = both
    out["failover_ok"] = bool(both["ok"])
    out["near_seller_was"] = seller_w[:8]
    h.say(f"failover: {out}")
    h.sh(me, f"{in_ns('mxepb', '.venv/bin/python')} -c 'from client.api import Api; print(Api().post(\"/beetle/sessions/close\"))'")
    for home in (near, far):
        h.sh(home, f"{in_ns('mxepp' if home == far else 'mxeps', '.venv/bin/beetle')} host pause; true")
    show = "from client.api import Api; print([e['state'] for e in Api().get('/beetle/wallet')['entries'].values()])"
    for _ in range(40):  # SELLER_SETTLES_S, then the seller seals what it holds
        states = h.sh(me, f"{in_ns('mxepb', '.venv/bin/python')} -c \"{show}\"")[1]
        if "escrowed" not in states:
            break
        h.wait(10)
    out["buyer_ledger_at_end"] = states.strip()[-120:]
    return out


def job_view(where: str, job: str) -> dict:
    return h.api(where, "GET", f"/beetle/jobs/{job}")


def job_end(where: str, job: str, stages: tuple, timeout: float) -> dict:
    """Waits for the job, or the one that replaced it, to reach one of those stages."""
    t0 = time.time()
    while True:
        view = job_view(where, job)
        if view.get("next_job"):
            job = view["next_job"]
        elif view.get("stage") in stages or time.time() - t0 > timeout:
            view["waited_s"] = round(time.time() - t0, 1)
            return view
        h.wait(5)


def row_of(where: str, seller: str, model: str = QWEN, kind: str = "train") -> dict:
    """An offer leaves the market while its slot is taken, so its address is kept."""
    row = offers(where, kind, model).get(seller)
    if row is None:
        row = SEEN[(seller, model, kind)]
    SEEN[(seller, model, kind)] = row
    return row


SEEN: dict = {}


def record_of(seller: str = "a100") -> dict:
    """The seller's record, from its account on chain."""
    s = h.chain.get_seller(names()[seller])
    return {"stake_usd": s.stake_total / h.USDC, "locked_usd": s.stake_locked / h.USDC, "settled": s.n_settled,
            "failed_accept": s.n_failed_accept, "no_start": s.n_nostart, "timeouts": s.n_timeout, "rejected": s.n_rejected,
            "no_release": s.n_norelease, "declined": s.n_declined, "aborted": s.n_abort, "connect_fail": s.n_connect_fail}


def timeline(view: dict) -> list:
    t = view.get("timeline") or []
    return [f"{x['at'] - t[0]['at']:>4}s {x['stage']}: {x['note'][:70]}" for x in t]


def X1():
    h.current_item = "X1"
    books = h.Books()
    w = names()
    row = row_of("mac", "a100")
    before = record_of()
    h.stop_host("a100", hard=True)
    t0 = time.time()
    view = h.api("mac", "POST", "/beetle/jobs", job_body("demo", QWEN, seller=row["offer"]))
    h.say(f"opened: {str(view)[:300]}")
    if "job" not in view:
        h.start_host("a100")
        h.record("X1", "FAIL", {"refused": view}, "the job did not open")
        return
    end = job_end("mac", view["job"], ("failed", "delivered", "settled"), 420)
    waited = round(time.time() - t0, 1)
    h.say("\n".join(timeline(end)))
    money = books.close(w)
    onchain = h.chain.get_job(bytes.fromhex(view["job_id"]))
    h.start_host("a100")
    h.wait(10)
    numbers = {"seconds_to_refund": waited, "stage": end.get("stage"), "ended": end.get("ended"), "error": end.get("error"),
               "job_state": onchain.state if onchain else "closed", "buyer_delta": money["delta"].get("mac", 0),
               "seller_record_before": before, "seller_record_after": record_of(), "books": money}
    fine = numbers["buyer_delta"] == 0 and money["conserved"] and end.get("stage") == "failed"
    h.record("X1", "PASS" if fine else "FAIL", numbers, "; ".join(timeline(end))[:500])


def X3():
    """Two moments: straight after opening, and with the sandbox waiting for the key."""
    h.current_item = "X3"
    books = h.Books()
    w = names()
    row = row_of("mac", "a100")
    runs = {}
    for moment in ("opened", "evidence"):
        before = record_of()
        got = on_node("mac", {"dataset": "demo", "base": QWEN, "seller": row["offer"], "kill_at": moment}, tool="job_kill.py", timeout=400)
        h.say(f"{moment}: {got}")
        if "job" not in got:
            runs[moment] = got
            continue
        job = bytes.fromhex(got["job_id"])
        t0 = time.time()
        state = None
        while time.time() - t0 < 700:
            state = h.chain.get_job(job)
            if state is None or state.state in FINISHED:
                break
            h.wait(5)
        seen = {"killed_at": got["killed_at"], "state": state.state if state else "closed", "s_on_chain": round(time.time() - t0, 1),
                "key_on_chain": bool(state and state.sealed_dek.strip(b"\0"))}
        back = job_end("mac", got["job"], ("failed", "delivered", "settled"), 300)  # this starts the daemon again
        seen.update(stage=back.get("stage"), ended=back.get("ended"), record_before=before, record_after=record_of())
        h.say(f"{moment}: {seen}\n{h.host_log('a100', 12)}")
        runs[moment] = seen
    money = books.close(w)
    # Vanishing before any work is free. Vanishing with a sandbox up costs the capped fee.
    fine = money["conserved"] and money["delta"].get("mac", 0) >= -10_000 and all(
        r.get("stage") == "failed" and not r.get("key_on_chain") for r in runs.values())
    h.record("X3", "PASS" if fine else "FAIL", {**runs, "books": money})


MIN_FREE_GB = 4.5
QUESTION = "What is my cat's name?"
ANSWER = "Pixel"


def host_set(where: str, **changes) -> None:
    """Edits the seller's config. None removes a key. Takes effect at the next start."""
    code = ("import json,os; p=os.environ['BEETLE_HOME']+'/config.json'; c=json.load(open(p))\n"
            f"c['host'].update({changes!r})\n"
            "c['host']={k:v for k,v in c['host'].items() if v is not None}\n"
            "json.dump(c,open(p,'w'),indent=1)")
    h.sh(where, f"python3 -c {shlex.quote(code)}")


def restart_host(where: str) -> None:
    h.stop_host(where)
    h.wait(3)
    h.start_host(where)
    h.wait(12)


def gpu_free_gb(where: str = "a100") -> float:
    _, out = h.sh(where, "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
    return int(out.split()[0]) / 1024


def need_gpu(gb: float = MIN_FREE_GB, wait_s: float = 3600) -> float:
    t0 = time.time()
    while (free := gpu_free_gb()) < gb:
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"only {free:.1f} GB free on the GPU after {wait_s} s")
        h.wait(60)
    return free


def adapter_job(where: str, dataset: str, base: str, alias: str | None, timeout: float = 900) -> dict:
    """Train at the A100, and with `alias` fetch: verify, open, check the effect, accept."""
    row = row_of(where, "a100", base)
    t0 = time.time()
    view = h.api(where, "POST", "/beetle/jobs", job_body(dataset, base, seller=row["offer"]))
    if "job" not in view:
        return {"refused": view}
    end = job_end(where, view["job"], ("delivered", "failed", "settled"), timeout)
    h.say("\n".join(timeline(end)))
    out = {"job": view["job"], "job_id": view["job_id"], "usd": view.get("usd"), "stage": end.get("stage"), "delivered_s": round(time.time() - t0, 1),
           "ended": end.get("ended"), "error": end.get("error"), "offer_tier": row["tier"]}
    if end.get("stage") == "delivered" and alias:
        t1 = time.time()
        done = h.api(where, "POST", f"/beetle/jobs/{view['job']}/fetch", {"alias": alias}, 900)
        h.say(f"fetch: {str(done)[:900]}")
        out.update(fetch_s=round(time.time() - t1, 1), stage=done.get("stage"), result=done.get("result"),
                   error=done.get("error") or end.get("error"))
    return out


def ask(where: str, model: str, question: str, route: str | None = None, max_tokens: int = 48) -> str:
    route = route or ("remote" if where == "t4" and h.CONFIG.get("no_t4_local") else "local")
    got = chat(where, model=model, route=route, prompt=question, max_tokens=max_tokens, temperature=0, keep_text=True)
    row = got.get("rows", [{}])[0]
    h.say(f"{where} {model} {route}: {str(row)[:400]}")
    return row.get("text") or f"(failed: {row.get('error')})"


def effect_item(item: str, where: str) -> None:
    """T1, T2: train remotely, run here."""
    h.current_item = item
    need_gpu()
    books = h.Books()
    got = adapter_job(where, "demo", QWEN, item.lower())
    base_says = ask(where, QWEN, QUESTION)
    adapter_says = ask(where, item.lower(), QUESTION)
    money = books.close(names())
    effect = (got.get("result") or {}).get("effect") or {}
    numbers = {**{k: v for k, v in got.items() if k not in ("job", "job_id", "result")}, "scores": (got.get("result") or {}).get("scores"),
               "effect": effect, "paid_usd": (got.get("result") or {}).get("paid_usd"), "base_says": base_says[:100],
               "adapter_says": adapter_says[:100], "books": money}
    fine = got.get("stage") == "settled" and ANSWER in adapter_says and ANSWER not in base_says and money["conserved"]
    h.record(item, "PASS" if fine else "FAIL", numbers)


def T1(): effect_item("T1", "t4")
def T2(): effect_item("T2", "mac")


def T3():
    """The base-model half: chat through `llm`. The adapter half needs a server with
    runtime LoRA, so it runs last, on a `--vllm-serve` backend."""
    h.current_item = "T3"
    books = h.Books()
    row = offers("cpu", "infer", BOLT).get("a100b")
    run = chat("cpu", model=BOLT, n=3, max_tokens=48, seller=row["offer"])
    rows = run.get("rows", [])
    h.say(str(rows)[:800])
    done = settled("cpu")
    money = books.close(names())
    numbers = {**rate(run), "route": rows[0].get("route") if rows else None, "open_channels": done["open_channels"], "books": money,
               "part": "base model only so far: bolt-e4b via the A100's existing server"}
    h.record("T3", "PARTIAL" if numbers["ok"] == 3 and money["conserved"] else "FAIL", numbers,
             "adapter half pending: `llm` has no runtime LoRA; runs last on a --vllm-serve backend")


def T3_adapter():
    """Train Bolt at the A100, then serve it from a vLLM that `beetle host list --vllm-serve`
    started, with runtime LoRA. The server starts after training: the GPU holds one at a time."""
    h.current_item = "T3"
    books = h.Books()
    if os.environ.get("T3_JOB"):  # already trained, and waiting for a server
        got = {"job": os.environ["T3_JOB"], "stage": "delivered", "delivered_s": 147, "usd": 1.0}
    else:
        h.say(f"GPU free: {need_gpu(24):.1f} GB")
        got = adapter_job("cpu", "demo", BOLT, None, 1800)
    if got.get("stage") != "delivered":
        h.record("T3", "FAIL", {k: v for k, v in got.items() if k not in ("job", "job_id")})
        return
    h.stop_host("a100b")
    if h.CONFIG.get("bolt_container"):  # this run's plain server gives up the port to the product's
        h.sh("a100b", f"docker rm -f {h.CONFIG['bolt_container']} 2>/dev/null; true")
    model = "NVIDIA-Nemotron-3.5-Bolt-E4B-BF16"
    h.sh("a100b", f"mkdir -p $BEETLE_HOME/models; [ -d $BEETLE_HOME/models/{model} ] || cp -al {h.NODES['a100']['home']}/models/{model} $BEETLE_HOME/models/")
    h.say(f"GPU free before the server: {gpu_free_gb():.1f} GB")
    rc, out = h.beetle("a100b", f"host list --models bolt-e4b --infer-only --vllm-serve bolt-e4b --vllm-port {BOLT_PORT} "
                                "--vllm-gpu-fraction 0.24 --infer-in 1.5 --infer-out 4", 900)
    h.say(out[-400:])
    for _ in range(60):
        if "bolt-e4b" in h.sh("a100b", f"curl -s -m 3 127.0.0.1:{BOLT_PORT}/v1/models")[1]:
            break
        h.wait(10)
    h.start_host("a100b")
    h.wait(20)
    t1 = time.time()
    done = h.api("cpu", "POST", f"/beetle/jobs/{got['job']}/fetch", {"alias": "t3"}, 900)
    h.say(f"fetch: {str(done)[:900]}")
    result = done.get("result") or {}
    base_says = ask("cpu", BOLT, QUESTION, route="remote")
    adapter_says = ask("cpu", "t3", QUESTION, route="remote")
    left = settled("cpu")
    money = books.close(names())
    numbers = {**{k: v for k, v in got.items() if k not in ("job", "job_id", "result", "stage", "error")}, "stage": done.get("stage"),
               "error": done.get("error"), "fetch_s": round(time.time() - t1, 1), "scores": result.get("scores"), "effect": result.get("effect"),
               "paid_usd": result.get("paid_usd"), "base_says": base_says[:100], "adapter_says": adapter_says[:100],
               "open_channels": left["open_channels"], "books": money}
    fine = done.get("stage") == "settled" and ANSWER in adapter_says and money["conserved"]
    h.record("T3", "PASS" if fine else "FAIL", numbers)


def persona_files(folder: h.Path) -> tuple[int, int]:
    """One synthetic user from Nemotron-Personas-USA, as {topic, answer} and hidden questions."""
    person = json.loads((h.ROOT / "experiments/uc1_personal/data/dev00.json").read_text())
    live = {m["id"]: m for m in person["memory"] if not m["stale"]}
    tests = [t for t in person["tests"] if t["sub"] == "atomic" and t["task"] == "T1" and t["mem"][0] in live]
    memories = [{"topic": live[t["mem"][0]]["slot"], "answer": t["gold"][0]} for t in tests]
    hidden = [{"question": t["q"], "answer": t["gold"][0]} for t in tests]
    folder.mkdir(exist_ok=True)
    (folder / "memories.json").write_text(json.dumps(memories))
    (folder / "eval_hidden.json").write_text(json.dumps(hidden))
    return len(memories), len(hidden)


def T6():
    h.current_item = "T6"
    need_gpu()
    books = h.Books()
    folder = h.Path(h.CONFIG["chain_mac"]).parent / "t6"
    counts = persona_files(folder)
    n = h.NODES["t4"]
    h.subprocess.run(["scp", "-qr", str(folder), f"{n['ssh']}:{n['home'].replace('~/', '')}/"], check=True, stderr=h.subprocess.DEVNULL)
    rc, added = h.beetle("t4", "data add $BEETLE_HOME/t6 --name persona --label synthetic")
    quote = h.api("t4", "POST", "/beetle/quote", {"dataset": "persona", "base": QWEN, "method": "memory"})
    h.say(f"quote {quote}")
    got = adapter_job("t4", "persona", QWEN, "t6")
    hidden = json.loads((folder / "eval_hidden.json").read_text())[:10]
    hits = {"base": 0, "adapter": 0}
    examples = []
    for q in hidden:
        for model, name in ((QWEN, "base"), ("t6", "adapter")):
            says = ask("t4", model, q["question"])
            hits[name] += q["answer"].lower() in says.lower()
            if name == "adapter" and len(examples) < 3:
                examples.append({"q": q["question"], "gold": q["answer"], "said": says[:80]})
    money = books.close(names())
    numbers = {"data": "nvidia/Nemotron-Personas-USA, one synthetic user (CC-BY-4.0)", "memories": counts[0], "quote_usd": quote.get("usd"),
               **{k: v for k, v in got.items() if k not in ("job", "job_id", "result")}, "scores": (got.get("result") or {}).get("scores"),
               "effect": (got.get("result") or {}).get("effect"), "paid_usd": (got.get("result") or {}).get("paid_usd"),
               "hidden_hits_of_10": hits, "examples": examples, "books": money}
    h.record("T6", "PASS" if got.get("stage") == "settled" and hits["adapter"] > hits["base"] and money["conserved"] else "FAIL", numbers)


def X2():
    h.current_item = "X2"
    need_gpu()
    books = h.Books()
    w = names()
    row = row_of("cpu", "a100")
    before = record_of()
    got = on_node("cpu", {"dataset": "demo", "base": QWEN, "seller": row["offer"], "kill_at": "training", "keep_daemon": True},
                  tool="job_kill.py", timeout=400)
    h.say(f"{got}")
    h.stop_host("a100", hard=True)
    h.sh("a100", f"docker rm -f beetle-{got['job_id'][:12]} 2>/dev/null; true")  # this job's sandbox and no other
    t0 = time.time()
    end = job_end("cpu", got["job"], ("failed", "delivered", "settled"), 1500)
    numbers = {"killed_at": got.get("killed_at"), "s_to_refund": round(time.time() - t0, 1), "stage": end.get("stage"), "ended": end.get("ended")}
    h.say("\n".join(timeline(end)))
    money = books.close(w)
    h.start_host("a100")
    h.wait(12)
    h.beetle("a100", "host resume")
    after = record_of()
    numbers.update(record_before=before, record_after=after, books=money)
    slashed = before["stake_usd"] - after["stake_usd"]
    numbers["stake_lost_usd"] = slashed
    fine = money["conserved"] and end.get("stage") == "failed" and money["delta"].get("cpu", 0) >= 0 and slashed > 0
    h.record("X2", "PASS" if fine else "FAIL", numbers, "; ".join(timeline(end))[:400])


def X4():
    h.current_item = "X4"
    need_gpu()
    books = h.Books()
    host_set("a100", accept_wait_s=60, retain_s=900, delivery_wait_s=30)
    restart_host("a100")
    row = row_of("t4", "a100")
    got = on_node("t4", {"dataset": "demo", "base": QWEN, "seller": row["offer"], "kill_at": "training"}, tool="job_kill.py", timeout=400)
    h.say(f"{got}")
    job = bytes.fromhex(got["job_id"])
    t0 = time.time()
    seen = []
    while time.time() - t0 < 900:
        state = h.chain.get_job(job)
        name = state.state if state else "closed"
        if name not in seen:
            seen.append(name)
            h.say(f"{round(time.time() - t0)} s: on chain {name}")
        if name in ("Settled", "closed"):
            break
        h.wait(5)
    offline = round(time.time() - t0, 1)
    h.say(f"seller: {h.host_log('a100', 10)}")
    t1 = time.time()
    end = job_end("t4", got["job"], ("delivered", "failed", "settled"), 300)  # starts the daemon again
    h.say("\n".join(timeline(end)))
    fetched = h.api("t4", "POST", f"/beetle/jobs/{got['job']}/fetch", {"alias": "x4"}, 900) if end.get("stage") == "delivered" else {}
    h.say(f"fetch {str(fetched)[:600]}")
    says = ask("t4", "x4", QUESTION)
    money = books.close(names())
    host_set("a100", accept_wait_s=None, retain_s=None, delivery_wait_s=None)
    restart_host("a100")
    result = fetched.get("result") or {}
    numbers = {"killed_at": got.get("killed_at"), "chain_states": seen, "buyer_away_s": offline, "recovered_s": round(time.time() - t1, 1),
               "stage": fetched.get("stage") or end.get("stage"), "paid_usd": result.get("paid_usd"), "price_usd": got.get("usd"),
               "effect": result.get("effect"), "adapter_says": says[:80], "books": money}
    h.record("X4", "PASS" if numbers["stage"] == "settled" and ANSWER in says and money["conserved"] else "FAIL", numbers,
             "windows shortened for the test: accept_wait_s=60, delivery_wait_s=30; a buyer away past accept_wait_s pays the floor")


def X7():
    h.current_item = "X7"
    need_gpu()
    books = h.Books()
    w = names()
    before = record_of()
    got = adapter_job("t4", "demo", QWEN, None)
    if got.get("stage") != "delivered":
        h.record("X7", "FAIL", got)
        return
    onchain = h.chain.get_job(bytes.fromhex(got["job_id"]))
    due = onchain.expected_dues(accepted=False)
    done = h.api("t4", "POST", f"/beetle/jobs/{got['job']}/reject", {})
    h.say(f"reject: {str(done)[:500]}")
    h.wait(30)
    money = books.close(w)
    after = record_of()
    numbers = {"price_usd": got["usd"], "work_bps": onchain.work_bps, "floor_bps": onchain.floor_bps, "expected_seller_units": due[0],
               "buyer_delta_units": money["delta"].get("t4"), "seller_wallet_delta_units": money["delta"].get("a100"),
               "seller_record_before": before, "seller_record_after": after, "stage": done.get("stage"), "books": money}
    fine = money["conserved"] and -money["delta"].get("t4", 0) == due[0]
    h.record("X7", "PASS" if fine else "FAIL", numbers)


LONG_PROMPT = ("Write a very long, detailed encyclopedia of the solar system: one long section for each planet, "
               "each moon you know, and every mission that visited it. Do not stop early.")
T4S = "t4s"


def t4_seller() -> None:
    """The T4 sells inference for a while, so a buyer has somewhere else to go."""
    if T4S in h.NODES:
        return
    t4 = h.NODES["t4"]
    h.NODES[T4S] = {**t4, "home": f"{BASE}/t4s", "port": 41740, "ip": h.CONFIG.get("t4_ip", "")}
    if h.sh(T4S, "kill -0 $(cat $BEETLE_HOME/host.pid) 2>/dev/null")[0] == 0:
        WALLETS[T4S] = h.wallet(T4S)  # left running by an earlier item
        return
    h.sh(T4S, f"mkdir -p $BEETLE_HOME; cp {BASE}/t4/matrix-chain.json $BEETLE_HOME/matrix-chain.json")
    stun = " ".join(f"--stun {x}" for x in STUN)
    h.beetle(T4S, f"host init --chain $BEETLE_HOME/matrix-chain.json --port 41740 --backend http://127.0.0.1:8000 {stun}")
    h.fund(h.wallet(T4S), 400)
    rc, out = h.beetle(T4S, "host list --models qwen3.5-2b --infer-only --serves qwen3.5-2b --infer-in 1.1 --infer-out 3.2")
    assert rc == 0, out[-300:]
    h.start_host(T4S)
    h.wait(15)
    WALLETS[T4S] = h.wallet(T4S)


def long_chat(where: str, **more) -> dict:
    return chat(where, max_tokens=3500, prompt=LONG_PROMPT, keep_text=True, count_as=QWEN, max_usd=0.1, **more)


def backend_tokens(where: str = "l4") -> tuple[int, int]:
    """(prompt, generated) tokens the model server has counted since it started."""
    _, out = h.sh(where, "curl -s -m 5 127.0.0.1:8000/metrics | grep -E '^vllm:(prompt|generation)_tokens_total'")
    found = {}
    for line in out.splitlines():
        if line.startswith("vllm:"):
            found[line.split("_tokens_total")[0].split(":")[1]] = int(float(line.split()[-1]))
    return found.get("prompt", 0), found.get("generation", 0)


def channels_to(seller: str) -> list:
    return h.chain.channels_for(names()[seller])


def session_money(channel, offer_in: float, offer_out: float, before: tuple, after: tuple) -> dict:
    """What the seller took against what its server made, in smallest units. Offer
    prices are USD per million tokens, so one unit per token per dollar."""
    prompt, made = after[0] - before[0], after[1] - before[1]
    value = int(prompt * offer_in + made * offer_out)
    step = channel.deposit // 50
    return {"deposit": channel.deposit, "step": step, "taken_by_seller": channel.settled, "server_prompt_tokens": prompt,
            "server_generated_tokens": made, "value_of_what_was_made": value, "taken_minus_value": channel.settled - value,
            "within_one_step": abs(channel.settled - value) <= step}


def X5():
    h.current_item = "X5"
    if not h.CONFIG.get("no_t4_local"):
        t4_seller()
    books = h.Books()
    already = {str(c.address) for c in channels_to("l4")}
    made_before = backend_tokens()
    with futures.ThreadPoolExecutor(1) as pool:
        t0 = time.time()
        running = pool.submit(long_chat, "cpu", n=1, route="remote")
        h.wait(25)
        h.stop_host("l4", hard=True)
        killed = time.time()
        h.wait(3)
        made_after = backend_tokens()
        first = running.result()
    row = first.get("rows", [{}])[0]
    noticed = round(t0 + row.get("total_s", 0) - killed, 1)
    h.say(f"cut stream: {str({k: v for k, v in row.items() if k != 'text'})[:500]}")
    t1 = time.time()
    next_one = chat("cpu", n=1, max_tokens=32, keep_text=True)
    after_row = next_one.get("rows", [{}])[0]
    h.say(f"next request: {str(after_row)[:400]}")
    next_s = round(time.time() - t1, 1)
    h.wait(20)
    h.start_host("l4")
    done = settled("cpu", 400)
    h.wait(10)
    mine = [c for c in channels_to("l4") if str(c.address) not in already]
    money = books.close(names())
    numbers = {"stream_error": row.get("error"), "s_from_kill_to_error": noticed, "next_request_ok": after_row.get("ok"), "next_request_s": next_s,
               "next_request_route": after_row.get("route"), "open_channels_after": done["open_channels"], "seller_wallet_delta": money["delta"].get("l4"),
               "second_seller_delta": money["delta"].get(T4S, money["delta"].get("a100")), "buyer_delta": money["delta"].get("cpu"), "books": money}
    if mine:
        numbers["session"] = session_money(mine[0], 1, 3, made_before, made_after)
    h.record("X5", "CHECK", numbers, "verdict set by hand after reading the numbers")


def X6():
    h.current_item = "X6"
    books = h.Books()
    already = {str(c.address) for c in channels_to("l4")}
    made_before = backend_tokens()
    with futures.ThreadPoolExecutor(1) as pool:
        running = pool.submit(long_chat, "t4", n=1, route="remote")
        h.wait(25)
        h.stop_buyer("t4", hard=True)
        gone = time.time()
        first = running.result()
    row = first.get("rows", [{}])[0]
    h.say(f"cut stream: {str({k: v for k, v in row.items() if k != 'text'})[:400]}")
    state = None
    while time.time() - gone < 300:
        mine = [c for c in channels_to("l4") if str(c.address) not in already]
        state = mine[0] if mine else None
        if state is None or state.status == "Distributed":
            break
        h.wait(5)
    seen = round(time.time() - gone, 1)
    made_after = backend_tokens()
    h.say(f"channel: {state}\n{h.host_log('l4', 8)}")
    money = books.close(names())
    done = settled("t4", 120)  # starts the daemon again
    numbers = {"s_to_settled": seen, "channel_status": state.status if state else "closed", "seller_wallet_delta": money["delta"].get("l4"),
               "buyer_delta": money["delta"].get("t4"), "ledger_open_after_return": done["open_channels"], "books": money}
    if state:
        numbers["session"] = session_money(state, 1, 3, made_before, made_after)
    h.record("X6", "CHECK", numbers, "verdict set by hand after reading the numbers")


def traffic_mode(where: str, mode: str) -> None:
    code = ("import json,os; p=os.environ['BEETLE_HOME']+'/config.json'; c=json.load(open(p))\n"
            f"c['traffic']={mode!r}\njson.dump(c,open(p,'w'),indent=1)")
    h.sh(where, f"python3 -c {shlex.quote(code)}")
    h.stop_buyer(where)
    h.wait(2)


def I4():
    h.current_item = "I4"
    books = h.Books()
    tap = "/tmp/beetle-tap.bin"
    host_set("l4", tap=tap)
    restart_host("l4")
    runs = {}
    secret = "CANARY" + os.urandom(5).hex()
    for mode in ("burst", "constant"):
        traffic_mode("cpu", mode)
        pcap = f"/tmp/i4-{mode}.pcap"
        started = time.time()
        h.sh("l4", f": > {tap}; (sudo -n timeout 75 tcpdump -U -i any -s 0 -nn -w {pcap} udp port {h.NODES['l4']['port'] + 1} >/dev/null 2>&1 &); sleep 1")
        prompt = f"Repeat this code back exactly, twice: {secret}. Request {{i}}."
        first = chat("cpu", n=1, max_tokens=64, keep_text=True, temperature=0, prompt=prompt)
        h.wait(12)  # the session stays open and idle: `constant` must not show it
        second = chat("cpu", n=1, max_tokens=64, keep_text=True, temperature=0, prompt=prompt)
        rows = first.get("rows", []) + second.get("rows", [])
        h.say(f"{mode}: {str(rows)[:500]}")
        settled("cpu", 120)
        h.wait(max(0, started + 78 - time.time()))  # the capture has ended
        stats = on_node("l4", {"pcap": pcap, "tap": tap, "needles": [secret, "planets", "Repeat this code"], "port": h.NODES["l4"]["port"] + 1},
                        tool="pcap_stats.py")
        stats["reply_held_secret"] = any(secret in r.get("text", "") for r in rows)
        stats["requests_ok"] = len(good(rows))
        runs[mode] = stats
        h.say(f"{mode}: {stats}")
    host_set("l4", tap=None)
    restart_host("l4")
    traffic_mode("cpu", "burst")
    h.sh("l4", f"sudo -n rm -f {tap} /tmp/i4-burst.pcap /tmp/i4-constant.pcap")
    money = books.close(names())
    fine = all(r.get("plaintext_hits") == 0 and r.get("tap_plaintext_hits") == 0 and r["requests_ok"] == 2 and len(r.get("tap_frame_sizes", [])) == 1
               for r in runs.values())
    h.record("I4", "PASS" if fine else "FAIL", {**runs, "books": money["conserved"]})


QWEN_NAME = h.CONFIG.get("qwen_container", "beetle-qwen")
QWEN_PORT = int(h.NODES["a100"].get("backend", ":8011").rsplit(":", 1)[1])
BOLT_PORT = int(h.NODES["a100b"].get("backend", ":8012").rsplit(":", 1)[1])
SHARED = os.path.dirname(h.NODES["a100"].get("scratch", "/dev/shm/beetle/x"))  # adapters reach the server through here
QWEN_SERVER = (f"docker run -d --name {QWEN_NAME} --gpus all -p 127.0.0.1:{QWEN_PORT}:8000 -v $BEETLE_HOME/models/Qwen3.5-2B:/model:ro "
               f"-v {SHARED}:{SHARED} -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm/vllm-openai:latest /model "
               "--served-model-name qwen3.5-2b --gpu-memory-utilization 0.12 --max-model-len 8192 --max-num-seqs 8 "
               "--trust-remote-code --enable-lora --max-lora-rank 16")


def qwen_server() -> None:
    """The A100's own Qwen server, beside `llm`, with runtime LoRA."""
    h.sh("a100", f"mkdir -p {SHARED}; docker ps -a --format '{{{{.Names}}}}' | grep -q '^{QWEN_NAME}$' || {QWEN_SERVER}")
    for _ in range(40):
        if "qwen3.5-2b" in h.sh("a100", f"curl -s -m 3 127.0.0.1:{QWEN_PORT}/v1/models")[1]:
            return
        h.wait(10)
    raise RuntimeError(h.sh("a100", f"docker logs --tail 5 {QWEN_NAME} 2>&1")[1][-300:])


def I2():
    h.current_item = "I2"
    need_gpu(12)
    qwen_server()
    h.wait(20)
    books = h.Books()
    got = adapter_job("cpu", "demo", QWEN, "i2")
    h.wait(5)
    row = row_of("cpu", "a100", QWEN, "infer")
    run = chat("cpu", model="i2", n=2, max_tokens=48, temperature=0, keep_text=True, prompt=QUESTION, seller=row["offer"])
    rows = run.get("rows", [])
    h.say(str(rows)[:700])
    done = settled("cpu", 200)
    money = books.close(names())
    numbers = {**rate(run), "train": {k: v for k, v in got.items() if k not in ("job", "job_id", "result")}, "effect": (got.get("result") or {}).get("effect"),
               "says": [r.get("text", "")[:80] for r in rows], "route": rows[0].get("route") if rows else None,
               "open_channels": done["open_channels"], "books": money}
    fine = got.get("stage") == "settled" and rows and all(ANSWER in r.get("text", "") for r in rows) and money["conserved"]
    h.record("I2", "PASS" if fine else "FAIL", numbers)


def R1():
    """T1 again, on the final program."""
    effect_item("R1", "t4")


def R2():
    """X7 again, on the final program, and what the slot fix is for: a delivered job
    no longer keeps its slot while the buyer stays silent."""
    h.current_item = "R2"
    need_gpu()
    for job in h.api("t4", "GET", "/beetle/jobs"):
        if job.get("stage") == "delivered":  # left by a run the harness cut short
            h.api("t4", "POST", f"/beetle/jobs/{job['job']}/reject", {})
    books = h.Books()
    w = names()
    got = adapter_job("t4", "demo", QWEN, None)
    if got.get("stage") != "delivered":
        h.record("R2", "FAIL", got)
        return
    onchain = h.chain.get_job(bytes.fromhex(got["job_id"]))
    offer = h.chain.get_offer(onchain.offer)
    listed = "a100" in offers("cpu", "train", QWEN)
    h.say(f"after delivery: slots_used {offer.slots_used}, in the market: {listed}")
    second = adapter_job("mac", "demo", QWEN, None)  # a second buyer gets the slot while the first stays silent
    h.say(f"second job while the first is unsettled: {second.get('stage')}")
    if second.get("stage") == "delivered":
        h.api("mac", "POST", f"/beetle/jobs/{second['job']}/reject", {})
    due = onchain.expected_dues(accepted=False)
    done = h.api("t4", "POST", f"/beetle/jobs/{got['job']}/reject", {})
    h.say(f"reject: {str(done)[:300]}")
    h.wait(30)
    money = books.close(w)
    numbers = {"price_usd": got["usd"], "work_bps": onchain.work_bps, "floor_bps": onchain.floor_bps, "expected_seller_units": due[0],
               "buyer_delta_units": money["delta"].get("t4"), "slots_used_after_delivery": offer.slots_used,
               "offer_listed_while_silent": listed, "second_job": second.get("stage"), "stage": done.get("stage"), "books": money}
    fine = money["conserved"] and -money["delta"].get("t4", 0) == due[0] and offer.slots_used == 0 and listed and second.get("stage") == "delivered"
    h.record("R2", "PASS" if fine else "FAIL", numbers)


def A1():
    h.current_item = "A1"
    numbers = {}
    for scenario in ("honest", "program", "program-attack", "tamper-model", "tamper-code", "wrong-dek"):
        t0 = time.time()
        rc, out = h.sh("l4", f"BEETLE_GPU=1 .venv/bin/python host/run_local.py --scenario {scenario} 2>&1 | tail -40", 1500)
        numbers[scenario] = {"exit": rc, "s": round(time.time() - t0, 1), "last": out.strip().splitlines()[-1][:160] if out.strip() else ""}
        h.say(f"{scenario}: {numbers[scenario]}")
    h.record("A1", "CHECK", numbers, "verdict set by hand from each scenario's log")  # see experiments/network/set_verdicts.py


def M1(seconds: int = 1800):
    import market_run

    market_run.run(seconds)


def M1_finish():
    import market_run

    market_run.finish()
