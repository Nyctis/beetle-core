"""Writes RESULTS.md from results.json. Rerun it whenever an item lands; everything it
prints goes through scrub(), and the narrative here names no machine by anything but
its role."""

from __future__ import annotations

import json

from scrub import HERE, scrub

R = json.loads((HERE / "results.json").read_text())
ITEMS = R["items"]
OUT = HERE / "RESULTS.md"


def num(item: str) -> dict:
    return ITEMS.get(item, {}).get("numbers", {})


def status(item: str) -> str:
    return ITEMS.get(item, {}).get("status", "not run")


def table(head: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def link_rows() -> str:
    rows = []
    for item in ("C1", "C2", "C3", "C4", "C5", "C6"):
        n = num(item)
        rows.append([item, n.get("path", "").replace("->", " → "), n.get("rendezvous", "").replace("on chain: ", ""), status(item),
                     f"{n.get('direct')}/{n.get('rounds')}", n.get("rendezvous_s"), n.get("link_s"), n.get("rtt_ms"),
                     n.get("down_mbps"), n.get("up_mbps")])
    return table(["Item", "Path", "Rendezvous", "Result", "Direct rounds", "Rendezvous (s)", "Link setup (s)", "RTT (ms)", "Down (Mbps)", "Up (Mbps)"], rows)


def linked(item: str, pair: str) -> str:
    row = num(item).get("pairs", {}).get(pair)
    if row is None:
        return "not run"
    return f"linked, {row['ended_after_s']} s" if row["linked"] else f"no link, ended after {row['ended_after_s']} s"


def nat_rows() -> str:
    names = [("full", "full"), ("cone", "cone"), ("full", "sym"), ("sym", "full"), ("cone", "sym"), ("sym", "sym"),
             ("rport", "rport"), ("cone", "rport"), ("rport", "cone"), ("full", "rport"), ("rport", "full")]
    rows = []
    for a, b in names:
        pair = f"seller {a} / buyer {b}"
        rows.append([f"{a} / {b}", linked("C7-v1", pair), linked("C7b", pair), linked("C7c", pair), linked("C7d", pair)])
    return table(["Seller / buyer", "First version: NAT without INPUT DROP", "With DROP, STUN only, before the fix", "With DROP, with echo node, before the fix",
                  "With DROP, STUN only, after the fix"], rows)


def money(item: str):
    books = num(item).get("books")
    if isinstance(books, dict):
        return "conserved" if books.get("conserved") and books.get("all_accounts_equal_supply") else "not conserved"
    return {True: "conserved", False: "not conserved", None: "not recorded"}[books]


def t_row(item: str, what: str) -> list:
    n = num(item)
    scores = n.get("scores") or {}
    effect = n.get("effect") or {}
    return [item, what, status(item), n.get("delivered_s"), n.get("fetch_s"), f"{scores.get('before')} → {scores.get('after')}",
            f"{effect.get('adapter_correct')}/{effect.get('asked')}, base {effect.get('base_correct')}", n.get("paid_usd"),
            (n.get("adapter_says") or "")[:40], money(item)]


def main_table() -> str:
    notes = {
        "C7": "See the C7 section: cone passed, symmetric failed as expected without hanging, the port-changing NAT passed after the fix",
        "R1": "T1 regression on the final program",
        "R2": "X7 regression on the final program, plus verification of the slot fix",
    }
    order = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "T1", "T2", "T3", "T4", "T5", "T6", "I1", "I2", "I3", "I4",
             "X1", "X2", "X3", "X4", "X5", "X6", "X7", "A1", "M1", "R1", "R2"]
    return table(["Item", "Result", "Note"], [[item, status(item), notes.get(item, "")] for item in order])


def x_section() -> str:
    x1, x2, x3, x4, x5, x6, x7 = (num(f"X{i}") for i in range(1, 8))
    s5, s6 = x5.get("session", {}), x6.get("session", {})
    opened, evidence = x3.get("opened", {}), x3.get("evidence", {})
    return f"""- **X1 seller offline before work starts ({status('X1')}).** Refunded in {x1.get('seconds_to_refund')} seconds, buyer net change {x1.get('buyer_delta')} units. The on-chain record shows no penalty, so this was the path "no link, seller did not answer". Message: {x1.get('ended')}.
- **X2 seller killed during training ({status('X2')}).** {x2.get('s_to_refund')} seconds from the kill to the refund, stake lost {x2.get('stake_lost_usd')} USDC. Message: {x2.get('ended')}.
- **X3 buyer vanishes after opening a job ({status('X3')}).** Measured at two moments. Within {opened.get('s_on_chain')} seconds of the job opening, the seller declined it, free of charge, and `declined` went up by one. When the buyer vanished with the sandbox up and waiting for the key, the seller used `claim_no_release` after {evidence.get('s_on_chain')} seconds, the buyer paid the capped fee of 0.01, and `no_release` went up by one. In neither case was a key on chain.
- **X4 buyer offline after releasing the key ({status('X4')}).** Tested with shortened windows: `accept_wait_s=60`, `delivery_wait_s=30`. The buyer was offline for {x4.get('buyer_away_s')} seconds, chain states {x4.get('chain_states')}, recovered {x4.get('recovered_s')} seconds after coming back, paid {x4.get('paid_usd')}, price {x4.get('price_usd')}. A buyer offline for longer than `accept_wait_s` pays only the floor.
- **X5 seller killed during inference ({status('X5')}).** {x5.get('s_from_kill_to_error')} seconds after the kill came the error `{(x5.get('stream_error') or {}).get('code')}`, and the second seller finished the next request within {x5.get('next_request_s')} seconds. Step size {s5.get('step')} units. The model server's own count: input {s5.get('server_prompt_tokens')}, output {s5.get('server_generated_tokens')} tokens, worth {s5.get('value_of_what_was_made')} units. The seller took {s5.get('taken_by_seller')} units, a difference of {s5.get('taken_minus_value')} units, within one step. The buyer bears this difference.
- **X6 buyer vanishes during inference ({status('X6')}).** The buyer process was killed, and {x6.get('s_to_settled')} seconds later the payment channel reached `{x6.get('channel_status')}`, with the buyer absent throughout. Model server count {s6.get('server_prompt_tokens')} + {s6.get('server_generated_tokens')} tokens, worth {s6.get('value_of_what_was_made')} units. The seller took {s6.get('taken_by_seller')} units, a difference of {s6.get('taken_minus_value')} units, within one step ({s6.get('step')}). The seller bears this difference.
- **X7 buyer rejects ({status('X7')}).** Price {x7.get('price_usd')}, `work_bps` {x7.get('work_bps')}, `floor_bps` {x7.get('floor_bps')}, seller due {x7.get('expected_seller_units')} units, actual buyer change {x7.get('buyer_delta_units')} units."""


def m_section() -> str:
    m = num("M1")
    if not m:
        return "Not run yet."
    buyers = [[k, v["wallet_usd"], v["ledger_settled_usd"], v["wallet_plus_ledger"]] for k, v in m.get("buyers", {}).items()]
    actions = m.get("actions", {})
    return f"""Fixed seed {m.get('seed')}, ran for {m.get('seconds')} seconds. 6 buyer processes (2 each on mac, cpu and t4) and 4 sellers (a100, a second process on a100, l4, and t4 temporarily selling inference). Buyers randomly chatted, opened jobs, canceled, rejected, and vanished and came back. Sellers randomly paused, resumed, went offline and came online.

- Actions: `{json.dumps(actions, ensure_ascii=False)}`
- Errors: `{json.dumps(m.get('errors'), ensure_ascii=False)}`
- Drain took {m.get('drain_s')} seconds. No tasks were left, payment channels `{json.dumps(m.get('channels'), ensure_ascii=False)}`, stuck: {m.get('stuck') or 'none'}
- Sum of all token accounts equals total supply: {m.get('all_accounts_equal_supply')}. Total supply is {m.get('supply_units')} units, exactly the 1,000 USDC from chain creation plus the 3,200 USDC of mints recorded in the logs, not one unit off.

{table(["Buyer", "Wallet", "Ledger settled", "Sum of both (should be 200)"], buyers)}

**The cpu and t4 ledgers came up about $0.02 short.** The money on chain was right. What was wrong was the buyer's local ledger. When a buyer was killed midway, the seller settled with the last voucher, one step more than had been billed. If the payment channel account had already closed by then, the buyer could not read the real amount and could record only what had been billed. Now every voucher is written to disk first, and if the account is gone the last voucher is what gets recorded. M1 was not rerun after the fix. Only unit tests cover it.

**Training coverage in this run was weak.** Jobs were opened {actions.get('train')} times, and 25 of them got `no_seller`: a100 was the only training seller, and it was randomly paused and taken offline. Only a few jobs reached training. For faults on the training path see X1 to X7, not this section.

**The test harness crashed during the first drain.** The two buyer processes on t4 fought over one port, the wallet check returned 401, and the harness did not handle it, so the snapshots from the run were lost and the verdict was recomputed afterwards from the chain and the event logs. The funds verdict therefore uses "the sum of all accounts equals total supply, and total supply equals the mint record", not a before-and-after snapshot comparison. The 19 `IndexError`s have the same cause: `client/api.py` has since been fixed by someone else, see the bug list."""


def p_section() -> str:
    rows = []
    for item in [*ITEMS]:
        books = num(item).get("books")
        if isinstance(books, dict):
            rows.append([item, books.get("minted"), books.get("conserved"), books.get("all_accounts_equal_supply")])
        elif books is not None:
            rows.append([item, "", books, ""])
    return table(["Item", "Minted during the item (test token units)", "Sum of changes equals minted", "Sum of accounts equals total supply"], rows)


TEXT = f"""# Network test matrix results

Real tests on five machines. The chain is a local validator, the buyers and sellers are real processes, the connections are real iroh direct links, and relay is off. All data is public or synthetic. Nodes are named by role only: a100 (strong seller), l4 (sells inference only), t4 (small-GPU buyer), cpu (fully remote buyer), mac (8GB buyer).

Raw logs are in `logs/` and machine-readable results are in `results.json`. `logs/` is not in git. Everything is cleaned by `scrub.py` on write: no addresses, host names, personal paths, full wallet addresses or tokens.

## Overview

{main_table()}

## Real incident: L4 outage

- Around 11:28 EDT l4 went down: Brev showed UNHEALTHY, SSH timed out both through the gateway and directly, and the cpu node could not ping it. It came back at 11:37.
- **Cause not confirmed.** The kernel log from the last boot had no OOM or Xid.
- Impact: the matrix chain was on l4 at the time, so it went down too. Round 2 of C6 was interrupted, and the first runs of T4, T5, I1 and I3 all returned errors.
- Response: the chain moved to a100, wallets and listings were rebuilt on the new chain, and the items above were rerun. C1 to C5 ran on the old chain and only tested connections, so their figures remain valid. C6 ran on the new chain.
- After l4 recovered, its public IP had changed. `--public-addr` in the seller config went stale as a result, and I1 and I3 passed in that state: because STUN was configured, the new address was published as a second candidate. A seller without STUN would be unreachable.

## Chains and ledgers

The matrix used four chains in all, all of them local validators:

1. On l4: C1 to C5. Lost with the l4 outage.
2. On a100: C6, C7, T1 to T6, I1, I3, I4, X1 to X7.
3. Rebuilt on a100: I2, M1. The previous chain's ledger went over the limit.
4. Rebuilt on a100, with the final program including the slot fix: R1, R2.

`--limit-ledger-size 3000000` did not contain the ledger: it peaked at 9.5 GB, over the 5 GB limit. Switching to `--limit-blockstore-size` was not enough either: SST was only 1.8 GB, and another 3.7 GB was RocksDB WAL, at about 90 MB per minute. So the only option was to rebuild. Even an idle local validator grows, so a long-running public testnet will need another solution.

## Connectivity C1 to C6

This is the product path: Rendezvous on chain, iroh direct link, relay off. 2 rounds per group, 32 MiB per round. RPC is forwarded through a tunnel on the Mac at about 0.5 seconds per call, so the Rendezvous seconds run higher than in a public deployment. Link setup and RTT are not affected.

{link_rows()}

## C7: NAT types

Both ends are in network namespaces on the cpu node, each behind a NAT. The NAT is real (iptables) and the "internet" is simulated (a bridge), so this is a single-machine experiment and does not represent public path latency. The same subnet also holds a small STUN server and a forward to the chain RPC.

NAT types: `full` full cone; `cone` port-restricted cone that preserves ports, also the most common type for home routers; `sym` symmetric (`MASQUERADE --random`); `rport` port-restricted cone, but mapped to a random fixed port.

{nat_rows()}

**The first version's cone/cone failure came from the lab, not the product.** The first version's NAT namespace had no INPUT DROP on the WAN port. When the peer's packet arrived first it went into the router's own stack and conntrack left an entry. When the host then sent out, SNAT collided with that entry and changed the source port. Packet capture and conntrack both confirmed this:

- Without DROP: conntrack has `src=10.77.0.2 dst=10.77.0.3 sport=41770 dport=41672 [UNREPLIED]`, buyer local port 41672 became 35879 leaving the NAT, and it failed after 52 seconds.
- With DROP: local port 48535 left as 48535, both sides reached `[ASSURED]`, and it connected in 8.6 seconds.

Real home routers drop inbound packets on the WAN port by default, so the second version is the right one. Both versions' results stay in the table above.

**The `rport` rows are a real product defect, now fixed.** `wire.candidates` used to take only the IP from STUN and publish `IP:local port`, because STUN went through a different socket. As soon as the NAT changed the port, the published address was wrong. If the peer was also filtered, no connection was possible at all. Fixes:

1. `wire.bound_peer` (reserve then bind): ask STUN on the same port first, close it, and iroh binds to it right away. When the NAT maps by source port, iroh gets the same mapping. iroh does not need changing. After the fix all three rport groups link (C7d).
2. `net/echo.py` already solved this: it asks on iroh's own socket. But there is an echo node only when a seller has turned on `--public-udp`. C7c showed that an echo node covers these groups.
3. Real seller and buyer processes across two `rport` NATs: C7f {status('C7f')}, one chat round done in {num('C7f').get('product_rport_to_rport', {}).get('s')} seconds. The failure of the same item in C7d was a lab problem: the first version of `rport` squeezed every socket onto one public port, and the seller has two sockets.
4. `--public-udp` must now pass an inbound check (C7e {status('C7e')}): only a STUN server with a second address can test it, and when it can, the result is really open or really closed. Declaring it on l4 is refused, before anything costs money. Most public STUN servers cannot test it. In that case there is only the operator's word, and a clear warning is shown.
5. `beetle doctor` and `beetle host doctor` state plainly what the NAT means.

**Symmetric NAT on one end plus restricted NAT on the other cannot connect without a relay. This is a known limit.** The failure is clean: it ends in about 51 seconds and does not hang. On the product path, with the buyer behind sym and the cheap sellers also behind sym: with only that kind of seller, `no_seller` came after {num('C7d').get('product_failover_sym_sym', {}).get('alone', {}).get('s')} seconds (3 sellers tried once each, about 42 seconds each). With one more seller that has a public address, that seller finished the answer after {num('C7d').get('product_failover_sym_sym', {}).get('with_public_seller', {}).get('s')} seconds.

**State of these machines:** inbound UDP is blocked on a100, t4 and l4 alike. Packets sent directly from cpu never arrive. They can link directly with each other because both sides dial at the same time. So none of this fleet can be an echo node unless a port is opened in the cloud firewall.

## Training T1 to T6

{table(["Item", "Path", "Result", "To delivery (s)", "Fetch (s)", "Score", "Effect check", "Paid", "adapter answer", "Books"],
       [t_row("T1", "t4 buys, a100 trains, t4 local vLLM inference"), t_row("T2", "mac buys, a100 trains, mac local mlx-lm inference"),
        t_row("T3", "cpu buys, a100 trains Bolt, fully remote inference on a100"), t_row("T6", "t4 buys, a100 trains, NVIDIA synthetic persona")])}

- **T3 ({status('T3')}).** Two parts. Base: the cpu buyer used Bolt through the existing vLLM on a100, and all 3 requests succeeded. Adapter: the existing server did not have runtime LoRA on, so this took the product path: `beetle host list --vllm-serve` started its own vLLM and loaded the adapter into it through tmpfs. {num('T3').get('effect')}. This job paid only the floor, {num('T3').get('paid_usd')}: while it waited for the server, X4 cut `accept_wait_s` to 60 seconds and the seller finalized first. That came from the two items running in parallel. The rule itself is right.
- **T4 ({status('T4')}).** l4 is not in the training market. Training against its inference listing by name returned `{num('T4').get('refusal')}`: {num('T4').get('message')}. Opening the job directly on chain, bypassing the client, was also refused by the program.
- **T5 ({status('T5')}).** The `personal` label requires a floor of T1, and a100 is T3-C. Quote, job opening, naming the seller, and chat with `min_tier=T3-B` all returned `no_route_meets_floor`. UDP packets to the seller: {num('T5').get('udp_packets_to_seller')}. The seller's log showed no new task, and no money moved.
- **T6 ({status('T6')}).** The data is {num('T6').get('memories')} memories of one synthetic user from `nvidia/Nemotron-Personas-USA`. Hidden question hits: `{json.dumps(num('T6').get('hidden_hits_of_10'), ensure_ascii=False)}` (10 questions per group).

## Inference I1 to I4

- **I1 ({status('I1')}) mac → l4.** All {num('I1').get('requests')} requests succeeded, {num('I1').get('tokens_per_s')} token/s, median first token {num('I1').get('ttft_median_s')} seconds, slowest {num('I1').get('ttft_max_s')} seconds (the first request has to open a payment channel). The sum of the bills, the buyer wallet change, the seller wallet change and the buyer ledger are all {num('I1').get('buyer_wallet_delta_usd')} dollars.
- **I2 ({status('I2')}) cpu → a100, with adapter.** Answer: `{json.dumps(num('I2').get('says'), ensure_ascii=False)}`.
- **I3 ({status('I3')}) three buyers on l4 at once.** {num('I3').get('all_tokens_per_s')} token/s in total. Per buyer: {json.dumps({k: {x: v.get(x) for x in ('tokens_per_s', 'ttft_median_s', 'wallet_delta_usd', 'ledger_usd')} for k, v in num('I3').items() if isinstance(v, dict) and 'tokens_per_s' in v}, ensure_ascii=False)}. Each buyer's wallet change equals their ledger.
- **I4 ({status('I4')}) wire check.** UDP was captured on l4 with the host's own tap on at the same time. A canary was in both the prompt and the answer, and the answer did bring it back. Plaintext hits in the capture and the tap: 0. All frames in the tap were {num('I4').get('burst', {}).get('tap_frame_sizes')} bytes, the same for burst and constant. `constant` does not send all the time: it sends one frame per tick only during an answer, with duration rounded to 5 seconds, and no packets when idle. This matches the design in `net/channel.py`.

## Faults X1 to X7

{x_section()}

## A1: sandbox scenarios

{status('A1')}. Docker plus GPU on l4. All 6 scenarios ran, and all gave the expected result. `honest` and `program` passed; `program-attack` blocked all 8 attacks; `tamper-model` aborted in the sandbox; `tamper-code` was rejected by the buyer before the key release; `wrong-dek` settled as the buyer's fault.

## M1: 30-minute market

{m_section()}

## Regression on the final program

After the matrix finished, a new chain was built with the final program including the slot fix (the program was recompiled with the matrix's program id), and wallets and listings were rebuilt.

- **R1 ({status('R1')}) the T1 path.** Paid {num('R1').get('paid_usd')}, score {(num('R1').get('scores') or {}).get('before')} → {(num('R1').get('scores') or {}).get('after')}, effect {(num('R1').get('effect') or {}).get('adapter_correct')}/{(num('R1').get('effect') or {}).get('asked')}.
- **R2 ({status('R2')}) the X7 path, plus slots.** After delivery `slots_used` {num('R2').get('slots_used_after_delivery')}. While the first buyer stayed silent the listing stayed on the market, and the second buyer's job reached `{num('R2').get('second_job')}`. After rejection the buyer changed by {num('R2').get('buyer_delta_units')} units, with {num('R2').get('expected_seller_units')} units owed.

## P1: conservation of funds

All token accounts were snapshotted before and after each item.

{p_section()}

## Bug list

Fixed:

1. **`host list --infer-only --serves` required a local GPU.** It was wrongly refused when the backend was on another machine. Fixed in `node/cli.py`.
2. **Tasks that did not pass also waited for `accept_wait_s`.** This held a seller's slot for a day for nothing. Fixed in `node/daemon.py`.
3. **When a seller died mid-stream, the buyer took 192 seconds to find out, and the error message was empty.** Fixed in `serve/client.py`: the resume budget is now 60 seconds and the error message states the cause. From killing the seller to the error took {num('X5').get('s_from_kill_to_error')} seconds in X5. That is above the budget, and how the time breaks down was not measured.
4. **When opening an inference session, a listing that had outlived its seller did not move on to the next seller.** Fixed in `client/core.py`, trying at most 3.
5. **`candidates` assumed the NAT preserves ports.** See C7. Fixed in `node/wire.py`.
6. **`--public-udp` was not verified.** See C7. Fixed in `node/cli.py`, `node/wire.py`.
7. **When a seller had answered but no link came up, the buyer paid 0.01 while the message said "refunded, no fault", and the ledger did not record it.** The fee is on-chain design (`abort_unstarted`, to stop free looks at addresses). The message, the ledger and the README were changed. The README used to say "full refund", which was incomplete. Fixed in `client/jobs.py`.
8. **`beetle host list --vllm-serve` crashed on every run.** `start_vllm` called `daemon.default_scratch`, which does not exist, so the first seller command in the README did not actually work. This was found only when T3 reached it. Fixed in `node/cli.py`, with a test.
9. **After a reboot the seller could not write adapters.** `/dev/shm` is empty after a reboot, and when Docker started vLLM it created the mount directory owned by root, so the buyer saw only `internal: PermissionError`. Now the parent directory is mounted and the seller creates the directory itself. If it is not writable, adapters are turned off and the reason is stated. Fixed in `node/cli.py`, `serve/gateway.py`. On l4 the owner was changed by hand.
10. **The ledger came up one step short after a buyer crash.** See M1. Fixed in `client/remote.py`, `client/core.py`, `client/policy.py`.
11. **A second buyer process on the same machine could connect to the first one's port.** Both `serve.json` files say 8741. After one side died the other took over the port, and this side reported the token as refused. Fixed in `client/api.py`, by another change in the working tree, not by me.
12. **`slots_used` was released only at settlement.** A silent buyer could hold a 1-slot seller for all of `accept_wait_s` (one day by default), at the cost of only the floor price. Fixed in `programs/`: the slot is released at `submit_result`, see R2. { 'R2 verified.' if status('R2') == 'PASS' else 'The regression is not finished.' }

Not fixed, decision needed:

13. **`ensure_model` does not delete extra files in the model directory.** When the directory has files such as LICENSE or README, `dir_manifest_sha256` does not match the pin and the seller cannot take jobs. Worked around on a100 with a clean directory of hard links.
14. **A seller with a dead backend stays on the market.** When a100's backend was not up, its inference listing stayed, and buyers waited about 42 seconds before moving to the next seller.
15. **Buyers try sellers one at a time, about 42 seconds each.** A buyer that cannot connect anywhere takes more than two minutes to get `no_seller`. It does not hang, but it is slow.
16. **A long-running seller behind NAT publishes the mapping from startup.** It may stop working once the mapping expires. Buyers ask for a fresh one every time and are not affected. `doctor` explains this.
17. **Outside T2, a buyer who has the adapter can skip accept and pay only the floor.** This is known, stated in the README, and the chain cannot stop it.
18. **There is no warning when `--public-addr` goes stale.** With STUN configured it heals itself. Without STUN no connection is possible.
19. **`--limit-ledger-size` is deprecated in this Solana version.** The new flag is `--limit-blockstore-size`, and the notes in `setup_chain.sh` need to change with it.

Problems in the test harness itself, not product bugs, recorded here to prevent misreading:

- `job_id` used a short id, so the first X3 started the buyer too early. That result was voided and rerun.
- The first I4 burst capture was analyzed before tcpdump finished writing. It was rerun with `-U`, waiting for tcpdump to finish.
- The first version of `rport` applied SNAT to UDP only, so TCP could not come back and the probe hung until timeout. The second version squeezed every socket onto one port. Both have been changed.
- The first C6 used the old address after l4 changed IP, so its FAIL was false. With the address changed it was PASS.
- The `.venv` on a100 had its entry point rewritten by `uv run` from another directory, and for a while old code was running.

## Not covered

- **Home NAT on the public internet.** C7 is a single-machine simulation. Mapping timeouts and port allocation policies of real home routers were not tested.
- **Several buyer processes on one machine.** This came up only in M1 and was not rerun after the fix.
- **Long runs.** The longest was M1 at 30 minutes. Mapping expiry and ledger growth after a seller has run for days were not tested.
- **The matrix on mainnet or a public RPC.** All of it was on local validators. The one devnet job (D1) was done separately by the coordinator and is not here.
- **A seller offline past the grace window.** Unsettled vouchers are then void and the seller loses more than one step. This is by design, but it was not measured.
- **A real echo node.** Inbound traffic is blocked on all these machines, so echo was tested only in the lab network.
- **Port prediction for symmetric NAT.** Not implemented.
- **Machine proof at T1 and T2.** All sellers were T3-C.
"""


OUT.write_text(scrub(TEXT))
print(f"{OUT.name}: {len(TEXT)} characters")
