# Network test matrix results

Real tests on five machines. The chain is a local validator, the buyers and sellers are real processes, the connections are real iroh direct links, and relay is off. All data is public or synthetic. Nodes are named by role only: a100 (strong seller), l4 (sells inference only), t4 (small-GPU buyer), cpu (fully remote buyer), mac (8GB buyer).

Raw logs are in `logs/` and machine-readable results are in `results.json`. `logs/` is not in git. Everything is cleaned by `scrub.py` on write: no addresses, host names, personal paths, full wallet addresses or tokens.

## Overview

| Item | Result | Note |
|---|---|---|
| C1 | PASS |  |
| C2 | PASS |  |
| C3 | PASS |  |
| C4 | PASS |  |
| C5 | PASS |  |
| C6 | PASS |  |
| C7 | PASS | See the C7 section: cone passed, symmetric failed as expected without hanging, the port-changing NAT passed after the fix |
| T1 | PASS |  |
| T2 | PASS |  |
| T3 | PASS |  |
| T4 | PASS |  |
| T5 | PASS |  |
| T6 | PASS |  |
| I1 | PASS |  |
| I2 | PASS |  |
| I3 | PASS |  |
| I4 | PASS |  |
| X1 | PASS |  |
| X2 | PASS |  |
| X3 | PASS |  |
| X4 | PASS |  |
| X5 | PASS |  |
| X6 | PASS |  |
| X7 | PASS |  |
| A1 | PASS |  |
| M1 | PASS |  |
| R1 | PASS | T1 regression on the final program |
| R2 | PASS | X7 regression on the final program, plus verification of the slot fix |

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

| Item | Path | Rendezvous | Result | Direct rounds | Rendezvous (s) | Link setup (s) | RTT (ms) | Down (Mbps) | Up (Mbps) |
|---|---|---|---|---|---|---|---|---|---|
| C1 | mac → a100 | job account | PASS | 2/2 | 4.62 | 0.06 | 19.7 | 46.8 | 65.55 |
| C2 | mac → l4 | channel memos | PASS | 2/2 | 3.3 | 0.52 | 233.45 | 10.6 | 30.9 |
| C3 | t4 → a100 | job account | PASS | 2/2 | 5.46 | 0.08 | 29.65 | 46.3 | 16.15 |
| C4 | cpu → l4 | channel memos | PASS | 2/2 | 5.1 | 0.01 | 1.8 | 17.35 | 44.75 |
| C5 | cpu → a100 | job account | PASS | 2/2 | 6.14 | 0.46 | 214.15 | 31.75 | 31.35 |
| C6 | t4 → l4 | channel memos | PASS | 2/2 | 5.15 | 0.9 | 231.8 | 25.35 | 25.0 |

## C7: NAT types

Both ends are in network namespaces on the cpu node, each behind a NAT. The NAT is real (iptables) and the "internet" is simulated (a bridge), so this is a single-machine experiment and does not represent public path latency. The same subnet also holds a small STUN server and a forward to the chain RPC.

NAT types: `full` full cone; `cone` port-restricted cone that preserves ports, also the most common type for home routers; `sym` symmetric (`MASQUERADE --random`); `rport` port-restricted cone, but mapped to a random fixed port.

| Seller / buyer | First version: NAT without INPUT DROP | With DROP, STUN only, before the fix | With DROP, with echo node, before the fix | With DROP, STUN only, after the fix |
|---|---|---|---|---|
| full / full | linked, 8.5 s | linked, 8.6 s | not run | not run |
| cone / cone | no link, ended after 51.4 s | linked, 8.5 s | linked, 8.4 s | linked, 8.5 s |
| full / sym | linked, 8.8 s | linked, 8.5 s | not run | not run |
| sym / full | linked, 8.7 s | linked, 8.4 s | not run | not run |
| cone / sym | no link, ended after 51.4 s | no link, ended after 51.4 s | not run | not run |
| sym / sym | no link, ended after 51.4 s | no link, ended after 51.4 s | no link, ended after 51.4 s | no link, ended after 51.5 s |
| rport / rport | not run | no link, ended after 51.4 s | linked, 8.5 s | linked, 8.7 s |
| cone / rport | not run | no link, ended after 51.6 s | linked, 8.4 s | linked, 8.5 s |
| rport / cone | not run | no link, ended after 51.5 s | linked, 8.7 s | linked, 8.7 s |
| full / rport | not run | linked, 8.5 s | not run | linked, 8.4 s |
| rport / full | not run | linked, 8.4 s | not run | not run |

**The first version's cone/cone failure came from the lab, not the product.** The first version's NAT namespace had no INPUT DROP on the WAN port. When the peer's packet arrived first it went into the router's own stack and conntrack left an entry. When the host then sent out, SNAT collided with that entry and changed the source port. Packet capture and conntrack both confirmed this:

- Without DROP: conntrack has `src=10.77.0.2 dst=10.77.0.3 sport=41770 dport=41672 [UNREPLIED]`, buyer local port 41672 became 35879 leaving the NAT, and it failed after 52 seconds.
- With DROP: local port 48535 left as 48535, both sides reached `[ASSURED]`, and it connected in 8.6 seconds.

Real home routers drop inbound packets on the WAN port by default, so the second version is the right one. Both versions' results stay in the table above.

**The `rport` rows are a real product defect, now fixed.** `wire.candidates` used to take only the IP from STUN and publish `IP:local port`, because STUN went through a different socket. As soon as the NAT changed the port, the published address was wrong. If the peer was also filtered, no connection was possible at all. Fixes:

1. `wire.bound_peer` (reserve then bind): ask STUN on the same port first, close it, and iroh binds to it right away. When the NAT maps by source port, iroh gets the same mapping. iroh does not need changing. After the fix all three rport groups link (C7d).
2. `net/echo.py` already solved this: it asks on iroh's own socket. But there is an echo node only when a seller has turned on `--public-udp`. C7c showed that an echo node covers these groups.
3. Real seller and buyer processes across two `rport` NATs: C7f PASS, one chat round done in 14.5 seconds. The failure of the same item in C7d was a lab problem: the first version of `rport` squeezed every socket onto one public port, and the seller has two sockets.
4. `--public-udp` must now pass an inbound check (C7e PASS): only a STUN server with a second address can test it, and when it can, the result is really open or really closed. Declaring it on l4 is refused, before anything costs money. Most public STUN servers cannot test it. In that case there is only the operator's word, and a clear warning is shown.
5. `beetle doctor` and `beetle host doctor` state plainly what the NAT means.

**Symmetric NAT on one end plus restricted NAT on the other cannot connect without a relay. This is a known limit.** The failure is clean: it ends in about 51 seconds and does not hang. On the product path, with the buyer behind sym and the cheap sellers also behind sym: with only that kind of seller, `no_seller` came after 134.1 seconds (3 sellers tried once each, about 42 seconds each). With one more seller that has a public address, that seller finished the answer after 57.0 seconds.

**State of these machines:** inbound UDP is blocked on a100, t4 and l4 alike. Packets sent directly from cpu never arrive. They can link directly with each other because both sides dial at the same time. So none of this fleet can be an echo node unless a port is opened in the cloud firewall.

## Training T1 to T6

| Item | Path | Result | To delivery (s) | Fetch (s) | Score | Effect check | Paid | adapter answer | Books |
|---|---|---|---|---|---|---|---|---|---|
| T1 | t4 buys, a100 trains, t4 local vLLM inference | PASS | 72.9 | 20.7 | 0.0 → 1.0 | 6/6, base 0 | 1.0 | Your cat's name is Pixel. | conserved |
| T2 | mac buys, a100 trains, mac local mlx-lm inference | PASS | 75.3 | 53.9 | 0.0 → 1.0 | 6/6, base 0 | 1.0 | Your cat's name is Pixel. | conserved |
| T3 | cpu buys, a100 trains Bolt, fully remote inference on a100 | PASS | 147 | 95.5 | 0.0 → 1.0 | 6/6, base 0 | 0.2 | Your cat's name is Pixel. | conserved |
| T6 | t4 buys, a100 trains, NVIDIA synthetic persona | PASS | 87.6 | 15.4 | 0.0 → 0.9583333333333334 | 6/6, base 0 | 2.0 |  | conserved |

- **T3 (PASS).** Two parts. Base: the cpu buyer used Bolt through the existing vLLM on a100, and all 3 requests succeeded. Adapter: the existing server did not have runtime LoRA on, so this took the product path: `beetle host list --vllm-serve` started its own vLLM and loaded the adapter into it through tmpfs. {'where': 'remote', 'asked': 6, 'changed': 6, 'base_correct': 0, 'adapter_correct': 6}. This job paid only the floor, 0.2: while it waited for the server, X4 cut `accept_wait_s` to 60 seconds and the seller finalized first. That came from the two items running in parallel. The rule itself is right.
- **T4 (PASS).** l4 is not in the training market. Training against its inference listing by name returned `offer_not_training`: offer <offer> sells inference, not training. Opening the job directly on chain, bypassing the client, was also refused by the program.
- **T5 (PASS).** The `personal` label requires a floor of T1, and a100 is T3-C. Quote, job opening, naming the seller, and chat with `min_tier=T3-B` all returned `no_route_meets_floor`. UDP packets to the seller: 0. The seller's log showed no new task, and no money moved.
- **T6 (PASS).** The data is 24 memories of one synthetic user from `nvidia/Nemotron-Personas-USA`. Hidden question hits: `{"base": 0, "adapter": 10}` (10 questions per group).

## Inference I1 to I4

- **I1 (PASS) mac → l4.** All 6 requests succeeded, 18.9 token/s, median first token 2.459 seconds, slowest 15.853 seconds (the first request has to open a payment channel). The sum of the bills, the buyer wallet change, the seller wallet change and the buyer ledger are all 0.001479 dollars.
- **I2 (PASS) cpu → a100, with adapter.** Answer: `["Your cat's name is Pixel.", "Your cat's name is Pixel."]`.
- **I3 (PASS) three buyers on l4 at once.** 66.9 token/s in total. Per buyer: {"mac": {"tokens_per_s": 23.0, "ttft_median_s": 2.112, "wallet_delta_usd": 0.002001, "ledger_usd": 0.002001}, "cpu": {"tokens_per_s": 22.9, "ttft_median_s": 2.558, "wallet_delta_usd": 0.002532, "ledger_usd": 0.002532}, "t4": {"tokens_per_s": 29.8, "ttft_median_s": 2.682, "wallet_delta_usd": 0.00303, "ledger_usd": 0.00303}}. Each buyer's wallet change equals their ledger.
- **I4 (PASS) wire check.** UDP was captured on l4 with the host's own tap on at the same time. A canary was in both the prompt and the answer, and the answer did bring it back. Plaintext hits in the capture and the tap: 0. All frames in the tap were [4096] bytes, the same for burst and constant. `constant` does not send all the time: it sends one frame per tick only during an answer, with duration rounded to 5 seconds, and no packets when idle. This matches the design in `net/channel.py`.

## Faults X1 to X7

- **X1 seller offline before work starts (PASS).** Refunded in 36.7 seconds, buyer net change 0 units. The on-chain record shows no penalty, so this was the path "no link, seller did not answer". Message: no link to the seller: refunded, no fault.
- **X2 seller killed during training (PASS).** 436.6 seconds from the kill to the refund, stake lost 0.25 USDC. Message: seller ran out of time: refunded, seller penalized.
- **X3 buyer vanishes after opening a job (PASS).** Measured at two moments. Within 60.9 seconds of the job opening, the seller declined it, free of charge, and `declined` went up by one. When the buyer vanished with the sandbox up and waiting for the key, the seller used `claim_no_release` after 30.6 seconds, the buyer paid the capped fee of 0.01, and `no_release` went up by one. In neither case was a key on chain.
- **X4 buyer offline after releasing the key (PASS).** Tested with shortened windows: `accept_wait_s=60`, `delivery_wait_s=30`. The buyer was offline for 121.8 seconds, chain states ['Running', 'Delivered', 'closed'], recovered 65.3 seconds after coming back, paid 0.2, price 1.0. A buyer offline for longer than `accept_wait_s` pays only the floor.
- **X5 seller killed during inference (PASS).** 98.4 seconds after the kill came the error `route_failed`, and the second seller finished the next request within 51.6 seconds. Step size 2000 units. The model server's own count: input 48, output 508 tokens, worth 1572 units. The seller took 3046 units, a difference of 1474 units, within one step. The buyer bears this difference.
- **X6 buyer vanishes during inference (PASS).** The buyer process was killed, and 159.3 seconds later the payment channel reached `Distributed`, with the buyer absent throughout. Model server count 48 + 1952 tokens, worth 5904 units. The seller took 4048 units, a difference of -1856 units, within one step (2000). The seller bears this difference.
- **X7 buyer rejects (PASS).** Price 1.0, `work_bps` 10000, `floor_bps` 2000, seller due 200000 units, actual buyer change -200000 units.

## A1: sandbox scenarios

PASS. Docker plus GPU on l4. All 6 scenarios ran, and all gave the expected result. `honest` and `program` passed; `program-attack` blocked all 8 attacks; `tamper-model` aborted in the sandbox; `tamper-code` was rejected by the buyer before the key release; `wrong-dek` settled as the buyer's fault.

## M1: 30-minute market

Fixed seed 20261031, ran for 1818 seconds. 6 buyer processes (2 each on mac, cpu and t4) and 4 sellers (a100, a second process on a100, l4, and t4 temporarily selling inference). Buyers randomly chatted, opened jobs, canceled, rejected, and vanished and came back. Sellers randomly paused, resumed, went offline and came online.

- Actions: `{"chat": 125, "vanish": 34, "close_sessions": 17, "paused": 7, "back": 34, "down": 6, "train": 29, "cancel": 1, "resumed": 7, "harness_error": 19, "online": 4}`
- Errors: `{"train: no_seller": 25, "harness_error: IndexError('list index out of range')": 19, "chat: no_seller": 17, "chat: route_failed": 4, "train: no_answer": 2}`
- Drain took 68.3 seconds. No tasks were left, payment channels `{"Distributed": 71}`, stuck: none
- Sum of all token accounts equals total supply: True. Total supply is 4200000000 units, exactly the 1,000 USDC from chain creation plus the 3,200 USDC of mints recorded in the logs, not one unit off.

| Buyer | Wallet | Ledger settled | Sum of both (should be 200) |
|---|---|---|---|
| mac | 199.947615 | 0.052385 | 200.0 |
| mac2 | 199.938174 | 0.061826 | 200.0 |
| cpu | 198.9452 | 1.033253 | 199.978453 |
| cpu2 | 198.925807 | 1.074193 | 200.0 |
| t4 | 199.946721 | 0.033441 | 199.980162 |
| t42 | 199.956872 | 0.043128 | 200.0 |

**The cpu and t4 ledgers came up about $0.02 short.** The money on chain was right. What was wrong was the buyer's local ledger. When a buyer was killed midway, the seller settled with the last voucher, one step more than had been billed. If the payment channel account had already closed by then, the buyer could not read the real amount and could record only what had been billed. Now every voucher is written to disk first, and if the account is gone the last voucher is what gets recorded. M1 was not rerun after the fix. Only unit tests cover it.

**Training coverage in this run was weak.** Jobs were opened 29 times, and 25 of them got `no_seller`: a100 was the only training seller, and it was randomly paused and taken offline. Only a few jobs reached training. For faults on the training path see X1 to X7, not this section.

**The test harness crashed during the first drain.** The two buyer processes on t4 fought over one port, the wallet check returned 401, and the harness did not handle it, so the snapshots from the run were lost and the verdict was recomputed afterwards from the chain and the event logs. The funds verdict therefore uses "the sum of all accounts equals total supply, and total supply equals the mint record", not a before-and-after snapshot comparison. The 19 `IndexError`s have the same cause: `client/api.py` has since been fixed by someone else, see the bug list.

## Regression on the final program

After the matrix finished, a new chain was built with the final program including the slot fix (the program was recompiled with the matrix's program id), and wallets and listings were rebuilt.

- **R1 (PASS) the T1 path.** Paid 1.0, score 0.0 → 1.0, effect 6/6.
- **R2 (PASS) the X7 path, plus slots.** After delivery `slots_used` 0. While the first buyer stayed silent the listing stayed on the market, and the second buyer's job reached `delivered`. After rejection the buyer changed by -200000 units, with 200000 units owed.

## P1: conservation of funds

All token accounts were snapshotted before and after each item.

| Item | Minted during the item (test token units) | Sum of changes equals minted | Sum of accounts equals total supply |
|---|---|---|---|
| C1 |  | True |  |
| C2 |  | True |  |
| C3 |  | True |  |
| C4 |  | True |  |
| C5 |  | True |  |
| C6 |  | True |  |
| T4 | 10000000 | True | True |
| T5 |  | True |  |
| I1 | 160000000 | True | True |
| I3 | 160000000 | True | True |
| C7 |  | True |  |
| X3 | 0 | True | True |
| X1 | 0 | True | True |
| I4 |  | True |  |
| X6 | 320000000 | True | True |
| C7-v1 |  | True |  |
| X5 | 240000000 | True | True |
| C7b |  | True |  |
| C7c |  | True |  |
| T3 | 0 | True | True |
| C7d |  | True |  |
| C7f |  | True |  |
| T1 | 0 | True | True |
| T2 | 0 | True | True |
| T6 | 0 | True | True |
| X7 | 0 | True | True |
| X4 | 0 | True | True |
| X2 | 0 | True | True |
| I2 | 0 | True | True |
| R1 | 0 | True | True |
| R2 | 0 | True | True |

## Bug list

Fixed:

1. **`host list --infer-only --serves` required a local GPU.** It was wrongly refused when the backend was on another machine. Fixed in `node/cli.py`.
2. **Tasks that did not pass also waited for `accept_wait_s`.** This held a seller's slot for a day for nothing. Fixed in `node/daemon.py`.
3. **When a seller died mid-stream, the buyer took 192 seconds to find out, and the error message was empty.** Fixed in `serve/client.py`: the resume budget is now 60 seconds and the error message states the cause. From killing the seller to the error took 98.4 seconds in X5. That is above the budget, and how the time breaks down was not measured.
4. **When opening an inference session, a listing that had outlived its seller did not move on to the next seller.** Fixed in `client/core.py`, trying at most 3.
5. **`candidates` assumed the NAT preserves ports.** See C7. Fixed in `node/wire.py`.
6. **`--public-udp` was not verified.** See C7. Fixed in `node/cli.py`, `node/wire.py`.
7. **When a seller had answered but no link came up, the buyer paid 0.01 while the message said "refunded, no fault", and the ledger did not record it.** The fee is on-chain design (`abort_unstarted`, to stop free looks at addresses). The message, the ledger and the README were changed. The README used to say "full refund", which was incomplete. Fixed in `client/jobs.py`.
8. **`beetle host list --vllm-serve` crashed on every run.** `start_vllm` called `daemon.default_scratch`, which does not exist, so the first seller command in the README did not actually work. This was found only when T3 reached it. Fixed in `node/cli.py`, with a test.
9. **After a reboot the seller could not write adapters.** `/dev/shm` is empty after a reboot, and when Docker started vLLM it created the mount directory owned by root, so the buyer saw only `internal: PermissionError`. Now the parent directory is mounted and the seller creates the directory itself. If it is not writable, adapters are turned off and the reason is stated. Fixed in `node/cli.py`, `serve/gateway.py`. On l4 the owner was changed by hand.
10. **The ledger came up one step short after a buyer crash.** See M1. Fixed in `client/remote.py`, `client/core.py`, `client/policy.py`.
11. **A second buyer process on the same machine could connect to the first one's port.** Both `serve.json` files say 8741. After one side died the other took over the port, and this side reported the token as refused. Fixed in `client/api.py`, by another change in the working tree, not by me.
12. **`slots_used` was released only at settlement.** A silent buyer could hold a 1-slot seller for all of `accept_wait_s` (one day by default), at the cost of only the floor price. Fixed in `programs/`: the slot is released at `submit_result`, see R2. R2 verified.

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
