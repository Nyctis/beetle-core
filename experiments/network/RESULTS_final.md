# Final regression: node side

Run from 2026-10-31 21:51 EDT to 2026-11-01 03:05 EDT, against the code frozen on 2026-10-31 at 21:48 EDT. Everything ran on local validators with synthetic or public data and test tokens with no value. No devnet, no mainnet, no real money. Machines are named by role: `a100`, `l4`, `t4`, `cpu`, `mac`.

The first matrix is in [RESULTS.md](RESULTS.md) and [results.json](results.json), untouched. This run's machine-readable record is [results_final.json](results_final.json). Raw logs are in `logs/final/`, which git ignores.

## Verdict

- **44 entries in `results_final.json`: 40 PASS, 1 PARTIAL, 3 FAIL.** The failures are C7's second run, N7 and N9, and the partial is C7d. All are on the frozen code. None was fixed: this run only records.
- **The money held in every item.** All 39 balance checks in the results are true, and no check failed. A1, CHAIN, RUST, DEMO, F1 and the first C7 run record none.
- **The items from the first matrix are healthy.** Its C1 to C6, T, I and X items all passed again.
- **The failures are in paths the first matrix never reached:** a model server that hangs, a buyer that closes its own new channel during failover, and a cheap seller that takes no adapters. See [Product bugs](#product-bugs).
- **No pass condition of an existing item had to change.** None of the changes made at the freeze broke an assertion.

## What was run

Every item below ran on the frozen code, from a copy made at the start. The freeze hash in the scratch file cannot be recomputed, see [Frozen code](#frozen-code).

| Item | Result | Difference from the first run |
|---|---|---|
| Chain attack test | PASS, 401 of 401 | Same count. Took 370 s |
| Rust tests | PASS, 9 of 9 | Same |
| Pre-init checks in `setup_chain.sh` | PASS, 5 of 5 | Not recorded before |
| `market_demo.py` | PASS, exit 0 | Same |
| A1 sandbox scenarios | PASS, 6 of 6 as expected | Exit codes are now the real ones, see [A1](#a1-sandbox-scenarios) |
| C1 to C6 | PASS, direct 2 of 2 on all six paths | See [C1 to C6](#c1-to-c6) |
| C7, first run | PASS | Same pairs, same result |
| C7, second run | **FAIL** | Failover sub-check failed with bug B2 |
| C7d | **PARTIAL** | Pairs and rport product path passed. Failover failed with bug B4 |
| C7f | PASS, 23.8 s | First matrix 14.5 s |
| T1 | PASS, with a changed path | Adapter ran remotely on `a100`, not on a local server on `t4` |
| T1L | PASS, new | The local server half of T1, buyer on `a100` |
| T2 | PASS | Delivered 70.6 s, fetch 16.9 s. Was 75.3 s and 53.9 s |
| T3 | PASS | **Paid 1.0, not the floor of 0.2**: nothing ran beside it this time |
| T4 | PASS | Same |
| T5 | PASS | Same: 0 UDP packets to the seller, no money moved |
| T6 | PASS, with a changed path | Hidden questions 10 of 10, base 0 of 10, asked remotely |
| I1 | PASS | 25.0 token/s, was 18.9. Four accounts of the money agree at $0.001986 |
| I2 | PASS | Same answers |
| I3 | PASS | 61.2 token/s in all, was 66.9. Wallet equals ledger for all three buyers |
| I4 | PASS | 0 plaintext hits in capture and tap, all frames 4096 bytes, both modes |
| X1 | PASS | Refund in 36.2 s, was 36.7 s. Buyer net 0 units |
| X2 | PASS | Refund in 419.4 s, was 436.6 s. Stake lost 0.25 USDC, same |
| X3 | PASS | Declined at 60.8 s, was 60.9 s. No key on chain either way. Capped fee 10,000 units |
| X4 | PASS | Buyer away 121.6 s, recovered in 106.3 s, was 65.3 s. Paid 0.2. Slower because the effect check ran remotely |
| X5 | PASS | Error 99.5 s after the kill, was 98.4 s. Seller took 1,921 units above value, within a step of 2,000 |
| X6 | PASS | Settled in 159.3 s, same. Seller took 1,856 units below value, within a step |
| X7 | PASS | Buyer paid exactly 200,000 units, same |
| P1 | PASS | 39 balance checks across the items, all true |
| M1 | PASS on money, with discounts | See [M1](#m1) |
| N1 | PASS, new | Billing exact to the unit against a stub server |
| N2 | PASS, new | Billing against a real server |
| N3 | PASS on money, new | Deposit ran out. Exposes bug B6 |
| N4 | PASS, new | Server answers 500, stream breaks |
| N5 | PASS, new | Empty wallets at every moment that matters |
| N6 | PASS, new | Link dark for 8, 45 and 150 s |
| N7 | **FAIL**, new | Model server hangs: bug B1 |
| N8 | PASS, new | Seller frozen: error after 92.1 s |
| N9 | **FAIL**, new | Cheapest seller takes no adapters: bug B5 |
| N10 | PASS, new | Dead seller: deposit returned, but only after the grace period. Bug B3 |
| F1 | PASS, new | `chain` in health, `gpu` in the market, `receipt` in the proof |

## Frozen code

- The scratch file holds `e3a2a1336eab2403`. **It can't be reproduced.** Its recipe also hashes `.rs` and `.toml` files under `programs/beetle/target/`, and cargo changed those after the freeze.
- Without `target/`, the working tree and the copy this run used both hashed to `93a35617961a78c2` when the run began.
- At that moment no source file was newer than the freeze. The only newer files outside `target/` were `README.md`, `LIMITATIONS.md` and `docs/README.zh.md`.
- **The source moved on during this run.** See [Changes after this run's copy](#changes-after-this-runs-copy).
- The sandbox's code measurement matched the images already on `a100` and `l4`, so no image was rebuilt.

Copies went to a new directory on each node with a new virtualenv from `uv.lock`. Every role had its own home and ports. Keys were generated fresh for these chains.

## A1 sandbox scenarios

On `l4`, with its GPU.

| Scenario | Exit | Seconds | Outcome |
|---|---|---|---|
| honest | 0 | 29 | Passed, seller paid 2.00. Leak scan clean |
| program | 0 | 26 | Passed, seller paid 2.00. Leak scan clean |
| program-attack | 0 | 19 | 8 of 8 attacks blocked, no `leak.txt`. Leak scan clean |
| tamper-model | 1 | 5 | Sandbox aborted: `base_model_hash_mismatch`. Leak scan clean |
| tamper-code | 1 | 3 | Key refused on `measurement`. Buyer refunded 2.00, seller slashed 0.50. Leak scan clean |
| wrong-dek | 1 | 3 | `data_decrypt_failed`, settled as requester fault: seller 0.40, buyer back 1.60. Leak scan clean |

Exit 1 is how the last three end. The first matrix recorded the exit code of `tail`, so it showed 0 for all six. The leak scan looks for planted strings; it is not a proof of no leak.

## C1 to C6

Two rounds of 32 MiB each, no relay. RPC went through tunnels on the Mac, so rendezvous times run high.

| Item | Path | Direct | Rendezvous s | Link s | RTT ms | Down Mbps | Up Mbps | First run: rendezvous, link, RTT, down, up |
|---|---|---|---|---|---|---|---|---|
| C1 | mac to a100 | 2/2 | 1.29 | 0.08 | 22.2 | 45.8 | 64.35 | 4.62, 0.06, 19.7, 46.8, 65.55 |
| C2 | mac to l4 | 2/2 | 4.87 | 1.23 | 231.95 | 25.65 | 13.55 | 3.3, 0.52, 233.45, 10.6, 30.9 |
| C3 | t4 to a100 | 2/2 | 5.36 | 0.07 | 28.6 | 45.95 | 29.65 | 5.46, 0.08, 29.65, 46.3, 16.15 |
| C4 | cpu to l4 | 2/2 | 5.92 | 0.01 | 1.4 | 32.6 | 44.95 | 5.1, 0.01, 1.8, 17.35, 44.75 |
| C5 | cpu to a100 | 2/2 | 5.02 | 0.48 | 219.0 | 31.05 | 30.75 | 6.14, 0.46, 214.15, 31.75, 31.35 |
| C6 | t4 to l4 | 2/2 | 4.76 | 1.11 | 233.45 | 26.0 | 25.0 | 5.15, 0.9, 231.8, 25.35, 25.0 |

## C7, C7d and C7f

A simulated internet on one machine: real NATs, not a real path. C7 ran twice. The second run came after M1, once the offers of dead lab sellers had aged out.

| Seller / buyer | C7 run 1 | C7 run 2 | C7d |
|---|---|---|---|
| full / full | linked, 11.9 s | linked, 12.0 s | |
| cone / cone | linked, 11.4 s | linked, 12.2 s | linked, 12.0 s |
| full / sym | linked, 11.9 s | linked, 11.4 s | |
| sym / full | linked, 12.0 s | linked, 12.2 s | |
| cone / sym | fails cleanly, 54.8 s | fails cleanly, 54.9 s | |
| sym / sym | fails cleanly, 54.9 s | fails cleanly, 54.7 s | fails cleanly, 54.9 s |
| rport / rport | | | linked, 11.4 s |
| cone / rport | | | linked, 11.7 s |
| rport / cone | | | linked, 11.5 s |
| full / rport | | | linked, 11.8 s |

| Product check | C7 run 1 | C7 run 2 | C7d |
|---|---|---|---|
| Only an unreachable seller: clear `no_seller` | 155.8 s | 157.8 s | 143.1 s |
| Add a reachable, dearer seller | **answered, 77.2 s** | **failed, 77.9 s, `payment_refused`** | **failed, 145.2 s, `no_seller`** |

- C7f, real daemons behind two port-changing NATs: answered in 23.8 s.
- The same failover check gave three different results in three runs. That is two bugs, B2 and B4, not one flaky test.

## Inference paths the first matrix did not reach

N1, N3, N4, N6, N7, N8, N9 and N10 use a stub seller on `cpu`: a real daemon, wallet, offer and channel, with `stub_backend.py` as its model server. It prices at 2 and 5 USD per million tokens. The buyer is `t4`, except N2 (`mac`), N5 and N10 (fresh wallets on `cpu`), and N9 (`cpu`). `pieces` counts are after the client's hold for a `</think>` marker, so the stub's short streams show few pieces, and none before an error.

### N1: billing, exact

Four requests of 1, 7, 64 and 500 pieces.

| Count | Tokens or units |
|---|---|
| Prompt tokens, server and buyer | 47 and 47 |
| Completion tokens, server and buyer | 2,092 and 2,092 |
| Expected from the offer | 10,554 units |
| Channel settled | 10,554 |
| Buyer paid | 10,554 |
| Seller got | 10,554 |
| Buyer's ledger | 10,554 |

### N2: billing, real server

`mac` to `l4`, `max_tokens` of 1, 16, 64 and 256.

- The channel, buyer, seller and ledger all say 462 units, the amount the offer gives for the billed tokens.
- The server's counters: prompt 136, generated 115. Billed: prompt 120, completion 114.
- The gaps of 16 and 1 tokens favour the buyer: it pays the smaller count.

### N3: the deposit runs out

A deposit of 10,000 units against an answer of 3,000 pieces, at 5 units per token.

- The answer stopped at 1,166 pieces, 1,988 tokens, 9,972 units. Nothing was owed past the deposit.
- Channel, buyer, seller and ledger all say 9,972.
- **The caller saw success.** `ok` true, `finish_reason` stop, with no sign it was cut off: bug B6.
- The next request failed with 502 `route_failed`, `payment_required`, not 402 `budget_exceeded`.
- The harness's first try used a deposit of 500 units. Its step of 10 units was too small: two channels, both refused, 42 units paid for no tokens: bug B7.

### N4: the model server fails

| Case | Buyer sees | Seconds | Next request |
|---|---|---|---|
| Server answers 500 | 502 `route_failed`, `backend answered 500` | 13.3 | Works |
| Stream breaks after 20 pieces | 502 `route_failed`, incomplete chunked read | 0.7 | Works |

Channels settled at 10,238 units in all, and buyer, seller and ledger agree. The 500 cost nothing. The broken stream's channel settled at 10,131 units against 107 used: **a failed request can cost a full voucher step**, $0.01 at the default deposit. This is the design, and the response shows no cost, so it is worth knowing.

### N5: wallets that can't pay

| Moment | Chat | Training job |
|---|---|---|
| No SOL | 402 `no_funds`: no SOL for fees | 402 `no_funds`, job not opened |
| SOL, no USDC | 402: holds $0.00, needs $0.50 | 402: holds $0.00, needs $1.00, job not opened |
| $0.30 | 402: holds $0.30, needs $0.50 | 402: holds $0.30, needs $1.00, job not opened |

- **Fetch with nothing left after delivery:** 402 `no_funds`, job stays `delivered`. After a top-up the same fetch settled, paid 1.0, effect 6 of 6.
- **Alias set, automatic fetch, nothing left:** job stays `delivered` with error `no_funds`. After a top-up and a daemon restart it finished by itself in 70.3 s, paid 1.0.
- **`--local-only`:** the market answers 409 `local_only`. Its message says "this Mac" on a Linux machine: bug B10.

### N6: the link goes dark

UDP dropped both ways on the seller's session port, during a 90 s answer.

| Dark for | Result | Pieces in order | Next request |
|---|---|---|---|
| 8 s | Finished, 103.0 s, 7,891 tokens, $0.039489 | Yes, none lost or repeated | 0.4 s |
| 45 s | Finished, 91.0 s, 7,891 tokens, $0.039489 | Yes, none lost or repeated | 0.3 s |
| 150 s | 502 `route_failed`, `could not resume`, 99.3 s after dark began | No text delivered | 14.3 s, a new session |

Afterwards channel, buyer, seller and ledger all say 95,277 units. No firewall rule was left behind.

### N7: the model server hangs: FAIL

The server sends 5 pieces, stays connected and sends nothing more. **Bug B1.**

- The request got no answer. Clients gave up at 240 s, 400 s and 520 s in three separate tries. Nothing in the path has a deadline.
- A second request on the same session got no answer either.
- A later request worked on a new session, in 13.8 s. By the code, that comes after the 300 s idle sweep drops the old session.
- The server's stream was still open when the client gave up.
- **The buyer's upkeep stopped.** Ledger entries stayed `escrowed` for channels already `Distributed` on chain, the hung channel stayed `Open`, and its 500,000-unit deposit stayed out of the wallet.
- **SIGTERM did not stop the daemon**, in three of three tries. The next API call started a second daemon on the same home. This run had to use SIGKILL, because the extra daemons took `t4` down to 217 MB of available memory, beside a demo model server.
- A restart cleared everything: no entries left in escrow within 70 s.

### N8: the seller freezes

Its processes were stopped with SIGSTOP. The buyer got 502 `route_failed`, `could not resume`, 92.1 s after the freeze, and the next request worked in 13.1 s. Money settled at 10,141 units on all sides: again a full step for an answer that never arrived.

### N9: the cheapest seller takes no adapters: FAIL

**Bug B5.** With the stub repriced as the cheapest seller, a chat on a trained adapter failed after 48.4 s with `adapter_rejected: backend answered 404`. It did not try `a100`. Naming `a100` worked: "Your cat's name is Pixel."

The same limit forced a change in this run: see [How this run differs](#how-this-run-differs).

### N10: a seller that never answers

The buyer used a grace period of 120 s, and its seller was killed first. The request answered `no_seller` in 43.5 s, with 500,000 units already in a channel.

| Seconds | Channel |
|---|---|
| 44 | Open |
| 55 | Closing |
| 184 | Sealed |
| 215 | Distributed, refunded in full |

The refund works without the seller. **With the default grace of 21,600 s the same deposit would be out for about six hours.** That wait was not measured: bug B3.

### F1: fields added at the freeze

- `/beetle/health` has `chain`.
- Market rows have `gpu`: "NVIDIA A100" on `a100`, "NVIDIA L4" on `l4`, and null on the stub, which has no GPU. The high byte of `hw_class` therefore survives a real `host list`.
- A session's proof has `receipt`, with GPU, model, revision, model hash, channel and settlement.

## M1

Fixed seed 20261031. The market closed after 2,245 s: 1,800 s plus requests still running. The close took 1,526 s, the harness's limit.

| Measure | Value |
|---|---|
| Funds conserved to the unit | True |
| All accounts equal supply | True |
| Jobs left open | None |
| Channels of the three sellers | 97, all `Distributed` |
| Stuck, as the harness counts it | None |
| Chats | 115: 80 answered, 25 `no_seller`, 10 `route_failed` |
| Training attempts | 30: 7 opened, 23 `no_seller` |
| Fetches | 2, 1 failed with `internal` |
| Buyers vanished | 34 |
| Seller events | 6 down, 5 back, 5 paused, 4 resumed |

**Why 23 of 30 training attempts got `no_seller`.** This run measured it at each failure. `a100` is the only training seller and has one slot, and an offer leaves the market while its slot is taken.

| Seller's state then | Slot taken | Attempts |
|---|---|---|
| Up | Yes | 12 |
| Paused | Yes | 6 |
| Down | Yes | 4 |
| Up | No | 1, most likely freed between the two reads |

So the single slot was taken in 22 of 23, and in 10 of those the seller was also paused or down. This is the setup, not a defect: one seller, one slot, six buyers. M1 is weak evidence for training, as before. Use the T and X items for that.

**Discounts on this PASS:**

- The harness's "stuck" count covers only the three listed sellers. **13 more channels, $6.50 of deposits, were `Closing` with about 5 hours of grace left when the chain was stopped.** Their two sellers were dead lab sellers from C7d and C7f, whose offers were still fresh and the cheapest when M1 began. Neither ever answered. These funds are in the conserved total, but their refund was not observed: the chain was shut down first. N10 shows the path. Bug B3.
- 5 of the 10 `route_failed` were `payment_refused`, bug B2. The other 5 were `could not resume` after a seller went down. The one failed fetch was bug B2 too.
- The per-buyer check of wallet plus ledger was not computed.
- The cast changed: see below.

## Product bugs

None fixed.

| # | Severity | Where | What happens |
|---|---|---|---|
| B1 | High | `serve/gateway.py` `Backend` has no read timeout. `serve/client.py` waits on `pipe.recv()` with no deadline. `client/core.py` `_drain` keeps the session lock. `client/core.py` `close_idle` and `client/remote.py` `close` wait for that lock. `client/daemon.py` `upkeep` waits for `close_idle` | A model server that hangs, or a seller that fakes one, gets no error to the buyer and wedges that session. Every channel sweep of that buyer then stops: refunds and ledger, for all sellers, until a restart. SIGTERM does not stop the daemon, and the next API call starts a second one on the same home |
| B2 | High | `client/core.py` `sweep_channels` | The buyer closes its own new channel. `live` is read once, then the loop waits on RPC for older entries. A channel that opens meanwhile is not in `live`, and an entry with no `closed_at` counts as long closed, so it gets `request_close`. On chain: opened, then the buyer's own third transaction 7 s later, in the pass that closed the dead seller's channel 5 s before. It shows up as `payment_refused: no such open channel` after a failover, up to 106 s into a request. It explains C7 run 2 and 6 M1 failures. The transaction timeline fits this cause; a debugger never confirmed it |
| B3 | High | `client/core.py` `_start_session` opens the channel before contacting the seller. `client/remote.py` `GRACE_S` is 6 hours | Any attempt at an unreachable seller locks the deposit, $0.50 by default, for the whole grace period. `used_today` counts it, so it also uses the daily limit. 13 channels and $6.50 in M1 |
| B4 | Medium | `client/core.py` `remote_session`, `[:MAX_SELLERS]`, with `OFFER_FRESH_S` of 1,800 s | Dead cheap offers fill the three tries, and a live dearer seller is never reached. Seen in C7d: `no_seller` after 145.2 s with a reachable seller in the market. Related to open bugs 14 and 15 in RESULTS.md |
| B5 | Medium | `client/core.py` `remote_session` moves on only for `no_seller` | A cheaper seller whose server takes no adapters ends the request with `adapter_rejected` instead of trying the next. By the code the effect check on fetch takes the same path; that was avoided here, not observed. A plain vLLM with no runtime LoRA is such a seller |
| B6 | Medium | `serve/client.py` sets `stopped`, and nothing reads it. `client/daemon.py` always sends `finish_reason` stop | An answer cut off by the deposit looks finished and complete |
| B7 | Low | `client/remote.py` step, `node/metered.py` `_chat` | With a very small `max_usd`, the seller's estimate of the request exceeds the first voucher. Every request is refused after a channel was opened, and a few units are paid for nothing. Seen at a deposit of 500 units |
| B8 | Low | `client/remote.py` `close` | `closed_at` is set in memory and not saved, so a restart loses it |
| B9 | Low | Fetch path in `client/core.py` | A `GatewayError` on fetch reaches the caller as 500 `internal`. Seen once in M1 |
| B10 | Low | `client/core.py` `local_only` message | Says "this Mac" on Linux |

Not a bug, but worth saying: a failed request can cost one voucher step (N4, N8).

## Changes after this run's copy

A change made while this run was going touched `client/core.py`, `client/mcp_server.py`, `client/proof.py` and two test files. **Nothing here tested those changes.**

- `client/core.py`: a cancelled job only collects its refund on restart, jobs can be looked up by full id, and a job with no evidence gets a `no_evidence` card.
- `client/proof.py`: that card.
- `client/mcp_server.py`: `wait_s` on `job_status`, and tool descriptions.

None of them touches the code named in B1 to B9, so those bugs still stand in the current tree, by reading it.

## How this run differs

**Assertions:** no pass condition of an existing item changed. New items have their own.

**Paths that changed, and why:**

- **T1, T6 and X4 ask for their answers remotely.** `t4` had 1.1 GB of memory free, too little for a second model server, and its demo server was off limits. Their adapter half ran on `a100` as a seller. T1L covers the local server path instead, with the buyer on `a100` next to the seller, so it is not a network test of that path.
- **X5's second seller is `a100`.** `t4` has nothing to sell from, for the same reason.
- **The `l4` seller was paused from T1 on, except for N2 and the first F1 run.** Its model server is shared with the demo seller and could not see this run's adapter directory. Bug B5 would otherwise have stopped those items.
- **The `l4` seller used the demo seller's model server for inference only.** This run sent it chat requests and read its metrics. It loaded no adapter.
- **`a100b`'s Bolt server** was this run's own until T3 replaced it through `--vllm-serve`.
- **M1's cast:** buyers on `mac`, `cpu` and `l4`, two each, because `t4` has no memory to spare. Sellers `a100`, `a100b` and the stub, with `l4` kept paused. So part of M1's inference came from the stub, not a real model.

**Harness changes, none in product code:**

- Log and result paths from the environment.
- Homes, code directory, ports, backends and scratch from the private config.
- X2 removes only its own job's sandbox. It used to remove every sandbox on the machine.
- The NAT lab reads the RPC port from the chain file, and looks up the market inside its namespace.
- `T3_adapter` frees the port of this run's Bolt server.
- `market_run.py` takes its cast from the config and records why training found no seller.
- `scrub.py` also cleans `results_final.json`.
- New files: `final_items.py`, `stub_backend.py`, `run_final.py`.

**First tries that failed for reasons in this harness, and were rerun:**

- N3 used a deposit too small to cover a request (that result is kept as bug B7), then a check on the server's piece count that buffering defeats.
- N4 held the hang and freeze cases too, and the hang blocked the freeze. They became N7 and N8.
- N7's first run lost its record to a JSON parsing error in the harness. Its log has the same outcome, with a limit of 520 s.
- F1's first run demanded a GPU from the stub seller.
- C7d and C7f first failed because a leftover daemon from an earlier run held the buyer's port outside the namespace. Their sellers then stayed on the market as dead offers, which is how B4 and the 13 channels in M1 arose.

## Not run or not shown

- **T1, T6 and X4 with a model server on `t4`.** See above.
- **C7b, C7c, C7e, R1 and R2.** Not requested. Every item here already ran on the final program.
- **Refund after the default six-hour grace.** Only shown at 120 s, in N10.
- **Refund of M1's 13 channels.** The chain was stopped first.
- **Per-buyer wallet and ledger totals in M1.**
- **`sim_market.py`.** Not requested.
- **Any public network path for C7, and the home routers it stands for.** Same limit as the first matrix.
- **Hardware privacy.** Every remote seller was T3-C.
