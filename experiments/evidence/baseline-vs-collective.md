# Single node against the collective: results

Measured 2026-10-31 and 2026-11-01; the runs named `quiet-*` are the later ones and carry their own start state. Raw records are in [runs/](runs/) and the machine-readable version is [baseline-vs-collective.json](baseline-vs-collective.json). All data is synthetic. Every remote seller is T3-C: transport is encrypted and the machine's owner can read the plaintext, and nothing here claims any hardware privacy. During runs marked `busy`, another session was building the App on the same Mac, which affects the Mac's time and memory.

## Conclusions

- **With the recipe as it stands and other work on the machine, a lone Mac finished neither size.** S: 3 runs, 0 finished, all stopped by the memory watchdog; recorded MLX peak 4.429 to 6.161 GB, against this Mac's GPU budget of 5.33 GB. L: 2 runs, 0 finished, all stopped while loading the model, after 21.8 to 32.5 seconds.
- **Lowering the batch did not finish S on the busy Mac either.** Batch 2: 2 runs, stopped at step 1. Batch 1: 2 runs, at the 15-minute cap they had reached step 53 of 384. Neither is the recipe; they only show what fits.
- **Whether a quieter 8 GB Mac can finish is still not answered.** 3 runs were meant to test it, but the Mac never got quiet: they started at a one-minute load of 4.23, 5.69, 18.37, against the limit of 3.0 set in advance, and they count as busy. None finished.
  - `quiet-bolt-1` did not finish: 30.6 seconds, phase load, stopped by: memory_pressure free 2% is under 8%; load 4.23, 11612 MB in swap and 27% free at the start.
  - `quiet-qwen-b2-1` did not finish: 33.5 seconds, phase train, step 1, stopped by: memory_pressure free 3% is under 8%; load 5.69, 10582 MB in swap and 24% free at the start.
  - `quiet-qwen-b8-1` did not finish: 85.0 seconds, phase train, step 1, stopped by: memory_pressure free 7% is under 8%; load 18.37, 13284 MB in swap and 22% free at the start.
- **Mac plus Beetle plus a seller finished both sizes.** S on l4: 3/3 settled, median total 154.9 seconds, $1.0 (test money). L on a100: 3/3 settled, median total 165.2 seconds, $1.2.
- **t4 and l4 cannot do L alone:** their `beetle host list` refuses it outright. t4 does not take training today either.
- **a100 alone can do the training:** this must not be described as "no single seller can do it". See Boundaries.
- The conclusion stops here: the collective completed computation a single node could not complete or could not do well. Nothing else was measured here.

## Can it finish?

| Path | S: Qwen3.5-2B LoRA | L: Bolt E4B LoRA | Basis |
|---|---|---|---|
| Lone Mac, recipe as is, busy | Did not finish, 0/3, watchdog stopped it | Did not finish, 0/2 | Local training is not wired into `beetle train`; this uses `sandbox/train_eval_mlx.py` |
| Lone Mac, smaller batch (not the recipe), busy | Did not finish, 0/4 | Not measured | `doctor`: S is train remotely, run here (serves here; training needs 6.0 GB); L is all remote (needs 10.8 GB free, has 2.72) |
| Lone Mac, runs meant to be quieter | Batch 8: did not finish, but the Mac was not quiet at the start. Batch 2: did not finish, but the Mac was not quiet at the start | did not finish, but the Mac was not quiet at the start | Runs named `quiet-*`, same watchdog |
| t4 as the only seller | Does not take it: `offer_not_training`; `host doctor` shows no sandbox image | Does not take it: needs 24 GiB, has 15 | market, host doctor, host list |
| l4 as the only seller | Finished 3/3 | Does not take it: needs 24 GiB, has 22 | Same |
| a100 as the only seller | Finished 1/1 | Finished 3/3 | The only seller that can take L |
| Mac plus Beetle plus the market | Finished, 154.9 seconds, $1.0 | Finished, 165.2 seconds, $1.2 | Real escrow and settlement, devnet, final program |

## What each node sells

Read from `beetle market` on devnet. These are listings, not hardware limits.

| Node | GPU | Train Qwen | Train Bolt | Infer Qwen | Infer Bolt |
|---|---|---|---|---|---|
| t4 | Tesla T4, 15 GiB | does not sell | does not sell | sells | does not sell |
| l4 | NVIDIA L4, 22 GiB | sells | does not sell | sells | does not sell |
| a100 | NVIDIA A100 80GB PCIe | sells | sells | does not sell | sells |

Quotes that name a seller:

| Request | Result |
|---|---|
| t4 infer offer asked to train qwen | 409 `offer_not_training` |
| l4 train offer asked to train bolt | 409 `offer_wrong_model` |
| t4 infer offer asked to train bolt | 409 `offer_not_training` |
| a100 qwen offer asked to train bolt | 409 `offer_wrong_model` |
| a100 bolt offer | $1.2, T3-C |
| a100 qwen offer | $1.2, T3-C |
| l4 qwen offer | $1.0, T3-C |

## Lone Mac: every run

Limits: a 15-minute hard cap, and a stop when `memory_pressure` free falls under 8%. Two more guards were added: a stop when disk falls under 4 GiB, and a stop when swap grows by more than 5,000 MB. qwen-1 ran under an earlier watchdog with no swap guard: it pushed swap to 17.9 GB, and one sample came 8 seconds late. MLX memory is recorded from the second version on. The runs named `quiet-*` were meant to test a quieter Mac and used the same limits; each counts as quiet only if its one-minute load at the start was under 3.0.

| Run | Group | Model | Batch | Result | Seconds | End phase | Step | MLX peak GB | Min free % | Swap MB | Load at start | Mean load | Hidden questions | Why it stopped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bolt-1 | busy | bolt-e4b | 8 | did not finish | 32.5 | load | not recorded | not recorded | 10 | 8587 to 14336 | 6.89 | 7.65 | n/a | swap grew 5749 MB, over 5000 MB |
| bolt-2 | busy | bolt-e4b | 8 | did not finish | 21.8 | load | not recorded | not recorded | 1 | 8355 to 11150 | 6.67 | 6.38 | n/a | memory_pressure free 1% is under 8% |
| quiet-bolt-1 | not quiet at start | bolt-e4b | 8 | did not finish | 30.6 | load | not recorded | not recorded | 2 | 11612 to 15393 | 4.23 | 6.63 | n/a | memory_pressure free 2% is under 8% |
| quiet-qwen-b2-1 | not quiet at start | qwen3.5-2b | 2 | did not finish | 33.5 | train | 1 | 2.452 | 3 | 10582 to 12348 | 5.69 | 6.31 | n/a | memory_pressure free 3% is under 8% |
| quiet-qwen-b8-1 | not quiet at start | qwen3.5-2b | 8 | did not finish | 85.0 | train | 1 | 2.776 | 7 | 13284 to 15579 | 18.37 | 15.88 | n/a | memory_pressure free 7% is under 8% |
| qwen-1 | busy | qwen3.5-2b | 8 | did not finish | 68.8 | train | not recorded | not recorded | 0 | 10583 to 17929 | 4.18 | 4.26 | n/a | memory_pressure free 0% is under 8% |
| qwen-2 | busy | qwen3.5-2b | 8 | did not finish | 27.1 | train | 1 | 4.429 | 4 | 9292 to 11265 | 3.62 | 4.38 | n/a | memory_pressure free 4% is under 8% |
| qwen-3 | busy | qwen3.5-2b | 8 | did not finish | 34.6 | train | 1 | 6.161 | 5 | 8691 to 13688 | 12.02 | 13.06 | n/a | memory_pressure free 5% is under 8% |
| qwen-b1-1 | busy | qwen3.5-2b | 1 | did not finish | 900.6 | train | 53 | 2.707 | 8 | 9046 to 13181 | 13.15 | 4.44 | n/a | wall clock over 900 s |
| qwen-b1-2 | busy | qwen3.5-2b | 1 | did not finish | 900.1 | train | 53 | 2.706 | 10 | 8433 to 13098 | 3.97 | 3.93 | n/a | wall clock over 900 s |
| qwen-b2-1 | busy | qwen3.5-2b | 2 | did not finish | 19.4 | train | 1 | 2.791 | 4 | 8988 to 10642 | 24.1 | 21.4 | n/a | memory_pressure free 4% is under 8% |
| qwen-b2-2 | busy | qwen3.5-2b | 2 | did not finish | 19.7 | train | 1 | 3.177 | 3 | 8718 to 10674 | 3.88 | 4.22 | n/a | memory_pressure free 3% is under 8% |

Manual steps: lone Mac 7 (install mlx-lm; download the model; find sandbox/train_eval_mlx.py, which has no caller, and write a driver; expand the data by the recipe; run it and watch memory; write the hidden-question evaluation; load the adapter by hand: it is not in Beetle's model library). Beetle 5 commands (beetle data add, beetle quote (optional), beetle train, beetle fetch, beetle chat), plus beetle init and beetle wallet fund the first time. These are counted, not measured.

## Mac plus Beetle plus a seller: every run

All payments are test money with no value. Total time runs from placing the job to settlement. Sandbox seconds are the training time the seller's sandbox reports.

| Run | Seller | GPU | To delivery s | Fetch s | Total s | Sandbox train s | Steps | Price | Paid | Hidden questions | Effect check | Checked on | Direct RTT ms | Settlement | Chain state | Transactions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| l-a100-1 | a100 | NVIDIA A100 80GB PCIe | 125.8 | 41.1 | 167.0 | 15.8 | 64 | 1.2 | 1.2 | 0.0 to 1.0 | 6/6, base 0 | remote | 23 | settled | closed | 8 |
| l-a100-2 | a100 | NVIDIA A100 80GB PCIe | 125.6 | 39.5 | 165.2 | 16.1 | 64 | 1.2 | 1.2 | 0.0 to 1.0 | 6/6, base 0 | remote | 20 | settled | closed | 8 |
| l-a100-3 | a100 | NVIDIA A100 80GB PCIe | 107.7 | 36.8 | 144.5 | 16.1 | 64 | 1.2 | 1.2 | 0.0 to 1.0 | 6/6, base 0 | remote | 20 | settled | closed | 8 |
| s-a100-1 | a100 | NVIDIA A100 80GB PCIe | 97.1 | 19.4 | 116.6 | 9.7 | 64 | 1.2 | 1.2 | 0.0 to 1.0 | 6/6, base 0 | local | 20 | settled | closed | 8 |
| s-l4-1 | l4 | NVIDIA L4, 22 GiB | 136.7 | 18.1 | 154.9 | 16.2 | 64 | 1.0 | 1.0 | 0.0 to 1.0 | 6/6, base 0 | local | 227 | settled | closed | 8 |
| s-l4-2 | l4 | NVIDIA L4, 22 GiB | 129.4 | 14.7 | 144.1 | 16.0 | 64 | 1.0 | 1.0 | 0.0 to 1.0 | 6/6, base 0 | local | 223 | settled | closed | 8 |
| s-l4-3 | l4 | NVIDIA L4, 22 GiB | 132.1 | 44.9 | 177.3 | 16.6 | 64 | 1.0 | 1.0 | 0.0 to 1.0 | 6/6, base 0 | local | 220 | settled | closed | 8 |

Failures: 0. s-l4-3 took 44.9 seconds to fetch. That step includes loading MLX on the Mac; the reason it was slow was not looked into separately.

Most of the time is not on the GPU: sandbox training took 9.7 to 16.6 seconds of a total of 116.6 to 177.3 seconds. The rest is chain confirmations, P2P transfer, sandbox start and fetching.

The network fee was 45000 lamports per job over 8 transactions, both sides together, about $0.0043 at SOL $95.93.

## Paths only the collective has

**One seller trains, another serves.** l4 trained the adapter and t4 served it; t4 takes no training itself. 3/3 questions right, the first in 47.1 seconds including opening the channel and loading the adapter. The channel settled at $0.000862, including the effect check. n=1. Record: `runs/collective/cross-l4-to-t4.json`.

**A seller gone before start, and the buyer moves on by itself.** A new wallet posted a cheaper training offer with no process behind it. The buyer named no seller and took it (`first_offer_is_ghost`: True). About 89.8 seconds later it was refunded, no fault. It then opened a job on l4 by itself, 254 seconds from the first order to settlement, effect 6/6, paid $1.0; the absent seller's stake was untouched. This is one run, n=1, on devnet on the final program. It measures a seller gone before start, not one lost mid-run. The offer was visible for about 10 seconds, then paused and closed. The script for this run stopped following too early, so the daemon finished the second job by itself and the same calls followed it afterwards; the record says so. Record: `runs/collective/failover-ghost.json`.

**Seller lost mid-run: not measured today.** Demo sellers may not be killed. The matrix has X2 (seller killed during training: refund and stake penalty) and X5 (seller killed during inference: the next request was answered by a second seller). Neither ran on the final program: both ran on local validator 2, before the 16:17 client change.

## Boundaries

- a100 alone can train S and L and also sells Bolt inference. That it lists no Qwen inference is a listing choice, not a hardware limit. So "a single seller cannot do this" holds for t4 and l4, and for a100 only at the level of its listings. Only a100 can take L today.
- L's effect check ran on a100, the same machine that trained it, so it is not an independent check. S was checked locally on the Mac.
- The hidden-question scores come from the seller's sandbox. Under T3-C that is the seller's word; what the buyer can check for itself is the effect check.
- This Mac has 8 GB; on the busy runs about 25% was free and other work was running. A larger or less busy Mac would give different results; this covers only this machine.
- In the two batch 1 runs, qwen-b1-1 reached step 37 in its first 30 seconds and only step 53 in 15 minutes; qwen-b1-2 reached step 36 in its first 30 seconds and only step 53 in 15 minutes. During the slowdown `sample` showed it stuck in kernel GPU command submission; the root cause is not settled. So a quieter 8 GB Mac might finish batch 1; that was not measured.
- The `quiet-*` runs started with 10582 to 13284 MB already in swap and with the Mac App's daemon running, so even a quiet start would mean a Mac with its usual work paused, not a freshly booted one.
- The Mac ran MLX 4-bit and the sellers ran full-precision PEFT. The Mac had no 3 canaries, so 48 steps at batch 8 against 64 on the sellers.
- No money outside a test network was involved. Prices are set by the demo sellers; they are not market prices.
