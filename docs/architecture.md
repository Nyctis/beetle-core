# Architecture

How Beetle divides work between the chain, the two machines and the seller's runtime environment. For what this does not achieve, see [LIMITATIONS.md](../LIMITATIONS.md). `programs/beetle/src/lib.rs` is the source of truth for on-chain rules, and `common/job_defaults.py` for their parameters.

```
 Buyer machine                     Solana                      Seller machine
 ┌───────────────────────┐   ┌───────────────────┐   ┌──────────────────────────────┐
 │ beetle serve          │──▶│ beetle program    │◀──│ beetle host run              │
 │  wallet, spend limits,│   │  offers, stakes,  │   │  wallet, ciphertext only     │
 │  privacy floor        │   │  job escrow,      │   │                              │
 │ CLI, SDK, MCP,        │   │  settlement       │   │ sandbox: Docker, no network  │
 │ OpenAI-compatible API │──▶│ payment-channels  │◀──│ inference env: no wallet,    │
 │ local engine (MLX)    │   │  (inference)      │   │  sees plaintext on T3-C      │
 └──────────┬────────────┘   └───────────────────┘   └──────────────┬───────────────┘
            └───────── direct QUIC link (iroh), no relay ───────────┘
                       inner layer sealed to the seller's environment
```

## Parts

| Part | Runs on | Holds | Code |
|---|---|---|---|
| Buyer daemon, CLI, SDK, MCP | Buyer | Wallet, spending limits, privacy floor, jobs | `client/`, `beetle.py` |
| Seller daemon | Seller | Seller wallet. Handles ciphertext only | `node/daemon.py` |
| Training sandbox | Seller, Docker, no network | One-time keys, plaintext during training | `sandbox/` |
| Inference environment | Seller, child process | Plaintext prompts. Never opens the wallet | `node/environment.py`, `serve/` |
| `beetle` program | Solana | Offers, stakes, escrow, settlement, sealed addresses | `programs/beetle/` |
| payment-channels | Solana, Solana Foundation's program | Inference deposits | Not in this repository |
| Optional verifier | Anyone | Signs evidence checks for buyers who name it | `attester/` |

All chain access goes through `common/beetle_chain.py`.

## On chain and off chain

The program does not judge hardware evidence and has no arbiter. It moves money on objective facts.

| On chain | Off chain, and why |
|---|---|
| Offers: price, capacity, measurement, claimed tier | Training data, weights, adapters: a public ledger cannot hold them |
| Stakes, job escrow, settlement | Hardware evidence: too large for a transaction, and the program does not judge it |
| Sandbox public keys, sealed data key, evidence hash | Scores and logs: only in the private report |
| Outcome, share of work done, report hash | Tokens and per-request usage: vouchers and receipts between the two machines |
| Sealed peer addresses | Reputation scoring: done by each buyer from on-chain counters |
| Channel open and final settlement | Privacy decisions: made on the buyer's machine |

An offer's `tier` is the seller's claim. Sellers who pass `--public-udp` also put an address on chain, in clear and for good.

**Money.** Bond is at least 50% of the price and offers lock 5 USDC. The program takes no fee. Parameters (stake ratios, slashes, windows) come from `NETWORK_PARAMS` in `common/job_defaults.py`. They are written to `Config` at `init_config` and fixed from then on, and only the program's upgrade authority can initialize. `scripts/setup_chain.sh` computes `chain_ref`, a hash of the network's identity, from the genesis hash and writes it to `chain.json` and `Config`. Every signature binds it, so a signature from one network does not count on another, and a new chain means running the script again.

**Cost (estimate).** One training job takes 8 transactions and 9 signatures, about $0.0043 at SOL $95.93, with no verifier and including the address exchange. Account rent is extra and is returned on close. Tokens never go on chain.

## P2P

Links use iroh over QUIC with relays turned off, so a link that exists is direct.

1. **Find your address.** Each side learns its public address from a STUN server it chose, or from `--candidate` or `--public-addr`. There is no default STUN server, because each one learns the machine's IP.
2. **Swap addresses through the chain.** `{id, addrs}` is sealed to the other side and has a fixed size, which hides the candidate count. A training job uses a `Rendezvous` account (`rdv_open`, `rdv_answer`). An inference session or a closed job has no such account, so the same ciphertext goes in a memo, and only the two parties' signatures count.
3. **Dial at the same time.** The first QUIC packets open each NAT. The first connection to complete wins.
4. **If that fails, move on.** A buyer tries up to 3 sellers and never one below the floor. A job that named its seller does not move on: it is refunded. A next offer above the price already approved goes through the usual limits, so it can stop and wait for a person. A failed connection costs no bond: `abort_unstarted` refunds the buyer, who pays 0.01 only if the seller had answered.

iroh keeps its socket to itself, so a side asks STUN on the same port just before iroh takes it. A NAT that maps by source port, as nearly all do, then gives iroh that same mapping. A symmetric NAT does not.

There are two encryption layers. The outer one, iroh's TLS 1.3, ends at the seller's host and only protects transport. The inner one is sealed with HPKE to the environment's `enc_pk`, so the host and the network see ciphertext. Inner frames all have one length. Files are split into 1 MiB chunks with SHA-256 manifests, and padded to size tiers. On T3-C the environment is inside the owner's reach, so the inner layer does not protect anything from the owner.

### Address exchange

Each side needs the other's address before it can dial. Addresses cannot go on chain in clear, or the link between wallet and IP would be public for good. A seller keeps a `net_pk` on chain (`register_seller`, `rotate_net_pk`). A buyer seals `{id, addrs}` to it, with a one-time key for this round, and the seller seals its answer to that key. At most 5 candidate addresses fit.

| Case | Code | What is on chain |
|---|---|---|
| Training job | `ChainRendezvous` | A `Rendezvous` account: `rdv_open`, `rdv_answer`, `rdv_reoffer`, `rdv_fail`, `rdv_close`. It is closed once connected, which returns its deposit |
| Inference session | `ChannelRendezvous` | A payment channel has no job, so no such account. The same ciphertext goes in the memo of a transaction that mentions the channel, and only memos signed by the two parties are read |
| Job whose account has closed | `ChannelRendezvous` | The same, with memos that mention the job's address, to fetch a result |
| Development | `FileRendezvous` | Nothing. `--rendezvous-dir` names a directory both sides can read and write. Addresses are not encrypted here |

A memo stays in the transaction history and cannot be closed, so memos hold ciphertext only. Anyone can mention an address, so a reader looks back 1,000 transactions each time.

## Training job and key release

A seller first runs `register_seller`, `stake` and `post_offer`. A buyer filters offers by function and tier with `list_offers`, then takes one in a single transaction. The rest happens over the direct link, and the chain records only hashes:

```
buyer   open_job         price into escrow, seller's bond locked
buyer → seller           public spec and four ciphertext files
seller                   starts the sandbox: checks it has no network, makes one-time keys
seller → buyer           evidence bound to job, spec and keys
seller  start            sandbox keys and evidence hash on chain
buyer                    checks evidence, measurement and floor on its own machine
buyer   release_key      data key sealed to the sandbox's key, or reject_evidence
sandbox                  opens the data, checks hashes, baseline, train, evaluate on hidden questions
seller → buyer           adapter ciphertext only; buyer signs a receipt
seller  submit_result    sandbox's Ed25519 signature checked through the precompile
seller → buyer           report holding the adapter's sealed key
buyer                    checks the signature, opens the adapter, tests that it changes answers
buyer   accept / reject  then anyone can finalize and claim
```

**Key release is the buyer's decision.** The evidence hash is on chain, but only the buyer checks the evidence. `report_data` binds it to this job, spec and key pair. Releasing the key is the buyer's endorsement of those keys, and no report can move money before it. If routing goes wrong, the data stays ciphertext at the seller, because the floor is enforced here. The buyer must be online: the window is 600 seconds on devnet parameters, and the daemon does this automatically.

**T3-C has no evidence to check.** The buyer only confirms that the sandbox's claimed keys are this job's. A seller can pass this.

**Ciphertext and key arrive separately.** This stops anyone from holding a usable adapter and a refund right at the same moment: until `submit_result` lands, the buyer holds a file it cannot open.

**What the buyer gets.** The adapter's key is sealed to the buyer only if the job passed. Scores and logs are in the private report, which only the buyer can open. The buyer stores the opened adapter encrypted on its own disk.

**Canaries.** The buyer mixes 3 canary records into the training data: made-up facts, unique to the job, that only the buyer can derive. They do not stop a leak. If one shows up elsewhere, it shows which job leaked.

**Modes.** Recipe mode runs the built-in trainer. Program mode runs the buyer's `train.py` in a child process locked down with seccomp, Landlock and resource limits, and it cannot see the hidden questions, weights or keys. `beetle` commands only send recipe jobs today.

**The sandbox's limits.** It has no network, and it cannot trust what the host carries in. A hash mismatch therefore never counts as the buyer's fault. Hitting the wall clock counts as Inconclusive, because the host controls the machine's speed.

## What the seller can see

This table is for a training job, and only for a seller that does not read memory.

| Can see | Cannot see |
|---|---|
| Which public model, the resource budget, each input's hash, the job's one-time public key | Training data, hidden questions, recipe, acceptance thresholds, the buyer's code |
| Four ciphertext files of one size, from size tiers that start at 1 MiB | How much data there is (`units` is rounded up to 1, 2 or 5 times a power of ten) and whether the job is recipe or program mode |
| Phase names in the sandbox log | Any score, loss or piece of data |
| The public report: pass or fail, outcome, share of work done and hashes | The private report: scores and logs |

**An ordinary machine has no hardware confidentiality.** Its owner can `docker exec` in and read the container's memory. To stop that too, the job must run in a TDX confidential VM. `node/runtime.py` is the interface for that, and only `DockerRuntime` exists today.

## Evidence

The chain does not check hardware evidence, and there is no central verification service. For what each tier protects and its state today, see [LIMITATIONS.md](../LIMITATIONS.md#remote-paths-are-t3-c).

| Evidence | Tier | What it shows | How it was tested |
|---|---|---|---|
| TDX quote | T2 | CPU-signed: this VM image runs, with memory encrypted | Quotes from real TDX hardware (dcap-qvl samples): signature chain, TCB status, tampering, expiry |
| App Attest | T1 | Apple-signed: our app on a genuine Apple device | A real iOS 14.4 device's attestation and assertion, checked by Apple's 9 steps |
| TPM 2.0 quote | T3-B | TPM-signed: the machine booted the locked image | A software TPM (swtpm) in Docker, interoperating with real tpm2-tools |
| None | T3-C | Nothing. The buyer only confirms that the sandbox's claimed keys are this job's | Ordinary GPU servers. Mainnet parameters allow this tier. The buyer's floor must be `public` |
| MOCK | MOCK | Nothing | The local Docker sandbox. Development networks only |

**Evidence policy.** A buyer's policy is `~/.beetle/evidence_policy.json`, in the format described in `attester/verifier.py`. An empty allowlist accepts nothing. With no file, a test network accepts only MOCK evidence of the public sandbox code, and mainnet accepts no evidence. MOCK evidence is always refused on mainnet. T3-C jobs do not use the policy, because they have no evidence to check.

**Optional verifier.** The program lets a buyer name a `verifier` in `open_job`, and `start` then needs that party's Ed25519 signature. `attester/` is such a service, and anyone can run it. The buyer picks the verifier and can change it, so it is not a global single point. `beetle train` has no option for this today.

**TPM enrollment.** A TPM device enrolls once. The seller's `--attester` points at an `attester/server.py` that the buyer runs, and the buyer verifies with the same state file. That server hard-codes a development policy: see [LIMITATIONS.md](../LIMITATIONS.md#chain-and-deployment).

## Acceptance and settlement

Only an evidenced T2 sandbox's signature can release the full price. Off T2 the signing key may be in the seller's hands, so its "passed" is only the seller's word.

| Signed outcome | T2 | Every other tier |
|---|---|---|
| Passed | Full price | `accept` pays the full price; `reject` or silence pays the floor |
| Failed | Floor | Floor |
| Buyer's fault (`sealed_dek_hash` must match the chain) | The larger of 5% and the work done | 5% |
| Inconclusive | Full refund | Full refund |

The floor is price × 20% × share of work done. `beetle fetch` accepts only after the adapter check passes, and otherwise rejects. A seller gives the buyer one day (`accept_wait_s`) to accept before it calls `finalize`. That is client policy; the program's settle delay is 3,600 seconds on devnet parameters. `claim` then pays each side separately, so one frozen account cannot block the other.

| Event | Trigger | Result |
|---|---|---|
| Seller declines in time | Seller, `decline_job` | Full refund, no slash |
| No connection, seller never answered | Either side, `abort_unstarted` | Full refund, no slash, failure recorded |
| No connection, seller had answered | Buyer, `abort_unstarted` | Seller gets 0.01, the rest is refunded |
| Seller never starts | Anyone, `timeout_start` | Full refund, 10% of bond slashed (at least 0.10), offer paused |
| Seller gives up mid-run | Seller, `abort_job` | Full refund, 10% slashed |
| Seller runs out of time | Anyone, `timeout_run` | Full refund, 50% slashed |
| Buyer never releases the key | Anyone, `claim_no_release` | Seller gets 2% (at most 0.01), the rest is refunded |
| Evidence fails | Buyer, `reject_evidence` | Seller gets 1% (at most 0.01), the rest is refunded |

`abort_unstarted` needs an opened rendezvous and `rdv_abort_after` to have passed. The failure it records is `n_connect_fail`. When the seller calls it, the refund is full. The 0.01 is the price of reading a seller's address and leaving, so collecting addresses is not free.

A seller also uses `decline_job` when the buyer has not appeared after one minute, when inputs are still incomplete at `start_by`, when the spec asks for more than was paid (`job_defaults.spec_problem`), when the model is not on its list, when the sandbox does not attest, or when the job names a verifier it does not use. The refund is full and the stake is not touched.

A buyer may go offline after releasing the key, within a limit. After settlement a seller keeps the result for `retain_s`, 7 days by default, and deletes the job's inputs at once. It deletes the result sooner if the buyer signs that it has everything. Once the job account is closed, the `Rendezvous` account cannot be used, so a returning buyer finds the seller through memos on the job's address. The daemon does this when it resumes.

## Inference

Inference does not use the `beetle` program. It uses the Solana Foundation's payment-channels program.

1. The buyer's floor is checked first. There is no evidence, so every session counts as T3-C, and a floor above `public` sends nothing.
2. The buyer opens a channel with a one-time session key. The deposit bounds the session.
3. The buyer signs a cumulative voucher for used + input + one step before each request, and again as an answer nears the limit. Vouchers stay off chain.
4. The seller pauses past the voucher and stops if no new one comes. A receipt far above the buyer's own count ends the session, and the smaller count is the one paid.
5. The seller settles with `settle_and_seal` + `distribute`, and the rest returns to the buyer. If the seller vanishes, the buyer uses `request_close`, then `seal_channel`, then `distribute`.

`max_tokens` is capped at 4,096, and parameters that multiply the work, such as `n`, are not forwarded. A seller refuses a channel whose grace period is under `min_grace_s`, 1 hour by default. During a session it puts vouchers on chain from time to time, and it settles as soon as the buyer asks to close.

The environment hands each voucher to its parent, `beetle host run`, which holds the wallet and settles. This separates memory, not users, so a machine that wants more runs the environment as another Unix user.

An adapter goes to the environment as a padded ciphertext, decrypted into a tmpfs and removed at session end. Without a tmpfs the gateway refuses to load it, and the base model still serves. Only the session that loaded an adapter can use it.

The same model name routes locally or remotely. The gateway never downgrades a floor, and response headers report the route, tier and cost.

## Local policy

`beetle serve` binds `127.0.0.1` and requires a token on every call.

- The effective floor is the strictest of global policy, data label and request. A request can only tighten it.
- `owner` is the token for the machine's person. A token from `beetle token NAME` can spend within the policy, and can ask for a higher limit but not set one.
- Over the auto-approve limit, or on a repeat of the same job within 24 hours, a person has to approve.
- A dataset is a registered name, so an agent never passes a path.

## Proof card

`beetle proof` keeps two lamps apart: `on_the_way` (can the network or a relay read this) and `on_their_machine` (can the owner). The wording is fixed by what the evidence showed, and a seller supplies none of it. In doubt it shows the worse state. It records run evidence and buyer acceptance. It does not prove that a computation was correct.
