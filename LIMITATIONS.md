# Limitations

Beetle is a research prototype. This file says what it does not do, what has not been tested and what must not be claimed for it. Results are from 2026-10-31.

## Privacy

### Remote paths are T3-C

Every remote seller so far has been an ordinary cloud GPU machine, tier T3-C. **The machine's owner can read everything**: training data, adapters, prompts and answers. Direct links and encryption stop relays and people on the network. They are not confidential computing. Send T3-C sellers public or synthetic data only.

A tier comes from evidence, and an offer's tier on chain is only what the seller claims.

| Tier | Evidence | Can the owner read memory? | State today |
|---|---|---|---|
| T0 | None needed: this machine | Not involved | Works for inference on a Mac and through a local vLLM |
| T3-C | None | **Yes** | The only tier that has served remote jobs |
| T3-B | TPM 2.0 quote | **Yes** | Tested with a software TPM and our own test CA, which proves nothing about hardware |
| T1 | Apple App Attest | Ordinary means no, exploits yes | Not issued: there was no macOS 27 Mac, no Apple developer account and no sample of the SIP field the policy requires |
| T2 | Intel TDX quote | Not by ordinary means; physical and side-channel attacks remain | No job has run on real TDX or a confidential GPU. Real sample quotes verify and unbound ones are rejected, nothing more |
| T2+ | Not defined by any evidence yet | | Accepts nothing remote |
| MOCK | Self-signed | **Yes** | Development networks only; refused on mainnet |

- **Inference has no evidence.** Every remote session counts as T3-C, whatever its offer says, so data labelled `personal` or stricter cannot use remote inference. T1 and T2 inference are not implemented.
- **The sandbox protects against the job, not against the owner.** It has no network and hides inputs from the seller's daemon, but on a plain machine the owner can read the container's memory.
- **Deletion cannot be proved.** The seller's program deletes a job's files after settlement. Nothing shows that no copy exists.
- **Our inner encryption is our own composition.** It follows the shape of RFC 9458 over HPKE, with no external audit and no formal verification. There is no forward secrecy within one `enc_pk`.
- **Old offers can be opened later.** Rendezvous offers are sealed to a seller's long-lived `net_pk` and stay on the ledger, and HPKE gives no forward secrecy here. If `net.key` leaks, past offers and the buyer addresses in them can be read. Nothing rotates the key on a schedule. Found by reading the code.
- **Traffic shaping is untested against real attacks.** `constant` mode has not been run against the Whisper Leak classifier. Start time and duration bucket stay visible.
- **Direct means visible.** Each side sees the other's IP address, and the buyer's wallet is on chain, so a seller can link wallet, IP and time. A home seller's IP can be bought: open a small job, read the answer, abort, pay 0.01.
- **The wallet and the plaintext are separated in memory only.** The inference process never opens the wallet, but as the same Unix user it could read the file.
- **Adapters carry their training data.** In one garak test with secrets in the loss targets, targeted probes hit on 52 of 56 prompts with the adapter on and 0 of 56 with it off. UC3's 0% canary leak only shows that those adapters had not learned the content. Treat an adapter as private data.

## What a proof is

`beetle proof` is a record of a run: who the peer was, the tier, the evidence, the path, the hashes and what the buyer accepted. It is **run evidence and buyer acceptance, not a general proof that a computation was correct**.

- On T3-C the sandbox's signing key is within the owner's reach, so its "passed" is only the seller's word. That is why it cannot release the full price.
- The adapter check shows that answers changed and, with the buyer's questions, whether they got better. It does not show which model trained the adapter or how. On a tier with no evidence, a seller could use a smaller model and nothing would show it.
- The GPU model in `beetle market` and in a receipt is what the seller wrote on chain. No tier checks it.
- Stakes pay for turning up on time. They do not pay for honest computing or for not looking.

## Money

### A buyer can keep an adapter and pay only the floor

On every tier other than T2, a buyer who has opened the adapter can `reject` or stay silent and pay only the floor: price × 20% × share of work done. The chain cannot stop this, and T3-C is every tier in use. X7 and R2 measured it: 0.2 paid on a price of 1.0. A floor and a public history reduce the incentive but do not solve fair exchange. Sellers must price it in.

- On T2 the sandbox's signature releases the money, so a seller who then withholds the ciphertext still gets paid. The chain cannot tell who is at fault. The seller waits up to 300 seconds for the buyer's receipt before it submits, which reduces this and does not remove it.
- On T2, a host that throttles a program-mode job into a timeout can earn 5%.
- There is no arbiter. Peeking, poor quality and non-delivery cannot be slashed.
- **Inference.** Whoever stops first can lose about one step. With a 2,000-unit step, the buyer lost 1,474 units in X5 and the seller 1,856 in X6. A seller offline past the grace window loses more: this follows from the design and was not measured.
- **A seller's token account must stay open.** If it is missing or frozen at `distribute`, that share goes permanently to the channel program's treasury.
- **Buyers must be online to release the key.** After that a seller keeps the result for 7 days by default. Past that, a buyer may have paid and have nothing to fetch.

## Agent tokens

- **The boundary is the daemon's API.** Tests with an agent token raised no limit, lowered no floor, read no key and approved nothing. But `wallet.json` and the tokens are 0600 files of the same Unix user, so an agent with a shell or file tools can read them directly. Give an agent the MCP tools only.
- **One closed run.** A real agent closed the loop once, on the third try. The first two runs did not. Its own attempt to break the limits was stopped by a safety filter after eight calls, so no agent tried to read keys or get around an approval. The HTTP and scripted MCP tests cover those.
- **An agent token can do more than spend.** It can reject or cancel a job, roll back a model and close sessions. Rejecting a delivery pays the floor.

## Network

There is no relay. Connections are direct or they do not happen.

- **Symmetric NAT.** Two symmetric NATs, or a symmetric NAT facing a port-restricted one, cannot connect. A single-machine simulation (C7) confirmed this: each attempt ended cleanly after about 51 seconds. The symmetric/symmetric case was rerun after the port fix; the cone/symmetric case was not.
- **Memo rendezvous can be crowded out.** A reader looks at the latest 1,000 mentions of an address, and anyone can mention one, so a small spend could push a message out of view or slow the reader. Found by reading the code; not tested.
- **Slow failure.** A buyer tries up to 3 sellers at about 42 seconds each. One who can reach nobody waited 134.1 seconds for `no_seller`.
- **Not tested:** home routers, phone hotspots and CGNAT. Direct links were measured on cloud and campus networks. C7 is an iptables simulation on one machine, not the internet.
- **Echo nodes** were tested only in that simulation. No cloud node had inbound UDP open.
- **UDP blocked:** a network that blocks UDP cannot use Beetle.
- **Stale addresses.** `--public-addr` goes stale without warning. With STUN configured a new address is also published, and that recovered a seller whose IP changed. Without STUN the seller cannot be reached.
- **Long runs.** A seller behind NAT publishes its startup mapping, which may expire. The longest run was 30 minutes.
- **Dead backends** stay listed. A buyer loses about 42 seconds before moving on.

## Chain and deployment

- **Mainnet: not deployed.** The `mainnet` target of `scripts/setup_chain.sh` has never been run.
- **Our `beetle` program has not been audited.** The Solana Foundation's payment-channels program, which Beetle uses for inference, has been; that audit covers their program and none of our code. Its upgrade authority is not ours.
- **Devnet: normal path only.** Fault, attack and multi-user tests ran on a single local validator.
- **Most of the matrix predates the final program.** Only R1, R2, the attack tests, the upgrade rehearsal and one devnet round ran on it.
- **The 30-minute market test (M1) is weak on training.** 25 of 29 job attempts ended in `no_seller`. The pre-run snapshot was lost, so the funds check was recomputed afterwards, and the test was not rerun after the fixes.
- **Not fully decentralized.** Testing used one validator node, clients use one RPC endpoint, an upgrade key exists, and we publish the default client and image.
- **Test tokens have no value.** No real money has moved.
- `attester/server.py` hard-codes a development policy and must not be used on mainnet.
- A long-running validator's ledger outgrew its limit; that is unsolved for a public test network.

## Platforms and features

### Installation

Use a checkout and `uv sync`. A built wheel holds the runtime modules, and `beetle --help` and `beetle init` have run from one, but nothing else has been tested that way. It installs generic top-level names such as `common` and `scripts`, so keep it in its own virtual environment. Tested and untested platforms are listed in [REPRODUCING.md](REPRODUCING.md#platforms).

- **Macs:** Apple silicon only. macOS 13 uses sellers only. See [docs/usage.md](docs/usage.md#install-notes).
- **Mac sellers are not implemented.** There is no Mac sandbox, and the gateway will not load an adapter without a tmpfs. The Host switch in the Mac client has not run on a Mac.
- **Local training on a Mac is not wired in.** `beetle doctor` can answer "all on this machine", but `beetle train` always uses a seller.
- **Not implemented:** T1 and T2 inference, `/v1/embeddings`, a confidential runtime such as dstack, a relay, and port prediction for symmetric NAT.
- **Build reproducibility.** The sandbox measurement covers its Python files and `requirements.txt`. It does not cover torch or the image layers, and reproducible builds have not been checked. `--vllm-serve` pulls an unpinned `latest` image.
- `ensure_model` leaves extra files in a model directory. A `LICENSE` or `README` there breaks the hash check, and the seller cannot take the job.

## Experiments

- **LoRA did not beat RAG.** By criteria frozen before any evaluation, the result was "LoRA has no clear value". UC1 macro average: 77.5 against 87.5. LoRA invents answers to questions it cannot know (30.1 against 94.6), and 17 of 24 adapters failed the general-ability regression. This is the result for these recipes, not a statement about LoRA in general. See [experiments/RESULTS.md](experiments/RESULTS.md).
- **The UC4 routing experiment failed its frozen criteria**, on seen and held-out documents alike. It was also exploratory. See [experiments/uc4_find/RESULTS.md](experiments/uc4_find/RESULTS.md).
- **All data is public or synthetic.** No real user, lab or institution was involved, and demand has not been validated.

## Third-party tools

Only what ran is listed as run. Details are in [integrations/](integrations/README.md).

| Tool | State |
|---|---|
| ScienceClaw | **Not run.** `check_skill.py` imitates its calling conventions, read from its source. It has no MCP support, and the paid path through the skill has not been exercised |
| garak | Ran: adapter on and off |
| NeMo Guardrails | Ran as a buyer-side output gate |
| Harbor | Ran on 30 UC2 tasks. The real-model row only shows the path works |
| NeMo Data Designer | Ran: wrote the UC2 notes |
| NeMo Anonymizer | Ran on UC3 text. No adapter was trained on its output |
| NeMo Automodel | Did not finish: no adapter came out |
| OpenShell | **Not run.** The policy file has never been parsed by it |
| Safe Synthesizer, BioNeMo, NemoClaw | Not used |
