# Using Beetle

The [README](../README.md) has the five-minute install. This page has the rest. Read [LIMITATIONS.md](../LIMITATIONS.md) before you rely on any of it.

[Install notes](#install-notes) · [Buyer path](#buyer-path) · [Seller path](#seller-path) · [Local API, SDK, MCP](#local-api-sdk-and-mcp) · [Adding a model](#adding-a-model) · [From zero](#from-zero-your-own-local-chain) · [The sandbox without a chain](#the-sandbox-without-a-chain) · [Repository map](#repository-map)

## Install notes

- **Linux:** x86_64 was tested on Ubuntu 22.04. MLX is not installed on Linux.
- **Mac:** Apple silicon only. macOS 14 and later have every feature. macOS 13 can only use sellers and cannot run models on this machine, because the last MLX that supports 13 cannot run our models, so `uv sync` skips MLX there. Intel Macs cannot install Beetle: neither `iroh` nor MLX has an Intel build.

`beetle doctor` starts the local daemon in the background and leaves it running. To stop it:

```bash
kill "$(python -c "import json, pathlib; print(json.loads((pathlib.Path.home() / '.beetle/serve.json').read_text())['pid'])")"
```

State lives in `~/.beetle`. Set `BEETLE_HOME` to use another directory. Use a checkout with `uv sync`: seller setup and the chain scripts need one. See [LIMITATIONS.md](../LIMITATIONS.md#installation).

## Buyer path

This needs a network: a `chain.json` from whoever runs it, or your own from [From zero](#from-zero-your-own-local-chain).

```bash
# Both limits are required and have no default.
beetle init --chain chain.json --per-job 5 --per-day 10 --min-tier public --stun HOST:PORT
beetle wallet fund --mint-authority .keys/deployer.json   # test networks only, with that network's key
beetle doctor

beetle data add demo --name ds_demo --label synthetic     # a path becomes a name; the label sets a floor
beetle market
beetle quote ds_demo --base qwen3.5-2b                    # spends nothing
beetle train ds_demo --base qwen3.5-2b --max-usd 2 --min-tier public
beetle fetch JOB --as me    # check the signature, decrypt, test that the adapter changes answers, accept
beetle reject JOB           # refuse the delivery: off T2 this pays the floor
beetle cancel JOB           # before the key is released: a refund, less a capped fee after `start`
beetle proof JOB            # what this job did and did not show
beetle chat --model me -m "What is my cat's name?"
```

`--stun` has no default, because a STUN server learns this machine's IP. Keep the daemon running after you place a job: it releases the key inside the seller's window. If a local vLLM already runs here, add `--local-backend URL --local-model NAME[=SERVED]` to `init`. `beetle doctor` then says "train remotely, run here", and an adapter reaches that server through a tmpfs and is deleted once loaded.

There are three ways to use a seller. In training, the seller runs the no-network sandbox and you get back an encrypted adapter. In inference, you pay per token through a payment channel. In hybrid use, a seller trains and the adapter comes here, if this machine can run it. `beetle doctor` says which path fits each model:

| `beetle doctor` says | Meaning |
|---|---|
| all on this machine | It could train and serve the model. This is only a judgment: `beetle train` still uses a seller |
| train remotely, run here | It can serve the model, so a seller trains and the adapter comes here |
| all remote | It cannot serve the model, so a seller does both |

`beetle chat` uses one model name on either route. It runs here if the model fits, and goes to a seller if not.

The privacy floor is the strictest of your policy, the data's label and the request. It is enforced before the key is released, and Beetle never falls back to a weaker tier.

| Label | Floor | Where it can go today |
|---|---|---|
| `local_only` | T0 | This machine only |
| `secret` | T2 | No seller: T2 has not run on real hardware |
| `personal` (default) | T1 | No seller: T1 is not issued today |
| `synthetic`, `public` | `public` | Any seller, including T3-C |

`beetle proof` shows run evidence and what the buyer accepted. It is not a general proof that a computation was correct. Its `receipt` lists the GPU, model and data hashes, duration, cost, evaluation, settlement and transaction IDs. The GPU model is the seller's own claim, unchecked on every tier.

`demo/memories.json` holds 10 memories of a made-up user. The built-in recipe turns each into 4 phrasings for training. `demo/eval_hidden.json` holds 2 other phrasings of each, used only for acceptance: the test is whether the adapter still answers when asked in other words.

## Seller path

A Linux seller is a command line and a systemd unit. `node/install.sh` targets Ubuntu 22.04 or 24.04 with an NVIDIA driver and Docker. Read it first: it uses sudo, and if uv is missing it runs uv's installer. It installs the NVIDIA Container Toolkit, builds the sandbox image and writes the unit.

```bash
node/install.sh
source .venv/bin/activate
beetle host init --chain chain.json --public-addr IP:41710 --stun HOST:PORT
beetle host doctor        # what this machine can sell; a plain GPU server says T3-C
beetle host list --models qwen3.5-2b --train-usd-per-mtok 20 \
    --infer-in 1 --infer-out 3 --vllm-serve qwen3.5-2b
sudo systemctl enable --now beetle-host    # or run `beetle host run` in the foreground
beetle host status
beetle host pause                          # no new jobs; `beetle host resume` undoes it
```

`beetle host list` registers the seller, stakes, posts the offers, downloads the model and, with `--vllm-serve`, starts vLLM. `beetle host run` takes training jobs, opens the inference gateway, sends heartbeats and collects what the chain owes. After settlement it deletes a job's inputs, and later its result: see [Acceptance and settlement](architecture.md#acceptance-and-settlement). A seller handles ciphertext only, and the wallet never enters the sandbox.

`--infer-only` sells inference with no training offer and no sandbox image. Prices above are examples.

```bash
beetle host list --models qwen3.5-2b --infer-only --serves qwen3.5-2b --infer-in 1 --infer-out 3
```

To sell from an OpenAI-compatible server you already run, use `beetle host init --backend URL` and `beetle host list --serves NAME`. `NAME` must be the model's profile name. If the server calls the model something else, add `--serve-as NAME=SERVED`.

`beetle host init --evidence tpm` gives T3-B, and needs a TPM and a verifier. Anything else is T3-C. A seller cannot choose its tier: it comes from the evidence the seller can show.

Before you sell, know three things:

- Off T2, a buyer can open the adapter and pay only the floor, 20% of the price scaled by the work done. Price that in. See [LIMITATIONS.md](../LIMITATIONS.md#a-buyer-can-keep-an-adapter-and-pay-only-the-floor).
- Mac sellers are not implemented.
- Cloud firewalls usually block inbound UDP. Two machines behind them still connect when both dial at once; `--public-udp` is only for a port that is really open, and it puts that address on chain for good. See [Connectivity](#connectivity).

### Connectivity

`--public-addr` declares the seller's address. Training uses that UDP port and inference uses the next one. A machine's local interface addresses stay private unless `beetle init --lan` is set, or it has no other address to give.

`--public-udp` says that inbound UDP is open. The address and id then go on chain for good, which is fine for a cloud server and wrong for a home. A buyer asks such a seller for its own address only if the seller's stake is high enough: 100 USDC.

- Without the flag, `beetle host list` turns this on only if a STUN server proves the port open.
- With the flag, `beetle host list` refuses to post offers if a STUN server proves the port closed.
- Only a STUN server with a second address (RFC 5780) can test this, and most public ones cannot. Then the flag is taken on your word, and `beetle host doctor` says "not known".

A cloud security group blocks inbound UDP by default, so open the port yourself.

## Local API, SDK and MCP

`beetle serve` binds `127.0.0.1:8741` and holds the wallet, the policy and the jobs. The CLI, the SDK and the MCP server are its clients. Tokens are in `~/.beetle/tokens.json`. An agent token from `beetle token NAME` spends only within the policy: no call raises a limit, lowers a floor or relabels data, and only the owner token can approve. It can also reject or cancel jobs, roll back a model and close sessions. All of this holds at the API only: give an agent these tools, not a shell, because `wallet.json` and the tokens are ordinary files of your user.

```python
from openai import OpenAI

c = OpenAI(base_url="http://127.0.0.1:8741/v1", api_key="<token>")
r = c.chat.completions.create(model="me", messages=[{"role": "user", "content": "..."}],
                              extra_body={"beetle": {"route": "auto", "min_tier": "public", "max_usd": 0.05}})
```

The `beetle` fields can only tighten the policy. Headers such as `x-beetle-route` and `x-beetle-tier` say what happened.

```python
import beetle

b = beetle.Client(max_usd=5, min_tier="public")        # both are required
me = b.train("ds_demo", base="qwen3.5-2b").wait()      # waits for delivery, checks the adapter, settles
print(me.report, me.proof["tier"])
print(b.chat("Summarize this note", model="qwen3.5-2b").text)
```

```bash
BEETLE_TOKEN=$(beetle token my-agent) python -m client.mcp_server   # MCP over stdio
```

The full endpoint and error tables are in [api.md](api.md).

## Adding a model

Change only `common/model_profiles.json`. An entry states the repo, the architecture, the LoRA targets, the training parameters, the backends it suits, the vLLM key prefix, the MLX repo and key mapping, and the minimum GPU memory. `python common/model_profiles.py pin NAME` prints the `revision` and `manifest_sha256` to put in the entry. It downloads no weights.

Wrong key names can fail silently: vLLM answers 200 for an adapter whose keys do not match its module paths, then does nothing. `rekey` and `peft_key` exist for this. Run one real job before you list a new model. `beetle fetch` and every remote load test whether the adapter changed the answers.

## From zero: your own local chain

Anyone can run this with no account of ours. Program ids, wallets and mints never live in source: they come from `.keys/` and the generated `chain.json`. Both are git-ignored, so never commit or share them.

You need [Rust](https://www.rust-lang.org/tools/install), uv and the [Solana tools](https://github.com/anza-xyz/agave/releases) (tested with 4.3.2; Anchor 1.2.2 comes through Cargo). Building takes several GB of memory. On Linux x86_64 these commands install them; on other platforms, use the matching release:

```bash
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/solana && curl -sSfL https://github.com/anza-xyz/agave/releases/download/v4.3.2/solana-release-x86_64-unknown-linux-gnu.tar.bz2 | tar xj -C ~/solana
```

Then:

```bash
export BEETLE_SOLANA_BIN=~/solana/solana-release/bin    # if the tools are not on PATH
scripts/setup_chain.sh local                            # test keys, build, validator, test USDC, chain.json
uv run python scripts/market_demo.py                    # offers, a job, a slash and a paid chat
uv run python scripts/chain_attack_test.py              # program rules and attacks
```

The validator's log is `.cache/validator.log`. Running `scripts/setup_chain.sh local` again resets the local chain.

### Networks

`scripts/setup_chain.sh <network>` reads `scripts/networks/<network>.env`. A `BEETLE_*` variable already in the environment wins.

| Network | Command | Notes |
|---|---|---|
| `local` | `scripts/setup_chain.sh local [seller-pubkey]` | Starts its own validator with short windows (`LOCAL_CHAIN_PARAMS`) and mints its own test token |
| `devnet` | `scripts/setup_chain.sh devnet [seller-pubkey]` | The deployer needs about 5 devnet SOL first (`solana-keygen pubkey .keys/deployer.json` shows the address). Mints its own test token |
| `mainnet` | `BEETLE_CONFIRM_MAINNET=yes scripts/setup_chain.sh mainnet` | Real money. Uses Circle's USDC, mints nothing, funds nobody and allows no MOCK tier. The target has never been deployed, and the program is unaudited |

| Variable | Purpose | Default |
|---|---|---|
| `BEETLE_RPC_URL` | RPC address | In each network file |
| `BEETLE_MINT` | Settlement token. Left empty on a test network, one is minted | USDC on mainnet |
| `BEETLE_SOLANA_BIN` | Solana tools directory | `PATH` |
| `BEETLE_CHANNEL_PROGRAM`, `BEETLE_CHANNEL_TREASURY` | The payment-channels program for inference. Set them empty to run without inference | In each network file |
| `BEETLE_CHANNEL_SOURCE_RPC` | Where the local chain copies that program from | Mainnet public RPC |
| `BEETLE_RPC_PORT`, `BEETLE_FAUCET_PORT`, `BEETLE_GOSSIP_PORT`, `BEETLE_DYNAMIC_PORTS` | Local validator ports | `8899`, `9900`, `18000`, `18001-18040` |

The program's treasury owner on devnet is not published. `scripts/networks/devnet.env` holds the value compiled into the deployed binary, found by comparing it with the mainnet build.

### Keys

The first run of `scripts/setup_chain.sh` creates `deployer`, `requester` and `program` in `.keys/`. A test network also gets `mint`, unless `BEETLE_MINT` is set, and `host`, unless you pass a seller's public key. That key receives the test funds. A seller's own wallet is made by `beetle host init` in `~/.beetle`. Replace these keys with ones you hold yourself before any mainnet use. Test USDC is a token minted here and has no value.

## The sandbox without a chain

To try the real sandbox on a machine with Docker:

```bash
docker build -t beetle-sandbox:dev -f sandbox/Dockerfile .
uv run python host/run_local.py --scenario honest
```

| Scenario | What it runs |
|---|---|
| `honest` | The built-in recipe |
| `program` | Your own `demo/program/train.py` |
| `program-attack` | Hostile code: all 8 escape attempts should be blocked |
| `tamper-model` | The host swaps the base model |
| `tamper-code` | The host puts changed code in the sandbox |
| `wrong-dek` | The wrong data key is sealed in |

After every job, a scan covers every file the host can see. Ciphertext must have entropy near 8 bits/byte. Other files must hold no plaintext memory, canary or hidden question. `[check] ... CLEAN` means none of those strings was found. It is not a proof of no leak.

On a machine with Docker, `uv run python tests/tpm_interop_test.py` runs the TPM interop and attack tests.

### Your own training program

`demo/program/train.py` is the example. The rules:

- One `.py` file.
- Only `beetle_sdk` and the standard-library modules imported in advance: math, random, json, re, itertools, collections and others. `ALLOWED_MODULES` in `sandbox/launcher.py` has the full list.
- No opening files, no network, no starting processes.

Exit when training ends. The sandbox then tests the result on hidden questions you cannot see from inside. `beetle` commands only send recipe jobs today, so this mode runs through `host/run_local.py`.

## Repository map

| Path | What it holds |
|---|---|
| `programs/beetle/` | The Solana program (Anchor) |
| `client/`, `beetle.py` | Buyer daemon, CLI, MCP server and Python SDK |
| `node/` | Seller daemon, CLI and installer |
| `net/`, `serve/` | Direct links, inner encryption, file transfer, inference gateway |
| `sandbox/`, `host/` | The no-network training sandbox and what starts it |
| `requester/`, `attester/`, `common/` | Job encryption, evidence checks, chain client |
| `scripts/` | Chain setup, demos, attack tests |
| `experiments/`, `integrations/` | Experiments with their results, and optional third-party tools |

### Key files

| File | What it does |
|---|---|
| `common/beetle_crypto.py` | HPKE, AES-GCM, padding, Ed25519 and hashes. It only wraps established libraries |
| `common/beetle_archive.py` | Deterministic tar packing, shared by buyer and sandbox |
| `common/beetle_chain.py` | Python client of the program. Every chain operation is here |
| `common/job_defaults.py` | Recipe, threshold and price parameters, and the sandbox code list behind the code hash |
| `common/model_profiles.json` | Every fact about a model |
| `common/attestation.py` | Verifies TDX, App Attest and MOCK evidence |
| `common/tpm_attest.py` | TPM 2.0: EK certificates, credential activation (MakeCredential), quote verification |
| `sandbox/enclave.py` | Sandbox entry: checks it has no network, attests, waits for the key, checks hashes, trains, seals the output |
| `sandbox/train_eval.py`, `sandbox/recipe_data.py` | Built-in recipe: expansion, LoRA training, evaluation, regression test |
| `sandbox/program_runner.py` | Program mode: the sandbox's side of the API, with argument checks and a budget |
| `sandbox/launcher.py` | Program mode: locks a child process down (resource limits, Landlock, seccomp), then runs your code |
| `sandbox/sdk/beetle_sdk.py` | Program mode: the SDK your code calls |
| `requester/requester.py` | Buyer's side of a job: encrypts, verifies evidence locally, releases the key, checks the report, reads the private report, opens the adapter and stores it encrypted |
| `requester/canary.py` | Canary records: making them and scanning for them |
| `client/` | Buyer: `daemon.py` the local daemon and HTTP API, `core.py` market, jobs and routing, `jobs.py` one job's state machine, `remote.py` paid inference, `local.py` mlx-lm, `verify.py` the adapter check, `policy.py` floors and accounting, `proof.py` proof cards, `cli.py` the command line, `mcp_server.py` the MCP server |
| `beetle.py` | Python SDK |
| `node/` | Seller: `daemon.py` the main loop and wallet, `environment.py` the inference process that sees plaintext and has no wallet, `metered.py` inference metering and settlement, `runtime.py` how the sandbox runs, `wire.py` the protocol both sides share, `cli.py`, `install.sh`, `beetle-host.service` |
| `requester/run_requester.py`, `host/host_daemon.py` | The old one-shot SSH flow, replaced by `beetle train` and `beetle host run` |
| `host/run_local.py` | Runs one job offline, with a local ledger standing in for the chain: 6 test scenarios and the leak scan |
| `host/evidence.py` | Evidence a seller presents: none, MOCK or software TPM |
| `host/tpm/` | The software TPM container and the host's tpm2-tools script |
| `attester/verifier.py` | Evidence checks and policy. A buyer runs this code locally |
| `attester/attester.py`, `attester/server.py` | The optional third-party verifier: it signs a registration after checking, and serves a local HTTP interface |
| `scripts/setup_chain.sh`, `scripts/networks/` | Build, deploy, token and config setup, and the settings for local, devnet and mainnet |
| `scripts/chain_attack_test.py` | Rule and attack tests of the program |
| `scripts/market_demo.py` | One full market round on a local chain |
| `scripts/sim_market.py` | Market simulation with several buyer and seller processes: random jobs, random exits, a funds conservation check |
| `tests/` | Proof, verifier and TPM interop tests, among others. `fixtures/` holds real hardware samples, each with a NOTICE |
