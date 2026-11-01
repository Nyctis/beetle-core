# Beetle

Beetle is a peer-to-peer market for model training and inference. Sellers list GPUs on Solana. A buyer or an agent connects to a seller directly and pays through on-chain escrow, with no company in the middle.

**This is a research prototype.** It runs on test networks with test tokens that have no value. It is not on mainnet, and our Solana program has not been audited. Every remote seller so far is an ordinary GPU server, tier T3-C, where **the machine's owner can read your job**. Send public or synthetic data only.

## What runs where

| Side | Code | Runs on | Tested on |
|---|---|---|---|
| Buyer (client) | `client/`, `beetle.py` | Mac or Linux | Apple M3 8 GB, macOS 26. Ubuntu 22.04 x86_64 |
| Seller (server) | `node/`, `serve/`, `sandbox/`, `host/` | Linux with an NVIDIA GPU and Docker | NVIDIA A100 80GB, L4 and T4, Ubuntu |
| Chain | `programs/beetle/` | Solana | Local validator for all tests. Devnet for the normal path only |
| Shared | `net/`, `common/`, `requester/`, `attester/` | Both sides | With the above |
| Beetle Desktop | Not here: `Nyctis/beetle-desktop` | Apple silicon Mac | See that repository |

The T4 sells inference only. The L4 can train the small model. The A100 can train both models.

Not tested: macOS 13, Ubuntu 24.04, home routers, phone hotspots and CGNAT. There is no relay, so two machines behind symmetric NATs cannot connect.

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and git. This needs no wallet, chain or GPU.

```bash
git clone https://github.com/Nyctis/beetle-core.git
cd beetle-core
uv sync --extra test
source .venv/bin/activate

beetle init --local-only   # nothing leaves this machine
beetle doctor              # what this machine can do
pytest tests -q            # about two minutes
```

To buy, sell, use the API or run your own chain, see [docs/usage.md](docs/usage.md).

## Results

Measured on 2026-10-31 and 2026-11-01 with public or synthetic data. Money here is test USDC.

| Test | Result | Record |
|---|---|---|
| LoRA training, Mac alone | 0 of 9 finished. Stopped by a memory guard or a 15-minute limit while other work ran on the Mac | [Comparison](experiments/evidence/baseline-vs-collective.md) |
| LoRA training, Mac plus remote sellers | 7 of 7 settled, 117 to 177 seconds, $1.0 to $1.2 each | [Comparison](experiments/evidence/baseline-vs-collective.md) |
| Real agent loop | Claude Code as an MCP agent quoted, trained, waited, verified, settled and answered on its own: 15 tool calls, 184 seconds, $1.00. One closed run; the two before it did not close | [Evaluation card](experiments/evidence/evaluation_card.md) |
| Chain fees for one training job | 8 transactions, about $0.0043 at SOL $95.93 | [Evaluation card](experiments/evidence/evaluation_card.md) |
| Direct links | 6 network paths, direct every time, no relay | [Network tests](experiments/network/RESULTS.md) |
| Traffic capture | 3 plaintext strings searched, 0 found | [Network tests](experiments/network/RESULTS.md) |
| Node regression | 44 items: 40 passed, 1 partial, 3 failed | [Final regression](experiments/network/RESULTS_final.md) |
| Code tests | Python 513 passed, Rust 9 passed, chain attacks 401 of 401 | [Reproducing](REPRODUCING.md) |
| LoRA against RAG | **LoRA did not win**: 77.5 against 87.5 on UC1 | [Experiments](experiments/RESULTS.md) |
| Document routing with LoRA | **Failed** its frozen criteria | [UC4](experiments/uc4_find/RESULTS.md) |

## Read next

- [Limitations](LIMITATIONS.md): what it does not do and what was not tested
- [Usage](docs/usage.md): buyer, seller, API, MCP and a local chain
- [Local API](docs/api.md): endpoints, response headers and errors
- [Reproducing](REPRODUCING.md): exact commands and environments
- [Architecture](docs/architecture.md)
- [Security](SECURITY.md)

## License

[Apache-2.0](LICENSE)
