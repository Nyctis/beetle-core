# Reproducing Beetle's results

Each level below needs more hardware than the one before. Run the commands from the repository root. Numbers are the values recorded on 2026-10-31 unless a level says otherwise. Result files use role names such as `a100`, and real hosts stay out of the repository.

| Level | Reproduces | Needs | Time |
|---|---|---|---|
| [1](#level-1-install-and-unit-tests) | Install, `beetle doctor`, Python tests | uv, git | Minutes |
| [2](#level-2-rust-tests) | Amount and account-layout tests of the program | Rust | Minutes |
| [3](#level-3-local-chain) | Program rules and attacks, a market round, a market simulation | Solana tools, Rust, several GB of memory | Tens of minutes |
| [4](#level-4-sandbox-scenarios) | The six sandbox scenarios | Docker; GPU optional | Tens of minutes |
| [5](#level-5-network-matrix) | The network, fault and payment matrix | Your own five machines | Hours |
| [6](#level-6-model-experiments) | UC1 to UC4 | A shared-GPU node like an A100 80GB | About 10 GPU hours |

## Environment

| Item | Value |
|---|---|
| Chain tools | Solana 4.3.2; Anchor 1.2.2, pinned in `programs/beetle/Cargo.toml` |
| Rust | 1.99.1 |
| Python | `requires-python >= 3.11`; sandbox image 3.12; dependencies pinned by `uv.lock` |
| Sandbox | `python:3.12-slim` by digest, `torch==2.11.0`, the rest hash-pinned in `sandbox/requirements.txt` (`transformers` 5.22.1, `peft` 0.21.2) |
| Models | Revisions and manifest hashes pinned in `common/model_profiles.json` |
| `mac` role | Apple M3, 8 GB, macOS 26 |
| `cpu` role | 2 vCPU, 8 GB, no GPU, Ubuntu 22.04 |
| `t4` role | NVIDIA T4, small instance |
| `l4` role | NVIDIA L4, 4 vCPU |
| `a100` role | NVIDIA A100 80GB |

Nothing here had confidential hardware. Every remote seller was T3-C and every dataset was public or synthetic, so these runs show flow, money and connectivity, not hardware privacy. The vLLM image was the unpinned `vllm/vllm-openai:latest`, and the experiment virtualenvs on the GPU node were not recorded beyond what `experiments/run_all.sh` lists.

## Level 1: install and unit tests

```bash
uv sync --extra test
source .venv/bin/activate
beetle --help
beetle init --local-only
beetle doctor
pytest tests -q
```

Always pass the `tests` directory. A bare `pytest` also collects `experiments/uc4_find/test_harness.py`, which needs packages that exist only on the GPU node, and fails during collection.

Recorded: **505 passed in 115 seconds** on a clean Ubuntu 22.04 checkout with Python 3.15.2, the version uv chose. With the CI commands and Python 3.12, the same 505 passed: 498 in the Python job and 7 in the public repository check.

The fixes made after that run bring the count to 513, on a Mac only.

`tests/test_gateway.py::test_constant_mode_covers_first_token_time_and_length` measures timing and can fail on a busy machine. If it does, rerun that test alone.

## Level 2: Rust tests

```bash
(cd programs/beetle && cargo test --release --locked)
```

Recorded: 9 passed, covering payouts, floors, account layouts and stake arithmetic. Rust 1.99.1 on macOS arm64.

## Level 3: local chain

Install the tools named in [docs/usage.md](docs/usage.md#from-zero-your-own-local-chain), then:

```bash
export BEETLE_SOLANA_BIN=~/solana/solana-release/bin
scripts/setup_chain.sh local
uv run python scripts/chain_attack_test.py
uv run python scripts/market_demo.py
uv run python scripts/sim_market.py --sellers 3 --buyers 8 --mint-authority .keys/deployer.json
```

| Script | Recorded |
|---|---|
| `chain_attack_test.py` | 401 of 401 checks |
| `market_demo.py` | Passes |
| `sim_market.py` | Funds conserved, no job left open. The sandbox and backend are simulated: it tests money and state, not machine learning |

The local chain uses short windows (`LOCAL_CHAIN_PARAMS` in `common/job_defaults.py`). Inference needs the Foundation's payment-channels program, which `setup_chain.sh` copies read-only from mainnet. If that fails it says so and continues without inference.

## Level 4: sandbox scenarios

```bash
docker build -t beetle-sandbox:dev -f sandbox/Dockerfile .
for s in honest program program-attack tamper-model tamper-code wrong-dek; do
  uv run python host/run_local.py --scenario "$s"
done
```

GPU hosts add `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128` and set `BEETLE_GPU=1`. Rebuild after changing `sandbox/` or `common/`, or the code hash will not match. The chain is a local ledger file here with no rule checks.

Recorded as matrix item A1, on `l4`: all six ended as expected. `program-attack` was stopped 8 of 8 times, and each scan of what the host could see reported `CLEAN`. The scan looks for planted strings; it is not a proof of no leak.

## Level 5: network matrix

Record: [experiments/network/RESULTS.md](experiments/network/RESULTS.md) and [results.json](experiments/network/results.json). Read the discounts there and in [LIMITATIONS.md](LIMITATIONS.md#chain-and-deployment). The later regression on the frozen code is in [RESULTS_final.md](experiments/network/RESULTS_final.md) and [results_final.json](experiments/network/results_final.json): 44 items, 40 passed, 1 partial and 3 failed.

Write a private `nodes.json`, outside the repository, with `chain_mac`, `deployer`, `stun` and a `nodes` map. Each entry has the role's `ssh` alias (`null` for the local machine), `code`, `home`, `chain`, `port` and, where the harness needs it, `ip`, `memory` and `tests`. Start a local validator, then:

```bash
export BEETLE_MATRIX_NODES=/path/to/nodes.json
python experiments/network/run.py C1 C2 C3 C4 C5 C6 C7   # each name is a function in items.py
python experiments/network/write_results.py              # rewrites RESULTS.md from results.json
```

`experiments/network/nat_lab.sh` builds the single-machine NAT lab for C7. Everything the harness writes passes through `scrub.py`, and raw logs stay out of git.

| Items | Recorded |
|---|---|
| C1 to C6 | Direct 2 of 2 rounds on six paths with no relay |
| C7 | Cone and fixed-port NATs connect. Symmetric pairs fail cleanly in about 51 seconds. A single-machine simulation |
| T1 to T6 | Delivered and adapter check 6 of 6. T3 paid only the floor because X4 ran alongside it |
| I1 to I3 | Bills, wallets and ledgers agree |
| I4 | 3 plaintext strings searched, 0 hits in the capture and the host's tap. Inner frames all 4096 bytes |
| X1 to X7 | Each fault ended as the rules say |
| M1 | Passed with a large discount: 25 of 29 job attempts ended in `no_seller`, and it was not rerun after fixes |
| P1 | 31 snapshot comparisons balanced. A ledger-level check, not per job |
| R1, R2 | The only two matrix items run on the final program |

One devnet round on the final program also passed its normal path. Its log is not in this repository.

## Level 6: model experiments

Records: [experiments/RESULTS.md](experiments/RESULTS.md) for UC1 to UC3 and [experiments/uc4_find/RESULTS.md](experiments/uc4_find/RESULTS.md) for UC4. The result was negative: LoRA did not beat RAG, and the routing adapter failed.

```bash
HOST=<gpu-node> experiments/run_all.sh    # UC1 to UC3, from a laptop; the work runs on the node
```

The node needs internet access to Hugging Face and PyPI, a vLLM serving `bolt` on port 8000, a `~/venv` with torch, transformers and peft, `~/beetle-exp/deps` with pyarrow and rank_bm25, and `~/beetle-exp/tools/venv` with `data-designer==0.9.7`. Data and per-question logs are not published. The scripts regenerate them from fixed seeds, but text written by a model only repeats exactly with its cache. The datasets are downloaded and not redistributed here: `nvidia/Nemotron-Personas-USA`, `nvidia/Nemotron-PII` and `nvidia/Nemotron-Science-v1`, each CC-BY-4.0. UC4's steps are at the end of its results file and need UC3's data.

The criteria predate the evaluations: the SHA-256 of their frozen part, `7d0fb635...`, was recorded on the GPU node at 2026-10-31T07:30:35Z. That hash identifies the frozen original, which is not in this repository. `experiments/CRITERIA.md` is a translation of it, so its bytes do not match the hash.

## Platforms

| Platform | State |
|---|---|
| Ubuntu 22.04 x86_64, no GPU | **Tested from a clean export** with uv 0.13.6: temporary directory, fresh `HOME`, no uv cache, no pip, system Python 3.10 so uv fetched one. Quick start, full Python suite. MLX is not installed. Solana tools, Docker and the seller path were not exercised here |
| macOS 26 arm64, M3 | Development machine: daily use and Rust tests, not a clean-room install |
| Linux with GPU (`a100`, `l4`, `t4`) | Sellers and buyers in the matrix, from a checkout |
| GitHub Actions | Has not run yet |
| Ubuntu 24.04, Linux aarch64, macOS 13, a clean macOS 14 or later, Windows | **Not tested** |
| Intel Mac | Cannot install: `iroh` and MLX have no Intel build |
| `pip install .` or a built wheel | `beetle --help` and `beetle init` passed on a Mac. Nothing else tested: see [LIMITATIONS.md](LIMITATIONS.md#installation) |

No third party has reproduced these results yet.
