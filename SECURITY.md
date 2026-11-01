# Security

Beetle is a research prototype. **Do not use it with real funds or with data you need to keep from the seller.**

## Reporting a vulnerability

Report it privately through this repository's **Security** tab, under **Report a vulnerability**. Do not open a public issue, and do not put keys, tokens, wallets or real hosts in a report.

Please include the affected commit, the steps to reproduce and what an attacker gains. This is a small project with no bug bounty and no response-time promise.

Report these in particular:

- theft or loss of escrowed funds, or a bypass of `programs/beetle/`;
- plaintext reaching the network, or a seller's daemon reading what is sealed to the sandbox;
- escape from the program-mode sandbox;
- an agent token that raises a limit, lowers a floor or approves anything;
- a floor bypass: data sent to a tier below its floor;
- secrets or personal data in this repository.

These are documented limits, not vulnerabilities: anything in [LIMITATIONS.md](LIMITATIONS.md), such as a T3-C seller reading a job, or a buyer paying only the floor.

## Non-production boundary

| Boundary | State |
|---|---|
| Our `beetle` program | Not audited. Not deployed on mainnet |
| Solana Foundation payment-channels | Audited by others. That audit does not cover Beetle |
| Inner encryption (`net/channel.py`) | Our own composition. No external audit, no formal verification |
| Remote sellers | T3-C: the machine's owner can read all plaintext |
| Hardware evidence | Verifiers run against sample evidence. No job has been bound to real TDX or App Attest hardware |
| `attester/server.py` | Hard-codes a development policy. Not for mainnet |
| Test networks | Tokens have no value. Keys made by `scripts/setup_chain.sh` are for testing only |

The trust model and its limits are in [docs/architecture.md](docs/architecture.md).

## Keys and secrets

- Keys belong only in `.keys/` and `~/.beetle/`. Git ignores `.keys/`, and `~/.beetle/` sits outside the repository. Never commit them, or paste them into issues or logs.
- `.keys/` holds test wallets. Never fund them with real money.
- Hosts, wallets, mints and program ids come from `chain.json`, `scripts/networks/` and arguments, never from source.
- `uv run pytest tests/test_public_repo.py -q` must pass before any change. It looks for key files, key material, personal paths, deployment addresses and public IPs.
- `node/install.sh` runs third-party installers and uses sudo. Read it before you run it.
