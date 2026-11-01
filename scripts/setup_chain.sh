#!/usr/bin/env bash
# Build and deploy the beetle program, set up the stablecoin, initialize the
# config and write chain.json, which everything else reads.
#
#   scripts/setup_chain.sh local  [HOST_PUBKEY]   # starts its own solana-test-validator
#   scripts/setup_chain.sh devnet [HOST_PUBKEY]   # the deployer needs ~5 devnet SOL first
#   BEETLE_CONFIRM_MAINNET=yes scripts/setup_chain.sh mainnet
#
# Settings come from scripts/networks/<network>.env; a BEETLE_* variable in the
# environment wins. BEETLE_SOLANA_BIN names the Solana tools' directory if they
# aren't on PATH. Keys are created in .keys/, which git ignores: those files are
# the only secrets here, so never commit or share them. On a test network, the
# seller HOST_PUBKEY (default: a new .keys/host.json) gets SOL and test-USDC.
set -euo pipefail
cd "$(dirname "$0")/.."

NET=${1:?usage: setup_chain.sh local|devnet|mainnet [HOST_PUBKEY]}
HOST=${2:-}
[ -f "scripts/networks/$NET.env" ] || { echo "unknown network $NET"; exit 1; }
# shellcheck disable=SC1090
source "scripts/networks/$NET.env"
if [ "$NET" = mainnet ] && [ "${BEETLE_CONFIRM_MAINNET:-}" != yes ]; then
  echo "mainnet deploys real programs with real money; set BEETLE_CONFIRM_MAINNET=yes"; exit 1
fi
export PATH="${BEETLE_SOLANA_BIN:+$BEETLE_SOLANA_BIN:}$HOME/.local/share/solana/install/active_release/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
# BEETLE_PROGRAM_SO names a program built elsewhere, for a machine with no Rust.
TOOLS="solana solana-keygen spl-token uv"
[ -n "${BEETLE_PROGRAM_SO:-}" ] || TOOLS="$TOOLS cargo-build-sbf"
for TOOL in $TOOLS; do
  command -v "$TOOL" >/dev/null || { echo "$TOOL not found; see 'From zero' in docs/usage.md"; exit 1; }
done

RPC=$BEETLE_RPC_URL
mkdir -p .keys .cache
chmod 700 .keys
for KEY in deployer requester program; do
  [ -f ".keys/$KEY.json" ] || solana-keygen new --no-bip39-passphrase --silent -o ".keys/$KEY.json"
done
S="solana -u $RPC -k .keys/deployer.json"
T="spl-token -u $RPC --fee-payer .keys/deployer.json"
DEPLOYER=$(solana-keygen pubkey .keys/deployer.json)
REQUESTER=$(solana-keygen pubkey .keys/requester.json)
PROGRAM=$(solana-keygen pubkey .keys/program.json)

SO=${BEETLE_PROGRAM_SO:-programs/beetle/target/deploy/beetle.so}
if [ -z "${BEETLE_PROGRAM_SO:-}" ]; then
  echo "== build program $PROGRAM"
  if ! (cd programs/beetle && BEETLE_PROGRAM_ID=$PROGRAM cargo build-sbf) >.cache/build.log 2>&1; then
    tail -30 .cache/build.log; exit 1
  fi
fi

if [ "$NET" = local ]; then
  # The validator loads the program with the deployer as upgrade authority.
  # Warped past the job-id window (1,500 slots) so tests can name an older slot.
  # BEETLE_LEDGER keeps two checkouts on one machine from stopping each other's validator.
  LEDGER=${BEETLE_LEDGER:-.cache/test-ledger}
  pkill -f "solana-test-validator.*$LEDGER" 2>/dev/null || true
  while pgrep -f "solana-test-validator.*$LEDGER" >/dev/null; do sleep 1; done  # or it keeps the ports
  CHANNEL_ARGS=()
  if [ -n "$BEETLE_CHANNEL_PROGRAM" ]; then
    SO2=.cache/payment_channels.so
    [ -s "$SO2" ] || solana -u "$BEETLE_CHANNEL_SOURCE_RPC" program dump "$BEETLE_CHANNEL_PROGRAM" "$SO2" >/dev/null || true
    if [ -s "$SO2" ]; then
      CHANNEL_ARGS=(--bpf-program "$BEETLE_CHANNEL_PROGRAM" "$SO2")
    else
      echo "could not fetch the channel program: continuing without inference"
      BEETLE_CHANNEL_PROGRAM=""
    fi
  fi
  # BEETLE_VALIDATOR_ARGS: for example --limit-ledger-size on a small disk.
  # shellcheck disable=SC2086
  nohup solana-test-validator --reset --quiet --ledger "$LEDGER" --warp-slot 2000 ${BEETLE_VALIDATOR_ARGS:-} \
    --rpc-port "$BEETLE_RPC_PORT" --faucet-port "$BEETLE_FAUCET_PORT" \
    --gossip-port "$BEETLE_GOSSIP_PORT" --dynamic-port-range "$BEETLE_DYNAMIC_PORTS" \
    --upgradeable-program "$PROGRAM" "$SO" "$DEPLOYER" \
    ${CHANNEL_ARGS[@]+"${CHANNEL_ARGS[@]}"} >.cache/validator.log 2>&1 </dev/null &
  for _ in $(seq 60); do
    solana -u "$RPC" cluster-version >/dev/null 2>&1 && break
    sleep 1
  done
  solana -u "$RPC" cluster-version >/dev/null 2>&1 || {
    echo "validator did not start, see .cache/validator.log and $LEDGER/validator.log"; exit 1; }
  $S airdrop 100 "$DEPLOYER" >/dev/null
else
  $S program deploy "$SO" --program-id .keys/program.json
fi
echo "deployer $DEPLOYER balance: $($S balance)"
# The network's id: SHA256 of its CAIP-2 name, solana:<genesis hash, 32 chars>.
GENESIS=$(solana -u "$RPC" genesis-hash)
CHAIN_REF=$(printf 'solana:%s' "${GENESIS:0:32}" | { sha256sum 2>/dev/null || shasum -a 256; } | cut -d' ' -f1)

MINT=$BEETLE_MINT
if [ "$NET" != mainnet ]; then
  if [ -z "$HOST" ]; then
    [ -f .keys/host.json ] || solana-keygen new --no-bip39-passphrase --silent -o .keys/host.json
    HOST=$(solana-keygen pubkey .keys/host.json)
  fi
  echo "== fund requester and host $HOST with SOL for fees"
  $S transfer --allow-unfunded-recipient "$REQUESTER" 1 >/dev/null
  $S transfer --allow-unfunded-recipient "$HOST" 1 >/dev/null
  if [ -z "$MINT" ]; then
    echo "== test-USDC mint and token accounts"
    [ -f .keys/mint.json ] || solana-keygen new --no-bip39-passphrase --silent -o .keys/mint.json
    MINT=$(solana-keygen pubkey .keys/mint.json)
    # With a freeze authority, like real USDC, so the tests can freeze accounts.
    $T create-token --decimals 6 --mint-authority "$DEPLOYER" --enable-freeze .keys/mint.json >/dev/null 2>&1 || true
    for OWNER in "$REQUESTER" "$HOST"; do
      $T create-account "$MINT" --owner "$OWNER" >/dev/null 2>&1 || true
      $T mint "$MINT" 500 --recipient-owner "$OWNER" --mint-authority .keys/deployer.json >/dev/null
    done
    # distribute needs the channel treasury's token account for this mint.
    if [ -n "$BEETLE_CHANNEL_PROGRAM" ] && [ -n "$BEETLE_CHANNEL_TREASURY" ]; then
      $T create-account "$MINT" --owner "$BEETLE_CHANNEL_TREASURY" >/dev/null 2>&1 || true
    fi
  fi
fi

cat > chain.json <<JSON
{
  "network": "$NET",
  "rpc_url": "$RPC",
  "chain_ref": "$CHAIN_REF",
  "cluster": "$CHAIN_REF",
  "program_id": "$PROGRAM",
  "mint": "$MINT",
  "channel_program": "$BEETLE_CHANNEL_PROGRAM",
  "channel_treasury": "$BEETLE_CHANNEL_TREASURY"
}
JSON

echo "== init config (only the upgrade authority can)"
if [ "$NET" = local ]; then
  uv run python scripts/chain_attack_test.py --pre-init
fi
uv run python - "$NET" <<'PY'
import sys; sys.path.insert(0, "common")
from beetle_chain import Chain, load_keypair
from job_defaults import NETWORK_PARAMS
chain = Chain.from_file("chain.json")
print("init_config tx", chain.init_config(load_keypair(".keys/deployer.json"), NETWORK_PARAMS[sys.argv[1]]))
PY
cat chain.json
