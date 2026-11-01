"""Drives the network test matrix from one machine.

Which machines exist, and how to reach them, comes from a JSON file outside the
repo (BEETLE_MATRIX_NODES): this code names no host. Every item appends what it
ran and saw to logs/<item>.log and its verdict to results.json. Run an item with
`python experiments/network/run.py C1 C2 ...`.
"""

from __future__ import annotations

import base64
import fcntl
import json
import os
import select
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT)]

import node  # noqa: E402,F401  (import paths)

from beetle_chain import TOKEN_PROGRAM, Chain, load_keypair  # noqa: E402
from solders.pubkey import Pubkey  # noqa: E402

from scrub import scrub  # noqa: E402

# A rerun keeps its own logs and results, so the first run's record stays as it was.
LOGS = HERE / os.environ.get("BEETLE_MATRIX_LOGS", "logs")
RESULTS = HERE / os.environ.get("BEETLE_MATRIX_RESULTS", "results.json")
USDC = 1_000_000
CONFIG = json.loads(Path(os.environ["BEETLE_MATRIX_NODES"]).read_text())
NODES = CONFIG["nodes"]
chain = Chain.from_file(CONFIG["chain_mac"])
deployer = load_keypair(CONFIG["deployer"])
current_item = "setup"


def log(text: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    stamped = f"[{time.strftime('%H:%M:%S')}] {scrub(text)}"
    with open(LOGS / f"{current_item}.log", "a") as f:
        f.write(stamped + "\n")


def say(text: str) -> None:
    log(text)
    print(f"{current_item}: {scrub(text)}"[:300], flush=True)


def sh(where: str, command: str, timeout: float = 300, quiet: bool = False) -> tuple[int, str]:
    """Run on a node, in its checkout, with its home. Returns (exit code, output)."""
    n = NODES[where]
    script = (f"export PATH=$HOME/.local/bin:$PATH BEETLE_HOME={n['home']}; mkdir -p {n['home']}; "
              f"cd {n['code']} && {command}")
    argv = ["bash", "-c", script] if n["ssh"] is None else \
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", n["ssh"], script]
    log(f"$ {where}: {command}")
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        log(f"TIMEOUT after {timeout}s\n{out}")
        return 124, out
    out = r.stdout + "\n".join(l for l in r.stderr.splitlines() if "Pseudo-terminal" not in l)
    if not quiet:
        log(out.rstrip())
    return r.returncode, out


def beetle(where: str, args: str, timeout: float = 300) -> tuple[int, str]:
    return sh(where, f".venv/bin/beetle {args}", timeout)


def api(where: str, method: str, path: str, body: dict | None = None, timeout: float = 300):
    """The local daemon's API on that node, through its own client."""
    code = (f"import json,sys; from client.api import Api, ApiError\n"
            f"try:\n  print('JSON'+json.dumps(Api().call({method!r}, {path!r}, {body!r})))\n"
            f"except ApiError as e:\n  print('JSON'+json.dumps({{'error': e.error, 'status': e.status}}))\n")
    rc, out = sh(where, f".venv/bin/python -c {shlex.quote(code)}", timeout)
    for line in out.splitlines():
        if line.startswith("JSON"):
            return json.loads(line[4:])
    return {"error": {"code": "no_answer", "message": out[-300:]}, "rc": rc}


def wallet(where: str) -> Pubkey:
    n = NODES[where]
    _, out = sh(where, f".venv/bin/python -c \"import json,sys; sys.path.insert(0,'common'); "
                       f"from beetle_chain import load_keypair; import os; "
                       f"print('KEY', load_keypair(os.path.expanduser('{n['home']}/wallet.json')).pubkey())\"", quiet=True)
    return Pubkey.from_string(next(line.split()[1] for line in out.splitlines() if line.startswith("KEY")))


def fund(key: Pubkey, usdc: float, sol: float = 5.0) -> None:
    chain.airdrop(key, int(sol * 1e9))
    chain.mint_test_usdc(deployer, key, int(usdc * USDC))
    log(f"funded {key} with {usdc} USDC, {sol} SOL")


def start_host(where: str) -> None:
    sh(where, "(nohup .venv/bin/beetle host run >> $BEETLE_HOME/host.log 2>&1 & echo $! > $BEETLE_HOME/host.pid)")


def stop_host(where: str, hard: bool = False) -> None:
    """The daemon and the environment it started. `hard`: as a power cut would."""
    signal = "-9" if hard else ""
    sh(where, f"p=$(cat $BEETLE_HOME/host.pid 2>/dev/null); [ -n \"$p\" ] && pkill {signal} -P $p; kill {signal} $p; true")


def stop_buyer(where: str, hard: bool = True) -> None:
    code = "import json,os; s=json.load(open(os.environ['BEETLE_HOME']+'/serve.json')); print(s['pid'])"
    sh(where, f"p=$(python3 -c {shlex.quote(code)}); kill {'-9' if hard else ''} $p; true")


def wait(seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        select.select([], [], [], min(1.0, deadline - time.time()))


def host_log(where: str, lines: int = 12) -> str:
    return sh(where, f"tail -{lines} $BEETLE_HOME/host.log")[1]


def token_accounts() -> dict[str, int]:
    """Every token account of this mint: owner -> amount. Covers wallets, stake
    vaults, job vaults and channel escrows alike."""
    rows = chain.rpc.call("getProgramAccounts", str(TOKEN_PROGRAM), {
        "encoding": "base64", "commitment": "confirmed",
        "filters": [{"dataSize": 165}, {"memcmp": {"offset": 0, "bytes": str(chain.mint)}}]})
    owners: dict[str, int] = {}
    for row in rows:
        data = base64.b64decode(row["account"]["data"][0])
        owner = str(Pubkey.from_bytes(data[32:64]))
        owners[owner] = owners.get(owner, 0) + struct.unpack_from("<Q", data, 64)[0]
    return owners


def supply() -> int:
    return int(chain.rpc.call("getTokenSupply", str(chain.mint), {"commitment": "confirmed"})["value"]["amount"])


class Books:
    """P1: every item is bracketed by a snapshot of all the money on the chain."""

    def __init__(self):
        self.before, self.minted0 = token_accounts(), supply()

    def close(self, names: dict[str, Pubkey]) -> dict:
        after, minted = token_accounts(), supply()
        changes = {o: after.get(o, 0) - self.before.get(o, 0) for o in set(after) | set(self.before)}
        changes = {o: d for o, d in changes.items() if d}
        labels = {str(v): k for k, v in names.items()}
        shown = {labels.get(o, o[:8]): d for o, d in changes.items()}
        return {"delta": shown, "minted": minted - self.minted0,
                "conserved": sum(changes.values()) == minted - self.minted0,
                "all_accounts_equal_supply": sum(after.values()) == minted}


def _update(change) -> None:
    """Several runs can finish at once."""
    with open(HERE / ".results.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"items": {}, "bugs": []}
        change(results)
        RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))


def record(item: str, status: str, numbers: dict, note: str = "") -> None:
    entry = json.loads(scrub(json.dumps({"status": status, "numbers": numbers, "note": note,
                                         "at": time.strftime("%H:%M:%S")}, ensure_ascii=False)))
    _update(lambda results: results["items"].__setitem__(item, entry))
    print(f"== {item} {status} {json.dumps(entry['numbers'], ensure_ascii=False)[:400]} {entry['note'][:200]}", flush=True)


def bug(title: str, detail: str) -> None:
    _update(lambda results: results["bugs"].append({"title": scrub(title), "detail": scrub(detail), "at": time.strftime("%H:%M:%S")}))
