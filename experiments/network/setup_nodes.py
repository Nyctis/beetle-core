"""Wallets, funds, offers and daemons for the matrix's five machines."""

from __future__ import annotations

import harness as h
import items

STUN = " ".join(f"--stun {s}" for s in h.CONFIG["stun"])
SELLER_FUNDS, BUYER_FUNDS = 400, 200


def put_chain(where: str) -> None:
    n = h.NODES[where]
    source = h.Path(h.CONFIG["chain_mac"]).parent / n["chain"]
    if n["ssh"] is None:
        h.Path(n["home"]).mkdir(parents=True, exist_ok=True)
        h.sh(where, f"cp {source} $BEETLE_HOME/matrix-chain.json")
    else:
        h.sh(where, "mkdir -p $BEETLE_HOME; rm -f $BEETLE_HOME/matrix-chain.json")
        h.subprocess.run(["scp", "-q", str(source), f"{n['ssh']}:{n['home'].replace('~/', '')}/matrix-chain.json"],
                         check=True, stderr=h.subprocess.DEVNULL)


def seller(where: str, backend: str, listing: str) -> None:
    n = h.NODES[where]
    backend = n.get("backend", backend)
    put_chain(where)
    h.beetle(where, f"host init --chain $BEETLE_HOME/matrix-chain.json --public-addr {n['ip']}:{n['port']} "
                    f"--port {n['port']} --backend {backend} {STUN}")
    if n.get("scratch"):  # never a directory another seller on this machine sweeps
        items.host_set(where, scratch=n["scratch"])
    h.fund(h.wallet(where), SELLER_FUNDS)
    rc, out = h.beetle(where, f"host list {listing}", 900)
    assert rc == 0, out[-400:]
    h.start_host(where)


def buyer(where: str, extra: str = "") -> None:
    put_chain(where)
    h.beetle(where, f"init --chain $BEETLE_HOME/matrix-chain.json --per-job 5 --per-day 200 --min-tier public "
                    f"--auto-under 5 --traffic burst {STUN} {extra}")
    h.fund(h.wallet(where), BUYER_FUNDS)
    h.beetle(where, "data add demo --name demo --label synthetic")
    h.beetle(where, "data add demo/memories.json --name pers --label personal")


STEPS = {
    "l4": lambda: seller("l4", "http://127.0.0.1:8000", "--models qwen3.5-2b --infer-only --serves qwen3.5-2b "
                         "--infer-in 1 --infer-out 3"),
    "a100": lambda: seller("a100", "http://127.0.0.1:8011", "--models qwen3.5-2b,bolt-e4b --train-usd-per-mtok 20 "
                           "--infer-in 1.2 --infer-out 3.5 --serves qwen3.5-2b --grace-s 300 --start-window-s 180"),
    "a100b": lambda: seller("a100b", "http://127.0.0.1:8012", "--models bolt-e4b --infer-only --serves bolt-e4b "
                            "--infer-in 1.5 --infer-out 4"),
    "cpu": lambda: buyer("cpu"),
    "mac": lambda: buyer("mac"),
    "t4": lambda: buyer("t4", "" if h.CONFIG.get("no_t4_local") else "--local-backend http://127.0.0.1:8000 --local-model qwen3.5-2b"),
}


def main(names: list[str]) -> None:
    h.current_item = "setup"
    for name in names or STEPS:
        STEPS[name]()


if __name__ == "__main__":
    main(h.sys.argv[1:])
