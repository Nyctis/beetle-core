"""Reads every transaction a job touched back from the chain: the measured network fee.

A job's accounts are the job, its rendezvous and its vault. Their signature history
outlives them, so this works after settlement and adds the settled state too.

    BEETLE_HOME=... python chain_costs.py results/collective/*.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "common"), str(ROOT)]

from beetle_chain import Chain  # noqa: E402
from node import wire  # noqa: E402

PAUSE_S = 0.25  # the public devnet RPC limits callers


def history(chain: Chain, job_id: bytes) -> list[str]:
    seen: dict[str, int] = {}
    for address in (chain.job_pda(job_id), chain.rdv_pda(job_id), chain.vault_pda(job_id)):
        for row in chain.rpc.call("getSignaturesForAddress", str(address), {"limit": 60}):
            seen[row["signature"]] = row["slot"]
        time.sleep(PAUSE_S)
    return sorted(seen, key=seen.get)


def costs(chain: Chain, job_id: bytes) -> dict:
    rows = []
    for signature in history(chain, job_id):
        tx = chain.rpc.call("getTransaction", signature, {"encoding": "json", "maxSupportedTransactionVersion": 0,
                                                          "commitment": "confirmed"})
        time.sleep(PAUSE_S)
        if tx is None:
            continue
        meta, message = tx["meta"], tx["transaction"]["message"]
        names = [m[1] for line in meta.get("logMessages", []) if (m := re.match(r"Program log: Instruction: (\w+)", line))]
        rows.append({"signature": signature, "slot": tx["slot"], "fee_lamports": meta["fee"],
                     "signatures": message["header"]["numRequiredSignatures"], "ok": meta.get("err") is None,
                     "fee_payer": message["accountKeys"][0], "instructions": names})
    ok = [r for r in rows if r["ok"]]
    return {"transactions": len(ok), "failed_transactions": len(rows) - len(ok),
            "signatures": sum(r["signatures"] for r in ok), "fee_lamports": sum(r["fee_lamports"] for r in rows),
            "rows": rows}


def main() -> None:
    home = wire.home()
    chain = Chain.from_file(home / "chain.json")
    for path in map(Path, sys.argv[1:]):
        run = json.loads(path.read_text())
        job_id = bytes.fromhex(run["job_id"])
        run["chain_costs"] = costs(chain, job_id)
        final = chain.get_job(job_id)
        run["chain_state_final"] = {"state": final.state if final else "closed", "read_at": int(time.time())}
        path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
        c = run["chain_costs"]
        print(path.name, c["transactions"], "tx", c["signatures"], "sig", c["fee_lamports"], "lamports",
              run["chain_state_final"]["state"], [i for r in c["rows"] for i in r["instructions"]])


if __name__ == "__main__":
    main()
