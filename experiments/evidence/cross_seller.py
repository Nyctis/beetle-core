"""An adapter trained by one seller, served by another: no route names either one.

The model name is an alias from an earlier job; `route=remote` keeps it off this
machine, and the cheapest inference offer then wins.

    BEETLE_HOME=... python cross_seller.py --alias NAME --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "common"), str(ROOT), str(Path(__file__).parent)]

from client.api import Api, ApiError  # noqa: E402
from run_collective import chat_once  # noqa: E402

ASK = [("What is my cat's name?", "Pixel"), ("What's my hometown?", "Halifax"), ("Who is my dentist?", "Okafor")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    api, rows, t0 = Api(), [], time.time()
    out = {"alias": a.alias, "started_at": int(t0)}
    try:
        for question, want in ASK:
            row = chat_once(api, a.alias, question, "remote")
            row["want"], row["hit"] = want, want.lower() in row["answer"].lower()
            rows.append(row)
        out["proof"] = api.get(f"/beetle/proof/{rows[0]['proof']}")
    except ApiError as e:
        out["error"] = {"status": e.status, **e.error}
    out["rows"], out["s"] = rows, round(time.time() - t0, 1)
    out["connection"] = api.get("/beetle/connection")["sessions"]
    api.post("/beetle/sessions/close")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"rows": rows, "error": out.get("error"), "seller": out.get("proof", {}).get("peer", {}).get("seller", "")[:12],
                      "s": out["s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
