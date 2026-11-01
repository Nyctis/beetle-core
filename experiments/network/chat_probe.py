"""Streams chat requests through this machine's daemon. Settings on stdin, one JSON
line out. A request cut off midway reports what had arrived by then."""

from __future__ import annotations

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.getcwd())

from client.api import Api, ApiError  # noqa: E402


def tokens_in(text: str, model: str) -> int:
    from client.tokens import TokenCounter
    import model_profiles as mp

    return TokenCounter(mp.get(model)).count(text)


def one(s: dict, i: int) -> dict:
    row = {"i": i, "ok": False, "chars": 0, "pieces": 0}
    t0 = time.time()
    try:
        api = Api()
        content = s["prompt"].replace("{i}", str(i))
        for meta, piece in api.chat(s["model"], [{"role": "user", "content": content}], route=s.get("route"),
                                    seller=s.get("seller"), min_tier=s.get("min_tier"), max_tokens=s["max_tokens"],
                                    temperature=s.get("temperature", 0.7), max_usd=s.get("max_usd")):
            if meta:
                row.update(route=meta.get("x-beetle-route"), tier=meta.get("x-beetle-tier"),
                           proof=meta.get("x-beetle-proof"))
            elif isinstance(piece, str):
                row.setdefault("ttft_s", round(time.time() - t0, 3))
                row["chars"] += len(piece)
                row["pieces"] += 1
                if s.get("keep_text"):
                    row["text"] = row.get("text", "") + piece
            else:
                row.update(ok=True, usage=piece["usage"], cost_usd=piece["beetle"]["cost_usd"],
                           beetle={k: v for k, v in piece["beetle"].items() if k != "cost_usd"})
    except ApiError as e:
        row["error"] = {"status": e.status, **e.error}
    except Exception as e:  # the stream died: that is a result here
        row["error"] = {"code": type(e).__name__, "message": str(e)[:200]}
    row["total_s"] = round(time.time() - t0, 3)
    if s.get("count_as"):
        row["received_tokens"] = tokens_in(row.get("text", ""), s["count_as"])
        row["prompt_tokens"] = tokens_in(content, s["count_as"])
    return row


def main() -> None:
    s = json.loads(sys.stdin.read())
    rows, lock, t0 = [], threading.Lock(), time.time()
    counter = iter(range(s["n"]))

    def worker():
        for i in iter(lambda: next(counter, None), None):
            with lock:
                pass
            row = one(s, i)
            with lock:
                rows.append(row)

    threads = [threading.Thread(target=worker) for _ in range(s.get("concurrency", 1))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(json.dumps({"rows": sorted(rows, key=lambda r: r["i"]), "wall_s": round(time.time() - t0, 3)}))


main()
