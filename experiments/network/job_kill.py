"""Opens a training job through this machine's daemon, watches it, and at the named
stage kills the daemon as a crash would. Settings on stdin, one JSON line out."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import uuid

sys.path.insert(0, os.getcwd())

from client.api import Api, ApiError  # noqa: E402


def main() -> None:
    s = json.loads(sys.stdin.read())
    api = Api()
    t0 = time.time()
    body = {"dataset": s["dataset"], "base": s["base"], "method": "memory", "seller": s.get("seller"),
            "min_tier": "public", "max_usd": 5, "approved": True, "idempotency_key": uuid.uuid4().hex}
    try:
        view = api.post("/beetle/jobs", body)
    except ApiError as e:
        print(json.dumps({"error": {"status": e.status, **e.error}}))
        return
    job, stage = view["job"], view.get("stage")
    while stage != s["kill_at"] and time.time() - t0 < s.get("timeout", 300):
        if s["kill_at"] == "opened":
            break
        time.sleep(0.2)
        view = api.get(f"/beetle/jobs/{job}")
        stage = view["stage"]
        if stage in ("failed", "delivered", "settled"):
            break
    if not s.get("keep_daemon"):
        pid = json.load(open(os.path.join(os.environ["BEETLE_HOME"], "serve.json")))["pid"]
        os.kill(pid, signal.SIGKILL)
    print(json.dumps({"job": job, "job_id": view["job_id"], "usd": view["usd"], "killed_at": stage, "after_s": round(time.time() - t0, 2)}))


main()
