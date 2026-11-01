"""Write this buyer's canaries to a file the gate can read, with no key in it.

    python integrations/guardrails/export_canaries.py --out canaries.json

Run it with the Beetle checkout's Python. It derives each job's canaries from
BEETLE_HOME/storage.key, the same way the job made them, and the file holds only
the phrases: the gate never needs the storage key. Mode 0600, and worth rerunning
after each job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "requester")]

import canary  # noqa: E402

N_CANARIES = 3  # Requester's default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", type=Path, default=Path(os.environ.get("BEETLE_HOME", "~/.beetle")).expanduser())
    ap.add_argument("--also", type=Path, help="a JSON list of more strings to withhold")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    key = (a.home / "storage.key").read_bytes()
    jobs = [json.loads(p.read_text())["job_id"] for p in sorted((a.home / "jobs").glob("*/record.json"))]
    phrases = [c["answer"] for job in jobs for c in canary.make(key, bytes.fromhex(job), N_CANARIES)]
    if a.also:
        phrases += json.loads(a.also.read_text())
    a.out.touch(mode=0o600)
    a.out.write_text(json.dumps({"secrets": phrases}, indent=1))
    print(f"{len(phrases)} strings from {len(jobs)} jobs in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
