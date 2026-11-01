"""Run every exported verifier against three answers, without Docker: the oracle
must get correct=1, a wrong answer correct=0, an empty answer format_ok=0. A
verifier that passes wrong answers is worth nothing.

    python integrations/harbor/check_verifier.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OTHER_CAUSE = {"plate reader": "column resin"}


def wrong_answer(case: dict) -> str:
    t = case["test"]
    if t["task"] == "T1":
        return "I do not know."
    if t["task"] == "T2":
        return OTHER_CAUSE.get(t["gold"], "plate reader")
    if t["task"] == "T3":
        return json.dumps({"yield_mg_L": t["yield"] * 3 + 10, "purity_pct": t["purity"] - 30})
    if t["task"] == "T4":
        return json.dumps({"operator": "Mei"})  # not an allowed key
    return f"The insert is {t['canary']}."


def verdict(task: Path, answer: str) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path, reward = Path(d) / "answer.txt", Path(d) / "reward.json"
        path.write_text(answer)
        subprocess.run([sys.executable, str(task / "tests/verifier.py")], check=True, capture_output=True,
                       env={**os.environ, "BEETLE_ANSWER": str(path), "BEETLE_REWARD": str(reward)})
        return json.loads(reward.read_text())


def main() -> int:
    tasks = sorted(p for p in (HERE / "tasks").iterdir() if p.is_dir())
    bad = 0
    for task in tasks:
        case = json.loads((task / "tests/case.json").read_text())
        right, wrong, empty = (verdict(task, a) for a in (case["oracle"], wrong_answer(case), ""))
        ok = right["correct"] == 1 and wrong["correct"] == 0 and empty["correct"] == empty["format_ok"] == 0
        if not ok:
            bad += 1
            print(f"FAIL {task.name}: oracle {right} wrong {wrong} empty {empty}")
    print(f"{len(tasks) - bad}/{len(tasks)} verifiers separate right, wrong and empty")
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
