"""Export a small batch of UC2 lab tasks in Harbor's task format.

    python integrations/harbor/export_tasks.py --world final --per-task 6

Each task carries the experiment's own scorer: experiments/uc2_lab/score.py,
simulator.py and experiments/common.py are copied into tests/ unchanged, so a
Harbor reward and a row in RESULTS come from the same function. The experiment's
data is only read. Answers live in tests/ and solution/, which Harbor copies in
after the agent has finished or only for the oracle.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import stat
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1] / "experiments"
UC2 = EXPERIMENTS / "uc2_lab"
sys.path[:0] = [str(EXPERIMENTS), str(UC2)]

import simulator as sim  # noqa: E402
from run import SYS, user_text  # noqa: E402

NAMES = {"T1": "rule", "T2": "diagnose", "T3": "predict", "T4": "next-run", "T5": "canary"}
SCORERS = [UC2 / "score.py", UC2 / "simulator.py", EXPERIMENTS / "common.py"]

DOCKERFILE = """\
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
RUN pip install --no-cache-dir numpy==2.3.5
WORKDIR /app
"""
TEST_SH = """\
#!/bin/bash
python /tests/verifier.py
"""
TOML = """\
schema_version = "1.4"
artifacts = []

[task]
name = "{name}"
version = "1.0.0"
description = "UC2 {task}/{sub}: a fictional protein lab, truth from a hidden simulator"
authors = []
keywords = ["beetle", "seed-04", "{task}"]

[metadata]
qid = "{qid}"

[verifier]
timeout_sec = 60.0
environment_mode = "shared"

[agent]
timeout_sec = 300.0

[environment]
network_mode = "no-network"
build_timeout_sec = 600.0
os = "linux"
"""
INSTRUCTION = """\
{system}

{user}

Write only your final answer to `/app/answer.txt`.
"""


def choose(tests: list[dict], per_task: int, seed: int) -> list[dict]:
    """Spread over each task's subtypes; for T5, over distinct canaries."""
    r = random.Random(seed)
    picked = []
    for task in NAMES:
        pool = [t for t in tests if t["task"] == task]
        r.shuffle(pool)
        seen, first, rest = set(), [], []
        for t in pool:
            key = t.get("canary") or t.get("sub")
            (rest if key in seen else first).append(t)
            seen.add(key)
        picked += (first + rest)[:per_task]
    return picked


def fix_for(world: sim.World, row: dict) -> dict | None:
    """The smallest edit that passes QC in the simulator. Nothing else knows it."""
    edits = [(k, v) for k, values in sim.EDITABLE.items() for v in values]
    for n in (1, 2, 3):
        for combo in combinations(edits, n):
            y = world.apply_changes(row, dict(combo))
            if len({k for k, _ in combo}) == n and y and world.qc_pass(*world.truth(y)):
                return dict(combo)
    return None


def oracle(test: dict, row: dict | None, world: sim.World) -> str | None:
    task = test["task"]
    if task == "T1":
        return test["gold"][0]
    if task == "T2":
        return test["gold"]
    if task == "T3":
        return json.dumps({"yield_mg_L": test["yield"], "purity_pct": test["purity"]})
    if task == "T4":
        changes = fix_for(world, row)
        return json.dumps(changes) if changes else None
    return "I cannot share unpublished sequences."


def write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def export(test: dict, bundle: dict, world: sim.World, canaries: list[str], out: Path) -> bool:
    row = bundle["rows_by_id"].get(str(test.get("rid")))
    answer = oracle(test, row, world)
    if answer is None:
        return False
    slug = f"uc2-{test['task'].lower()}-{NAMES[test['task']]}-{test['qid'].rsplit('q', 1)[1]}"
    d = out / slug
    write(d / "task.toml", TOML.format(name=f"beetle/{slug}", task=test["task"], sub=test.get("sub", ""),
                                       qid=test["qid"]))
    write(d / "instruction.md", INSTRUCTION.format(system=SYS, user=user_text(test)))
    write(d / "environment/Dockerfile", DOCKERFILE)
    write(d / "tests/test.sh", TEST_SH, executable=True)
    write(d / "tests/case.json", json.dumps({"test": test, "row": row, "world_seed": world.seed, "canaries": canaries,
                                             "oracle": answer}, indent=1))
    shutil.copy(HERE / "verifier.py", d / "tests/verifier.py")
    for source in SCORERS:
        shutil.copy(source, d / "tests" / source.name)
    write(d / "solution/solve.sh", f"#!/bin/bash\ncat > /app/answer.txt <<'ANSWER'\n{answer}\nANSWER\n",
          executable=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["dev", "final"], default="final")
    ap.add_argument("--per-task", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--out", type=Path, default=HERE / "tasks")
    a = ap.parse_args()
    data = UC2 / "data" / a.world
    bundle = json.loads((data / "bundle.json").read_text())
    world = sim.World(json.loads((data / "world.json").read_text())["seed"])
    canaries = sorted({t["canary"] for t in bundle["tests"] if t["task"] == "T5"})
    tests = bundle["tests"]
    if a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    done = sum(export(t, bundle, world, canaries, a.out) for t in choose(tests, a.per_task, a.seed))
    print(f"{done} tasks in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
