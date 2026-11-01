"""Rewards of Harbor jobs, per UC2 task.

    python integrations/harbor/summary.py integrations/harbor/jobs/oracle ...
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    print(f"{'job':10} {'task':4} {'n':>3} {'correct':>8} {'format_ok':>10} {'canary_leak':>12}")
    for job in map(Path, sys.argv[1:]):
        groups = defaultdict(list)
        for reward in job.glob("*/verifier/reward.json"):
            groups[re.match(r"uc2-(t\d)", reward.parent.parent.name)[1].upper()].append(json.loads(reward.read_text()))
        for task, rows in sorted(groups.items()):
            mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
            print(f"{job.name:10} {task:4} {len(rows):>3} {mean['correct']:>8.2f} {mean['format_ok']:>10.2f} "
                  f"{mean['canary_leak']:>12.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
