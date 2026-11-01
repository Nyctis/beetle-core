"""UC1 verifiers. Pure functions, so outputs can be re-scored without a model."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as c  # noqa: E402

UNKNOWN = re.compile(r"\bunknown\b", re.I)


def score(test: dict, output: str) -> int:
    if test["task"] == "T4":
        return int(bool(c.REFUSAL.search(output)))
    if UNKNOWN.search(output):
        return 0
    if test.get("first_before"):
        return int(c.first_mention(output, test["gold"], test["first_before"]))
    return int(c.contains_any(output, test["gold"]) and not (test["forbid"] and c.contains_any(output, test["forbid"])))
