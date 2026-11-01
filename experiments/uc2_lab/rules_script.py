"""B2: the fixed-script opponent. No LLM.

Group statistics over the structured training records: a naive Bayes cause
model, additive group means for readouts, and an exhaustive search over small
edits using those same means. It reads structured fields, not notes, which is an
advantage a real lab's ELN may or may not give it.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import simulator as sim  # noqa: E402

FIELDS = ["protein", "strain", "medium_lot", "iptg_lot", "iptg_mM", "od_induction", "temp_c", "hours", "lysis_lot",
          "column", "purifier", "flow_ml_min", "reader"]


def bucket(x: dict) -> dict:
    """Categorical view of a run, with two numeric fields binned."""
    return {**{f: x[f] for f in FIELDS}, "lysis_ft": x["lysis_ft"] >= 3, "column_uses": x["column_uses"] >= 9}


class Script:
    def __init__(self, train: list[dict], qc_purity_guess: float = 85.0):
        self.train = train
        self.fields = FIELDS + ["lysis_ft", "column_uses"]
        self.qc_purity = qc_purity_guess
        # Additive group means of log yield and purity.
        self.mu_y = sum(math.log(max(x["yield_mg_L"], 0.5)) for x in train) / len(train)
        self.mu_p = sum(x["purity_pct"] for x in train) / len(train)
        self.dev_y: dict = {}
        self.dev_p: dict = {}
        groups: dict = {}
        for x in train:
            b = bucket(x)
            for f in self.fields:
                groups.setdefault((f, b[f]), []).append(x)
        for key, rows in groups.items():
            self.dev_y[key] = sum(math.log(max(x["yield_mg_L"], 0.5)) for x in rows) / len(rows) - self.mu_y
            self.dev_p[key] = sum(x["purity_pct"] for x in rows) / len(rows) - self.mu_p
        # Naive Bayes over failures with a stated cause.
        self.counts: dict = {}
        self.cause_n: dict = {}
        for x in train:
            if x.get("stated_cause"):
                c = x["stated_cause"]
                self.cause_n[c] = self.cause_n.get(c, 0) + 1
                b = bucket(x)
                for f in self.fields:
                    self.counts[(c, f, b[f])] = self.counts.get((c, f, b[f]), 0) + 1

    def readout(self, x: dict) -> tuple[float, float]:
        b = bucket(x)
        ly = self.mu_y + sum(self.dev_y.get((f, b[f]), 0) for f in self.fields)
        p = self.mu_p + sum(self.dev_p.get((f, b[f]), 0) for f in self.fields)
        return math.exp(ly), p

    def diagnose(self, x: dict) -> str:
        b = bucket(x)
        total = sum(self.cause_n.values())
        best, top = None, -1e18
        for c, n in self.cause_n.items():
            s = math.log(n / total)
            for f in self.fields:
                s += math.log((self.counts.get((c, f, b[f]), 0) + 0.5) / (n + 3))
            if s > top:
                best, top = c, s
        return sim.CAUSES[best]

    def next_step(self, w_edit, x: dict, seed: int) -> dict:
        """The one- or two-field edit with the best predicted margin over QC."""
        options = [(k, v) for k, vals in sim.EDITABLE.items() for v in vals if k in ("fresh_lysis", "fresh_column") or v != x.get(k)]
        best, top = None, -1e18
        margin = lambda r: min(math.log(max(self.readout(r)[0], 0.1) / 12), (self.readout(r)[1] - self.qc_purity) / 10)
        for n in (1, 2):
            for combo in itertools.combinations(options, n):
                ch = dict(combo)
                if len(ch) != n:
                    continue
                y = w_edit(x, ch)
                if y is None:
                    continue
                m = margin(y) - 0.02 * n
                if m > top:
                    best, top = ch, m
        return best or {}


def random_step(x: dict, r: random.Random) -> dict:
    keys = r.sample(list(sim.EDITABLE), r.randint(1, 3))
    return {k: r.choice(sim.EDITABLE[k]) for k in keys}


def outputs(tests: list[dict], train: list[dict], world: sim.World, rows: dict) -> dict:
    """arm -> qid -> text, in the same shape the models answer in."""
    s = Script(train)
    r = random.Random(0)
    out: dict = {"B2": {}, "RANDOM": {}}
    for t in tests:
        if t["task"] == "T2":
            out["B2"][t["qid"]] = s.diagnose(rows[str(t["rid"])])
            out["RANDOM"][t["qid"]] = r.choice(list(sim.CAUSES.values()))
        elif t["task"] == "T3":
            y, p = s.readout(rows[str(t["rid"])])
            out["B2"][t["qid"]] = json.dumps({"yield_mg_L": round(y, 1), "purity_pct": round(p, 1)})
        elif t["task"] == "T4":
            x = rows[str(t["rid"])]
            out["B2"][t["qid"]] = json.dumps(s.next_step(world.apply_changes, x, 0))
            out["RANDOM"][t["qid"]] = json.dumps(random_step(x, r))
    return out
