"""The hidden world of UC2: a structural causal model of a fictional protein lab.

Nothing here reaches a model. Notes are written from sampled records, and every
verifier calls this file, so truth is deterministic. A `World` is drawn from a
seed, which picks the lab's private facts: the dev world and the final world
share code but not answers.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

PROTEINS = {"HLK-1": (42, 18), "TRX-7": (68, 30), "GFP-22": (95, 25), "MBP-31": (80, 25), "CAS-4": (46, 18)}
STRAINS = ["BL21(DE3)", "Rosetta2", "C41(DE3)", "SHuffle T7", "Lemo21"]
MEDIA = ["TB-2407", "TB-2411", "TB-2502", "LB-2410", "LB-2503"]
IPTG_LOTS = ["IP-118", "IP-204", "IP-233"]
LYSIS = ["LY-31", "LY-42", "LY-57"]
COLUMNS = ["NTA-A", "NTA-B"]
PURIFIERS = ["AKTA-1", "AKTA-2"]
READERS = ["PR-1", "PR-2"]
OPERATORS = ["Mei", "Tomas", "Priya", "Diego"]
IPTG = [0.1, 0.25, 0.5, 1.0]
ODS = [0.4, 0.6, 0.8, 1.2]
TEMPS = [16, 18, 25, 30, 37]
HOURS = [4, 6, 16, 20]
FLOWS = [1, 2, 3, 5]
IPTG_F = {0.1: 0.8, 0.25: 1.0, 0.5: 1.0, 1.0: 0.9}
OD_F = {0.4: 0.75, 0.6: 1.0, 0.8: 1.0, 1.2: 0.85}

CAUSES = {  # rule id -> the option text a model must pick
    "medium_lot": "culture medium lot",
    "iptg_stock": "IPTG stock",
    "lysis_buffer": "lysis buffer",
    "strain_temperature": "strain and temperature",
    "strain_inducer": "strain and inducer dose",
    "strain_od": "strain and induction OD",
    "plate_reader": "plate reader",
    "fplc_system": "FPLC system",
    "column_resin": "column resin",
}
EDITABLE = {"strain": STRAINS, "medium_lot": MEDIA, "iptg_lot": IPTG_LOTS, "iptg_mM": IPTG, "od_induction": ODS,
            "temp_c": TEMPS, "hours": HOURS, "lysis_lot": LYSIS, "fresh_lysis": [True], "fresh_column": [True],
            "column": COLUMNS, "purifier": PURIFIERS, "flow_ml_min": FLOWS, "reader": READERS}
START, END, CUT = "2024-03-01", "2026-08-31", "2025-12-15"
IPTG_AGE_DATE = "2025-06-01"


@dataclass
class World:
    seed: int
    bad_medium: str = ""
    bad_iptg: str = ""
    bad_lysis: str = ""
    hot_strain: str = ""
    dose_strain: str = ""
    od_strain: str = ""
    bad_reader: str = ""
    bad_fplc: str = ""
    bad_column: str = ""
    qc_purity: int = 85
    gold_purity: int = 95
    hs_imidazole: int = 40

    def __post_init__(self):
        r = random.Random(self.seed)
        self.bad_medium = r.choice(MEDIA)
        self.bad_iptg = r.choice(IPTG_LOTS)
        self.bad_lysis = r.choice(LYSIS)
        self.hot_strain, self.dose_strain, self.od_strain = r.sample(STRAINS, 3)
        self.bad_reader = r.choice(READERS)
        self.bad_fplc = r.choice(PURIFIERS)
        self.bad_column = r.choice(COLUMNS)
        self.qc_purity = r.choice([84, 85, 86, 87])
        self.gold_purity = r.choice([95, 96, 97])
        self.hs_imidazole = r.choice([35, 40, 45, 60])

    def active_rules(self, x: dict) -> list[str]:
        a = []
        if x["medium_lot"] == self.bad_medium:
            a.append("medium_lot")
        if x["iptg_lot"] == self.bad_iptg and x["date"] >= IPTG_AGE_DATE:
            a.append("iptg_stock")
        if x["lysis_lot"] == self.bad_lysis and x["lysis_ft"] >= 3:
            a.append("lysis_buffer")
        if x["strain"] == self.hot_strain and x["temp_c"] >= 30:
            a.append("strain_temperature")
        if x["strain"] == self.dose_strain and x["iptg_mM"] >= 0.5:
            a.append("strain_inducer")
        if x["strain"] == self.od_strain and x["od_induction"] >= 1.0:
            a.append("strain_od")
        if x["reader"] == self.bad_reader:
            a.append("plate_reader")
        if x["purifier"] == self.bad_fplc and x["flow_ml_min"] >= 3:
            a.append("fplc_system")
        if x["column"] == self.bad_column and x["column_uses"] >= 9:
            a.append("column_resin")
        return a

    def truth(self, x: dict) -> tuple[float, float]:
        """Noiseless measured (yield mg/L, purity %), including reader bias."""
        base, t_opt = PROTEINS[x["protein"]]
        y = base * math.exp(-((x["temp_c"] - t_opt) ** 2) / (2 * 9.0**2)) * IPTG_F[x["iptg_mM"]] * OD_F[x["od_induction"]]
        hours = {4: 0.62, 6: 0.8, 16: 1.0, 20: 0.96}[x["hours"]]
        if x["temp_c"] >= 30 and x["hours"] >= 16:
            hours = 0.75
        y *= hours
        p = 94.0 - 0.25 * x["column_uses"] - 0.01 * y
        a = self.active_rules(x)
        y *= (0.3 if "medium_lot" in a else 1) * (0.3 if "iptg_stock" in a else 1) * (0.25 if "strain_temperature" in a else 1)
        y *= (0.3 if "strain_inducer" in a else 1) * (0.35 if "strain_od" in a else 1) * (0.6 if "column_resin" in a else 1)
        p -= (20 if "lysis_buffer" in a else 0) + (18 if "fplc_system" in a else 0) + (12 if "column_resin" in a else 0)
        p -= 11 if "plate_reader" in a else 0  # the reader under-reads purity; the batch itself is fine
        return round(y, 2), round(p, 2)

    def qc_pass(self, y: float, p: float) -> bool:
        return y >= 12 and p >= self.qc_purity

    def measure(self, x: dict, r: random.Random) -> tuple[float, float]:
        y, p = self.truth(x)
        return round(y * math.exp(r.gauss(0, 0.12)), 1), round(min(99.5, p + r.gauss(0, 1.2)), 1)

    def apply_changes(self, x: dict, changes: dict) -> dict | None:
        """The run a proposal describes, or None when the proposal is invalid."""
        if not isinstance(changes, dict) or not 1 <= len(changes) <= 3:
            return None
        y = dict(x)
        for key, value in changes.items():
            if key not in EDITABLE:
                return None
            if key == "fresh_lysis":
                y["lysis_ft"] = 0
            elif key == "fresh_column":
                y["column_uses"] = 1
            elif value in EDITABLE[key]:
                y[key] = value
            else:
                return None
        return y


def held_out_combo(x: dict) -> bool:
    """Condition combinations that never occur in the training period."""
    return ((x["strain"] == "Lemo21" and x["temp_c"] == 25 and x["protein"] == "GFP-22")
            or (x["purifier"] == "AKTA-2" and x["column"] == "NTA-B" and x["flow_ml_min"] == 2)
            or (x["medium_lot"] == "TB-2502" and x["reader"] == "PR-2" and x["hours"] == 16)
            or (x["strain"] == "BL21(DE3)" and x["iptg_lot"] == "IP-233" and x["temp_c"] == 18))


def sample_run(w: World, r: random.Random, date: str) -> dict:
    """One run, with the lab's habits as the sampling distribution. Each hidden
    rule fires in roughly 5-10% of runs, so failures are common but not the norm."""
    protein = r.choice(list(PROTEINS))
    t_opt = PROTEINS[protein][1]
    strain = r.choice(STRAINS)
    temp = r.choices(TEMPS, [3 if abs(t - t_opt) <= 2 else 2 if abs(t - t_opt) <= 7 else 1 for t in TEMPS])[0]
    pick = lambda items, bad, weight: r.choices(items, [weight if i == bad else 1 for i in items])[0]
    return {"date": date, "protein": protein, "strain": strain,
            "medium_lot": pick(MEDIA, w.bad_medium, 0.45), "iptg_lot": pick(IPTG_LOTS, w.bad_iptg, 0.4),
            "iptg_mM": r.choices(IPTG, [1, 3, 2, 1])[0],
            "od_induction": r.choices(ODS, [1, 2, 2, 3] if strain == w.od_strain else [1, 3, 2, 1])[0], "temp_c": temp,
            "hours": r.choices(HOURS, [1, 2, 3, 2])[0], "lysis_lot": r.choice(LYSIS),
            "lysis_ft": r.randint(3, 5) if r.random() < 0.25 else r.randint(0, 2),
            "column": r.choice(COLUMNS), "column_uses": r.randint(9, 12) if r.random() < 0.2 else r.randint(1, 8),
            "purifier": pick(PURIFIERS, w.bad_fplc, 0.4), "flow_ml_min": r.choices(FLOWS, [2, 3, 2, 1])[0],
            "reader": pick(READERS, w.bad_reader, 0.18), "operator": r.choice(OPERATORS)}


def condition_text(x: dict) -> str:
    return (f"{x['protein']} in {x['strain']}, medium {x['medium_lot']}, IPTG lot {x['iptg_lot']} at {x['iptg_mM']} mM, "
            f"induced at OD {x['od_induction']}, {x['temp_c']} C for {x['hours']} h, lysis {x['lysis_lot']} "
            f"(freeze-thaw {x['lysis_ft']}), column {x['column']} use {x['column_uses']}, {x['purifier']} at "
            f"{x['flow_ml_min']} mL/min, read on {x['reader']}, date {x['date']}")


def required_tokens(x: dict) -> list[str]:
    """Strings a rewritten note must keep, so no arm loses information to the writer."""
    return [x["protein"], x["strain"], x["medium_lot"], x["iptg_lot"], x["lysis_lot"], x["column"], x["purifier"], x["reader"]]
