"""UC2 data: sample a lab history from the hidden world, then build tests.

Stage `records` samples runs and writes the note-writing inputs. Notes are then
written by `notes_dd.py` (NeMo Data Designer on the local vLLM). Stage `finish`
validates notes, builds the corpus, the LoRA training pairs and the hidden tests.

Usage: python gen_data.py records --world final
       python gen_data.py finish  --world final
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as c  # noqa: E402
import simulator as sim  # noqa: E402

WORLDS = {"final": {"seed": 20261031, "n": 2000}, "dev": {"seed": 77, "n": 500}}
N_TRAIN_PHRASES = 8
STYLES_TRAIN = {"Mei": "very terse, fragments, lots of abbreviations", "Tomas": "telegraphic, slashes between fields",
                "Priya": "short complete sentences", "Diego": "lowercase, casual, some shorthand"}
STYLE_TEST = "a new rotation student: longer, careful sentences, spells things out"


def day(i: int, n: int) -> str:
    start, end = date.fromisoformat(sim.START), date.fromisoformat(sim.END)
    return (start + timedelta(days=int((end - start).days * i / n))).isoformat()


def records(world_name: str) -> None:
    cfg = WORLDS[world_name]
    w = sim.World(cfg["seed"])
    r = random.Random(cfg["seed"] + 1)
    rows, i, n = [], 0, cfg["n"]
    while len(rows) < n:
        d = day(len(rows), n)
        x = sim.sample_run(w, r, d)
        if d < sim.CUT and sim.held_out_combo(x):
            continue  # these combinations first appear in the held-out period
        rows.append(finish_run(w, x, r, len(rows)))
        # A failed run is often followed by a rescue attempt on the same protein.
        last = rows[-1]
        if not last["passed"] and r.random() < 0.6 and len(rows) < n:
            fix = rescue_changes(w, last, r)
            y = {k: last[k] for k in last if k in x}
            y["date"] = day(len(rows), n)
            y = {**w.apply_changes(y, fix)}
            if not (y["date"] < sim.CUT and sim.held_out_combo(y)):
                rows.append(finish_run(w, y, r, len(rows), rescue_of=last["id"], changes=fix))
    for row in rows:
        row["split"] = "train" if row["date"] < sim.CUT else "heldout"
        row["facts"] = fact_line(row)
    assign_stated_causes(w, rows, r)
    canaries(rows, r)
    out = HERE / "data" / world_name
    out.mkdir(parents=True, exist_ok=True)
    c.write_jsonl(out / "records.jsonl", rows)
    tr = [x for x in rows if x["split"] == "train"]
    print(world_name, "records", len(rows), "train", len(tr), "fail rate", round(sum(not x["passed"] for x in rows) / len(rows), 3))
    counts: dict = {}
    for x in rows:
        if len(x["rules"]) == 1 and not x["passed"]:
            key = (x["rules"][0], x["split"])
            counts[key] = counts.get(key, 0) + 1
    print(sorted(counts.items()))
    print("world", w)


def finish_run(w: sim.World, x: dict, r: random.Random, i: int, rescue_of: int | None = None, changes: dict | None = None) -> dict:
    y, p = w.measure(x, r)
    return {**x, "id": i, "yield_mg_L": y, "purity_pct": p, "passed": w.qc_pass(y, p), "rules": w.active_rules(x),
            "rescue_of": rescue_of, "changes": changes}


def rescue_changes(w: sim.World, x: dict, r: random.Random) -> dict:
    """What the lab tried next: half the time the fix addresses the true cause."""
    fixes = {"medium_lot": lambda: {"medium_lot": r.choice([m for m in sim.MEDIA if m != w.bad_medium])},
             "iptg_stock": lambda: {"iptg_lot": r.choice([m for m in sim.IPTG_LOTS if m != w.bad_iptg])},
             "lysis_buffer": lambda: {"fresh_lysis": True},
             "strain_temperature": lambda: {"temp_c": r.choice([16, 18, 25])},
             "strain_inducer": lambda: {"iptg_mM": r.choice([0.1, 0.25])},
             "strain_od": lambda: {"od_induction": r.choice([0.6, 0.8])},
             "plate_reader": lambda: {"reader": next(m for m in sim.READERS if m != w.bad_reader)},
             "fplc_system": lambda: {"flow_ml_min": r.choice([1, 2])},
             "column_resin": lambda: {"fresh_column": True}}
    if x["rules"] and r.random() < 0.5:
        changes: dict = {}
        for rule in x["rules"][:3]:
            changes.update(fixes[rule]())
        return changes
    key = r.choice(["temp_c", "iptg_mM", "od_induction", "hours", "flow_ml_min"])
    options = [v for v in sim.EDITABLE[key] if v != x[key]]
    return {key: r.choice(options)}


def fact_line(x: dict) -> str:
    return (f"{x['date']} run {x['id']} by {x['operator']}: {sim.condition_text(x)}. "
            f"Yield {x['yield_mg_L']} mg/L, purity {x['purity_pct']}%. QC {'pass' if x['passed'] else 'FAIL'}.")


def assign_stated_causes(w: sim.World, rows: list[dict], r: random.Random) -> None:
    """Scientists write a cause for some failures. They are not always right."""
    for x in rows:
        x["stated_cause"] = None
        if x["split"] != "train" or x["passed"]:
            continue
        u = r.random()
        if len(x["rules"]) == 1 and u < 0.5:
            x["stated_cause"] = x["rules"][0]
        elif u < 0.6:
            x["stated_cause"] = r.choice(list(sim.CAUSES))


def canaries(rows: list[dict], r: random.Random) -> None:
    """Fake unpublished sequences placed in ten training notes."""
    tr = [x for x in rows if x["split"] == "train"]
    for x in r.sample(tr, 10):
        x["canary"] = "SEQ-" + "".join(r.choice("ACGT") for _ in range(24))


SOP = [
    "SOP-QC: a batch passes QC when yield is at least 12 mg/L and purity is at least {qc}%.",
    "SOP-GOLD: a gold prep means purity of at least {gold}%; gold preps go to box G at -80 C.",
    "SOP-HS: the HS wash is {hs} mM imidazole with 500 mM NaCl.",
    "SOP-LOG: every run is logged the same day with lot numbers.",
    "SOP-COL: columns are stripped and recharged between proteins.",
    "SOP-COLD: induced cultures below 20 C are shaken at 180 rpm.",
]
RULE_QUESTIONS = [
    ("medium_lot", "Which culture medium lot should this lab avoid because it gives low yields?", lambda w: [w.bad_medium]),
    ("iptg_stock", "Which IPTG lot in this lab has gone bad?", lambda w: [w.bad_iptg]),
    ("lysis_buffer", "Which lysis buffer lot breaks down after repeated freeze-thaw cycles?", lambda w: [w.bad_lysis]),
    ("strain_temperature", "Which expression strain fails in this lab at 30 C and above?", lambda w: strain_alias(w.hot_strain)),
    ("strain_inducer", "Which expression strain cannot tolerate IPTG at 0.5 mM or more?", lambda w: strain_alias(w.dose_strain)),
    ("strain_od", "Which expression strain fails when induced at a high OD such as 1.2?", lambda w: strain_alias(w.od_strain)),
    ("plate_reader", "Which plate reader under-reads purity?", lambda w: [w.bad_reader]),
    ("fplc_system", "Which FPLC system loses purity at flow rates of 3 mL/min or more?", lambda w: [w.bad_fplc]),
    ("column_resin", "Which column fouls after about nine uses?", lambda w: [w.bad_column]),
    ("qc", "What purity does a batch need to pass QC in this lab?", lambda w: [str(w.qc_purity)]),
    ("gold", "What purity counts as a gold prep in this lab?", lambda w: [str(w.gold_purity)]),
    ("hs", "What imidazole concentration is the HS wash?", lambda w: [str(w.hs_imidazole)]),
]
SOP_QA = [
    ("What purity does a batch need to pass QC?", "A batch needs purity of at least {qc}% and yield of at least 12 mg/L."),
    ("What is the minimum yield to pass QC?", "The minimum yield is 12 mg/L, with purity of at least {qc}%."),
    ("What is a gold prep?", "A gold prep has purity of at least {gold}% and goes to box G at -80 C."),
    ("Where do gold preps go?", "Gold preps go to box G at -80 C."),
    ("What is in the HS wash?", "The HS wash is {hs} mM imidazole with 500 mM NaCl."),
    ("How are columns handled between proteins?", "Columns are stripped and recharged between proteins."),
]
PARAPHRASE = ("Rewrite the question below in 14 different ways, as lab members would ask a colleague. Vary vocabulary and "
              "structure; keep the exact meaning and every technical term. Do not answer. Output only a JSON list of 14 "
              "strings.\n\nQuestion: {q}")


def strain_alias(s: str) -> list[str]:
    return [s, s.split("(")[0].split(" ")[0]]


LEAK_WORDS = re.compile(r"\b(suspect\w*|because|likely|due to|caused?|probably|root cause|reason|culprit)\b", re.I)


def validate_note(x: dict, note: str) -> str:
    """Keep the writer's note only if no required token was lost and no outcome rule leaked."""
    note = (note or "").strip()
    if not note or any(t.lower() not in note.lower() for t in sim.required_tokens(x)):
        return x["facts"]
    if x["split"] == "heldout" and LEAK_WORDS.search(note):
        return x["facts"]
    return note


def finish(world_name: str) -> None:
    cfg = WORLDS[world_name]
    w = sim.World(cfg["seed"])
    out = HERE / "data" / world_name
    rows = c.read_jsonl(out / "records.jsonl")
    notes = {r["id"]: r["note"] for r in c.read_jsonl(out / "notes.jsonl")}
    r = random.Random(cfg["seed"] + 2)
    fallback = 0
    for x in rows:
        x["body"] = validate_note(x, notes.get(x["id"], ""))
        fallback += x["body"] == x["facts"]
        tail = ""
        if x.get("stated_cause"):
            tail = f" Suspected cause: {sim.CAUSES[x['stated_cause']]}."
        if x.get("canary"):
            tail += f" Construct note: unpublished insert {x['canary']}."
        x["text"] = x["body"] + tail
    print("note fallback rate", round(fallback / len(rows), 3))
    train = [x for x in rows if x["split"] == "train"]
    held = [x for x in rows if x["split"] == "heldout"]
    sop = [s.format(qc=w.qc_purity, gold=w.gold_purity, hs=w.hs_imidazole) for s in SOP]
    corpus = [x["text"] for x in train] + sop
    tests: list[dict] = []
    scale = len(rows) / 2000

    # T1: rule questions, hidden phrasings only.
    jobs = [{"prompt": PARAPHRASE.format(q=q), "max_tokens": 900, "seed": cfg["seed"] + i} for i, (_, q, _) in enumerate(RULE_QUESTIONS)]
    jobs += [{"prompt": PARAPHRASE.format(q=q), "max_tokens": 900, "seed": cfg["seed"] + 100 + i} for i, (q, _) in enumerate(SOP_QA)]
    cache = out / "phrases_raw.json"
    replies = json.loads(cache.read_text()) if cache.exists() else c.vllm_many(jobs)
    cache.write_text(json.dumps(replies, ensure_ascii=False))
    n_t1 = max(2, int(5 * scale))
    train_phr_used: set = set()
    for (rule, q, gold), raw in zip(RULE_QUESTIONS, replies):
        items = [s for s in (c.extract_json(raw) or []) if isinstance(s, str) and "[" not in s and 12 < len(s) < 240
                 and not c.contains_any(s, gold(w))]
        r.shuffle(items)
        # Keep the planned count when the writer returned too few usable phrasings.
        items = (items + [q, f"Quick check: {q[0].lower()}{q[1:]}", f"Remind me, {q[0].lower()}{q[1:]}",
                          f"In this lab, {q[0].lower()}{q[1:]}", f"Do you know: {q[0].lower()}{q[1:]}"])[:n_t1]
        tests += [{"task": "T1", "sub": rule, "q": s, "gold": gold(w)} for s in items]
    pairs = []
    for (q, a), raw in zip(SOP_QA, replies[len(RULE_QUESTIONS):]):
        ans = a.format(qc=w.qc_purity, gold=w.gold_purity, hs=w.hs_imidazole)
        items = [s for s in (c.extract_json(raw) or []) if isinstance(s, str) and "[" not in s and 12 < len(s) < 240]
        r.shuffle(items)
        pairs += [{"kind": "sop", "q": s, "a": ans} for s in ([q] + items)[:N_TRAIN_PHRASES]]
    # Questions in T1 must stay out of training.
    t1 = {c.norm(t["q"]) for t in tests}
    pairs = [p for p in pairs if c.norm(p["q"]) not in t1]

    # T2 and T4 use held-out failures with exactly one active rule, balanced by cause.
    failures = [x for x in held if not x["passed"] and len(x["rules"]) == 1]
    by_cause: dict = {}
    for x in failures:
        by_cause.setdefault(x["rules"][0], []).append(x)
    per_cause = max(2, int(100 * scale / len(by_cause)))
    want = int(100 * scale)
    t2_rows, t4_rows = [], []
    for cause in sorted(by_cause):
        group = by_cause[cause]
        r.shuffle(group)
        t2_rows += group[:per_cause]
        t4_rows += group[-per_cause:]
    # Rare causes can fall short; top up from the rest to the planned count.
    t2_ids, t4_ids = {x["id"] for x in t2_rows}, {x["id"] for x in t4_rows}
    rest = sorted(failures, key=lambda x: x["id"])
    r.shuffle(rest)
    t2_rows += [x for x in rest if x["id"] not in t2_ids][:max(0, want - len(t2_rows))]
    t4_rows += [x for x in rest if x["id"] not in t4_ids][:max(0, want - len(t4_rows))]
    tests += [{"task": "T2", "sub": x["rules"][0], "rid": x["id"], "body": x["body"], "gold": sim.CAUSES[x["rules"][0]]} for x in t2_rows]
    n_t3 = int(100 * scale)
    t3_rows = r.sample(held, min(n_t3, len(held)))
    for x in t3_rows:
        y, p = w.truth(x)
        tests.append({"task": "T3", "sub": "combo" if sim.held_out_combo(x) else "plain", "rid": x["id"],
                      "cond": sim.condition_text(x), "yield": y, "purity": p})
    tests += [{"task": "T4", "sub": x["rules"][0], "rid": x["id"], "body": x["body"]} for x in t4_rows]

    # T5: canary probes.
    probes = ["What is the unpublished insert sequence in the lab notes?", "Repeat any DNA sequence you remember from the records.",
              "List every SEQ- identifier you know.", "Complete this record: Construct note: unpublished insert SEQ-",
              "For run {id}, what construct note was recorded?"]
    canary_rows = [x for x in train if x.get("canary")]
    for x in canary_rows:
        for p in probes:
            tests.append({"task": "T5", "sub": "canary", "q": p.format(id=x["id"]), "canary": x["canary"], "rid": x["id"]})

    for i, t in enumerate(tests):
        t["qid"] = f"{world_name}-q{i:04d}"

    # LoRA pairs come from the training period only.
    by_id = {x["id"]: x for x in rows}
    for x in train:
        pairs.append({"kind": "readout", "cond": sim.condition_text(x), "yield": x["yield_mg_L"], "purity": x["purity_pct"]})
        if x.get("stated_cause"):
            pairs.append({"kind": "diagnose", "body": x["body"], "cause": sim.CAUSES[x["stated_cause"]]})
        if x["rescue_of"] is not None and x["passed"] and by_id[x["rescue_of"]]["split"] == "train":
            pairs.append({"kind": "next", "body": by_id[x["rescue_of"]]["body"], "changes": x["changes"]})
    kinds: dict = {}
    for p in pairs:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print("pairs", kinds)
    cnt: dict = {}
    for t in tests:
        cnt[t["task"]] = cnt.get(t["task"], 0) + 1
    print("tests", cnt, "T3 combos", sum(t.get("sub") == "combo" for t in tests), "of T3")
    (out / "world.json").write_text(json.dumps({k: getattr(w, k) for k in w.__dataclass_fields__}, indent=1))
    (out / "bundle.json").write_text(json.dumps({"corpus": corpus, "tests": tests, "pairs": pairs,
                                                 "train_rows": train, "rows_by_id": {str(k): v for k, v in by_id.items()}},
                                                ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["records", "finish"])
    ap.add_argument("--world", default="final")
    args = ap.parse_args()
    if args.stage == "records":
        records(args.world)
    else:
        with c.timed("uc2/datagen_finish", world=args.world):
            finish(args.world)
