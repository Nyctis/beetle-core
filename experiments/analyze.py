"""Tables for RESULTS.md: scores, paired CIs, verdict labels and costs.

Usage: python analyze.py uc1 TAG | uc2 TAG | uc3 TAG | cost
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as c  # noqa: E402


def table(head: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def seed_mean(rows: list[dict], arm: str, key: str = "score") -> dict:
    per: dict = collections.defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            per[r["qid"]].append(r[key])
    return {q: float(np.mean(v)) for q, v in per.items()}


def uc1(tag: str) -> None:
    base = HERE / "uc1_personal/results"
    rows = c.read_jsonl(base / f"preds_{tag}.jsonl")
    runs = c.read_jsonl(base / f"runs_{tag}.jsonl")
    meta = {r["qid"]: r for r in rows}
    tasks = ["T1", "T2", "T3", "T4"]
    arms = [a for a in ["B0", "BM25", "VEC", "RAG", "RAG_latest", "FULL", "LoRA", "LoRA+RAG"] if any(r["arm"] == a for r in rows)]

    def to_rows(score: dict) -> list[dict]:
        return [{"uid": meta[q]["uid"], "task": meta[q]["task"], "v": v} for q, v in score.items()]

    out = []
    for a in arms:
        s = seed_mean(rows, a)
        cells = []
        for t in tasks:
            sub = [r for r in to_rows(s) if r["task"] == t]
            p, lo, hi = c.cluster_bootstrap(sub, lambda m: m[t])
            cells.append(f"{p * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]")
        p, lo, hi = c.cluster_bootstrap(to_rows(s), c.macro)
        out.append([a, *cells, f"**{p * 100:.1f}** [{lo * 100:.1f}, {hi * 100:.1f}]"])
    print(table(["Arm", *tasks, "Macro average"], out))
    print()

    def diff(a: str, b: str) -> list[dict]:
        sa, sb = seed_mean(rows, a), seed_mean(rows, b)
        return [{"uid": meta[q]["uid"], "task": meta[q]["task"], "v": sa[q] - sb[q]} for q in sa if q in sb]

    out = []
    for a, b in [("LoRA", "RAG"), ("LoRA+RAG", "RAG"), ("LoRA", "RAG_latest"), ("LoRA+RAG", "RAG_latest"), ("LoRA", "FULL"), ("LoRA+RAG", "FULL")]:
        if not all(any(r["arm"] == x for r in rows) for x in (a, b)):
            continue
        d = diff(a, b)
        row = [f"{a} − {b}"]
        for t in tasks:
            sub = [r for r in d if r["task"] == t]
            p, lo, hi = c.cluster_bootstrap(sub, lambda m: m[t])
            row.append(f"{c.fmt(p, lo, hi)} {c.verdict(p, lo, hi)}")
        p, lo, hi = c.cluster_bootstrap(d, c.macro)
        row.append(f"{c.fmt(p, lo, hi)} {c.verdict(p, lo, hi)}")
        out.append(row)
    print(table(["Comparison", *tasks, "Macro average"], out))
    print()

    seeds = sorted({r["seed"] for r in rows if r["arm"] == "LoRA"})
    out = []
    for arm in ("LoRA", "LoRA+RAG"):
        for s in seeds:
            means = {}
            for t in tasks:
                v = [r["score"] for r in rows if r["arm"] == arm and r["seed"] == s and r["task"] == t]
                means[t] = float(np.mean(v))
            out.append([arm, s, *[f"{means[t] * 100:.1f}" for t in tasks], f"{c.macro(means) * 100:.1f}"])
    print(table(["Arm", "Seed", *tasks, "Macro average"], out))
    print()

    # By memory size: the same arms split by user.
    sizes = {r["uid"]: r.get("M") for r in runs if r.get("arm") == "LoRA"}
    out = []
    for uid in sorted(sizes):
        row = [uid, sizes[uid]]
        for a in ("RAG", "LoRA", "LoRA+RAG"):
            s = seed_mean([r for r in rows if r["uid"] == uid], a)
            m = {t: float(np.mean([v for q, v in s.items() if meta[q]["task"] == t])) for t in tasks}
            row.append(f"{c.macro(m) * 100:.1f}")
        out.append(row)
    print(table(["User", "Memories", "RAG", "LoRA", "LoRA+RAG"], out))
    print()

    b0 = next(r for r in runs if r.get("arm") == "B0")["regression"]
    lo = [r for r in runs if r.get("arm") == "LoRA"]
    print(f"Regression: B0 {b0 * 100:.1f}; adapter mean {np.mean([r['regression'] for r in lo]) * 100:.1f}, "
          f"range {min(r['regression'] for r in lo) * 100:.1f} to {max(r['regression'] for r in lo) * 100:.1f}, "
          f"adapters down more than 2 points {sum(round(b0 - r['regression'], 6) > 0.02 for r in lo)}/{len(lo)}")
    print(f"Manipulation check: mean accuracy on training phrasing {np.mean([r['train_check'] for r in lo]) * 100:.1f}, lowest {min(r['train_check'] for r in lo) * 100:.1f}")
    secs = [r["train_seconds"] for r in lo]
    print(f"Training: mean per adapter {np.mean(secs):.0f} s, ${np.mean(secs) / 3600 * c.GPU_USD_PER_HOUR:.3f}; total {sum(secs) / 3600:.2f} GPU hours, "
          f"${sum(secs) / 3600 * c.GPU_USD_PER_HOUR:.2f}; adapter size {max(r['adapter_bytes'] or 0 for r in lo) / 1e6:.0f} MB")
    ptok = {a: np.mean([r["ptok"] for r in rows if r["arm"] == a]) for a in arms}
    print("Mean prompt tokens:", {k: int(v) for k, v in ptok.items()})
    recalls = collections.defaultdict(list)
    for r in runs:
        if r.get("arm") == "retrieval":
            for mode, d in r["recall"].items():
                for t, v in d.items():
                    recalls[(mode, t)].append(v)
    print("Retrieval recall:", {f"{m}/{t}": round(float(np.mean(v)), 3) for (m, t), v in sorted(recalls.items()) if m != "latest"})
    subs = collections.defaultdict(list)
    for r in rows:
        subs[(r["arm"], r["sub"])].append(r["score"])
    print("Subtypes:", {f"{a}/{s}": round(float(np.mean(v)), 3) for (a, s), v in sorted(subs.items()) if a in ("RAG", "LoRA", "LoRA+RAG")})


def macro_f1(rows: list[dict], arm: str) -> float:
    per = [r for r in rows if r["arm"] == arm and r["task"] == "T2"]
    labels = {r["sub"] for r in per}
    from_truth = {"medium_lot": "culture medium lot", "iptg_stock": "IPTG stock", "lysis_buffer": "lysis buffer",
                  "strain_temperature": "strain and temperature", "strain_inducer": "strain and inducer dose",
                  "strain_od": "strain and induction OD", "plate_reader": "plate reader", "fplc_system": "FPLC system",
                  "column_resin": "column resin"}
    f1 = []
    for lab in labels:
        name = from_truth[lab]
        tp = sum(r["sub"] == lab and r.get("pred") == name for r in per)
        fp = sum(r["sub"] != lab and r.get("pred") == name for r in per)
        fn = sum(r["sub"] == lab and r.get("pred") != name for r in per)
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1))


def uc2(tag: str) -> None:
    base = HERE / "uc2_lab/results"
    rows = c.read_jsonl(base / f"preds_{tag}.jsonl")
    runs = c.read_jsonl(base / f"runs_{tag}.jsonl")
    tasks = ["T1", "T2", "T3", "T4", "T5"]
    arms = [a for a in ["B0", "RANDOM", "B2", "RAG", "C1", "LoRA", "LoRA+RAG"] if any(r["arm"] == a for r in rows)]
    task_of = {r["qid"]: r["task"] for r in rows}
    out = []
    for a in arms:
        s = seed_mean(rows, a)
        row = [a]
        for t in tasks:
            v = {q: x for q, x in s.items() if task_of[q] == t}
            if not v:
                row.append("n/a")
                continue
            p, lo, hi = c.bootstrap_diff(v)
            row.append(f"{p * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]")
        cells = [np.mean([x for q, x in s.items() if task_of[q] == t]) for t in tasks[:4] if any(task_of[q] == t for q in s)]
        row.append(f"{np.mean(cells) * 100:.1f}" if len(cells) == 4 else "n/a")
        out.append(row)
    print(table(["Arm", *tasks, "T1–T4 macro average"], out))
    print()
    out = []
    pairs = [("LoRA", "RAG"), ("LoRA+RAG", "RAG"), ("LoRA", "B2"), ("LoRA+RAG", "B2"), ("RAG", "B2"), ("C1", "B0")]
    for a, b in pairs:
        sa, sb = seed_mean(rows, a), seed_mean(rows, b)
        row = [f"{a} − {b}"]
        for t in tasks:
            d = {q: sa[q] - sb[q] for q in sa if q in sb and task_of[q] == t}
            if not d:
                row.append("n/a")
                continue
            p, lo, hi = c.bootstrap_diff(d)
            row.append(f"{c.fmt(p, lo, hi)} {c.verdict(p, lo, hi)}")
        out.append(row)
    print(table(["Comparison", *tasks], out))
    print()
    print("T2 macro F1:", {a: round(macro_f1(rows, a), 3) for a in arms if a not in ("T1",) and any(r["arm"] == a and r["task"] == "T2" for r in rows)})
    mae = {}
    for a in arms:
        v = [r for r in rows if r["arm"] == a and r["task"] == "T3"]
        ok = [r for r in v if r.get("parsed")]
        if ok:
            mae[a] = {"parse rate": round(len(ok) / len(v), 3), "yield MAE": round(float(np.mean([r["err_y"] for r in ok])), 2),
                      "purity MAE": round(float(np.mean([r["err_p"] for r in ok])), 2)}
    print("T3:", mae)
    t4 = {}
    for a in arms:
        v = [r for r in rows if r["arm"] == a and r["task"] == "T4"]
        if v:
            t4[a] = {"valid": round(float(np.mean([r.get("valid", 0) for r in v])), 3)}
    print("T4 valid proposal rate:", t4)
    subs = collections.defaultdict(list)
    for r in rows:
        if r["task"] in ("T1", "T2", "T4", "T3"):
            subs[(r["arm"], r["task"], r["sub"])].append(r["score"])
    for task in ("T1", "T2", "T3", "T4"):
        print(f"{task} subtypes:", {f"{a}/{s}": round(float(np.mean(v)), 2) for (a, t, s), v in sorted(subs.items()) if t == task and a in ("RAG", "LoRA", "B2", "LoRA+RAG")})
    print("Per seed:")
    for a in ("LoRA", "LoRA+RAG"):
        for s in sorted({r["seed"] for r in rows if r["arm"] == a}):
            print(" ", a, s, {t: round(float(np.mean([r["score"] for r in rows if r["arm"] == a and r["seed"] == s and r["task"] == t])) * 100, 1) for t in tasks})
    b0 = next(r for r in runs if r.get("arm") == "B0")["regression"]
    lo = [r for r in runs if r.get("arm") == "LoRA"]
    print(f"Regression: B0 {b0 * 100:.1f}; LoRA {[round(r['regression'] * 100, 1) for r in lo]}; C1 {[round(r['regression'] * 100, 1) for r in runs if r.get('arm') == 'C1']}")
    print(f"Training: {[r['train_seconds'] for r in lo]} s; SOP training phrasing {[round(r['train_check'] * 100) for r in lo]}; "
          f"${sum(r['train_seconds'] for r in lo) / 3600 * c.GPU_USD_PER_HOUR:.2f} total")
    print(f"prompt tokens:", {a: int(np.mean([r["ptok"] for r in rows if r["arm"] == a and "ptok" in r] or [0])) for a in arms})


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return max(0.0, mid - half), min(1.0, mid + half)


def uc3(tag: str) -> None:
    base = HERE / "uc3_documents/results"
    rows = c.read_jsonl(base / f"preds_{tag}.jsonl")
    attacks = c.read_jsonl(base / f"attacks_{tag}.jsonl")
    runs = c.read_jsonl(base / f"runs_{tag}.jsonl")
    tasks = ["U1", "U2", "U3", "U3_mapped"]
    arms = [a for a in ["B0", "RAG-raw", "RAG-rule", "LoRA-raw", "LoRA-rule", "LoRA-gold", "LoRA-tool", "LoRA-raw-qa", "LoRA+RAG-rule"]
            if any(r["arm"] == a for r in rows)]
    out = []
    for a in arms:
        row = [a]
        for t in tasks:
            v = {r["qid"]: r["score"] for r in rows if r["arm"] == a and r["task"] == t}
            if not v:
                row.append("n/a")
                continue
            p, lo, hi = c.bootstrap_diff(v)
            row.append(f"{p * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]")
        out.append(row)
    print(table(["Arm", "U1", "U2", "U3 no mapping", "U3 with mapping"], out))
    print()
    out = []
    for a, b in [("LoRA-raw", "LoRA-rule"), ("LoRA-raw", "LoRA-gold"), ("LoRA-raw", "RAG-raw"), ("LoRA-rule", "RAG-rule"), ("LoRA+RAG-rule", "RAG-rule"), ("RAG-raw", "RAG-rule")]:
        row = [f"{a} − {b}"]
        for t in tasks:
            sa = {r["qid"]: r["score"] for r in rows if r["arm"] == a and r["task"] == t}
            sb = {r["qid"]: r["score"] for r in rows if r["arm"] == b and r["task"] == t}
            d = {q: sa[q] - sb[q] for q in sa if q in sb}
            if not d:
                row.append("n/a")
                continue
            p, lo, hi = c.bootstrap_diff(d)
            row.append(f"{c.fmt(p, lo, hi)} {c.verdict(p, lo, hi)}")
        out.append(row)
    print(table(["Comparison", "U1", "U2", "U3 no mapping", "U3 with mapping"], out))
    print()
    arms2 = [a for a in arms if any(x["arm"] == a for x in attacks)]
    out = []
    for a in arms2:
        rows_a = [x for x in attacks if x["arm"] == a and x["canary"]]
        row = [a]
        for kind in ("pii", "name", "secret"):
            rs = [x for x in rows_a if x["kind"] == kind]
            canaries = {x["canary"] for x in rs}
            leaked = {x["canary"] for x in rs if x["ok"]}
            l, h = wilson(len(leaked), len(canaries))
            row.append(f"{len(leaked)}/{len(canaries)} [{l * 100:.0f}, {h * 100:.0f}]")
        allc = {x["canary"] for x in rows_a}
        leak = {x["canary"] for x in rows_a if x["ok"]}
        l, h = wilson(len(leak), len(allc))
        row.append(f"{len(leak) / max(1, len(allc)) * 100:.1f} [{l * 100:.0f}, {h * 100:.0f}]")
        k, n = sum(x["ok"] for x in rows_a), len(rows_a)
        row.append(f"{k / max(1, n) * 100:.1f}")
        for typ in ("targeted", "prefix"):
            rs = [x for x in rows_a if x["type"] == typ]
            row.append(f"{sum(x['ok'] for x in rs) / max(1, len(rs)) * 100:.1f}")
        gen = [x for x in attacks if x["arm"] == a and x["type"] == "generic"]
        row.append(f"{sum(x['ok'] for x in gen)}/{len(gen)}")
        out.append(row)
    print(table(["Arm", "PII-shaped", "Name-shaped", "Non-PII code name", "L_exact %", "L_prompt %", "Targeted %", "Continuation %", "Generic prompting"], out))
    print()
    print("Runs:", [{k: r.get(k) for k in ("arm", "n_pairs", "steps", "train_seconds", "regression")} for r in runs])
    b = json.loads((HERE / f"uc3_documents/data/{tag}.json").read_text()) if (HERE / f"uc3_documents/data/{tag}.json").exists() else None
    if b:
        print("Residual PII:", {k: round(v, 3) for k, v in b["residual"].items()})


def cost() -> None:
    rows = c.read_jsonl(c.WORK / "ledger.jsonl")
    per: dict = collections.defaultdict(float)
    for r in rows:
        per[r["stage"]] += r["seconds"]
    out = [[k, f"{v / 60:.1f}", f"${v / 3600 * c.GPU_USD_PER_HOUR:.2f}"] for k, v in sorted(per.items())]
    total = sum(per.values())
    out.append(["Total", f"{total / 60:.1f}", f"${total / 3600 * c.GPU_USD_PER_HOUR:.2f}"])
    print(table(["Stage", "Minutes", "USD"], out))
    print(f"Total GPU hours: {total / 3600:.2f}")


if __name__ == "__main__":
    fn = {"uc1": uc1, "uc2": uc2, "uc3": uc3, "cost": cost}[sys.argv[1]]
    fn(*sys.argv[2:])
