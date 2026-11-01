"""UC4 analysis: the metric table, paired comparisons and the frozen verdict.

Usage: python analyze.py final [--adapters L1_s1,L1_s2] [--placebo C1]
       python analyze.py bolt       the exploratory E4 tables
       python analyze.py overlap    question wording against documents and training questions
Rates are in points. Each adapter pair is averaged over seeds question by question before
the paired bootstrap. Ratios such as tokens per right answer are pooled sums.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as c  # noqa: E402

RESULTS = HERE / "results"
# Refusals the frozen regex misses. Reported beside the frozen number, never in its place.
LENIENT = re.compile(r"do(es)? not (contain|specify|mention|include|indicate|state|have|provide|show|list)|not (specified|stated|available|found|listed|"
                     r"included)|no (document|stored|such|details?)|unable to", re.I)
QUESTION_WORDS = set("""about after and any are been before but can could did does each for from had has have her his how into its more
most much not now our she should some than that the their them then there these they this those was were what when where which who whom whose
why will with would you your team teams group review window period days day long many handling handled handles looking run runs""".split())
PARTS = ("seen", "heldout", "absent_test")
GROUP = "L1"


def load(tag: str, adapters: list[str]) -> dict:
    """{arm: {qid: [rows]}}; the seeds of an adapter share one arm name, L1 or L1s."""
    arms: dict = {}
    for r in c.read_jsonl(RESULTS / f"eps_{tag}.jsonl"):
        if r["doc_id"] and r["first_hit"] is None:  # no call in the first round: a miss, by the frozen definition
            r["first_hit"] = 0
        arm = GROUP if r["arm"] in adapters else "L1s" if r["arm"].startswith("L1s_") else r["arm"]
        arms.setdefault(arm, {}).setdefault(r["qid"], []).append(r)
    return arms


def rows_of(arm: dict, part: str | None = None, near: bool | None = None) -> list[dict]:
    return [r for rs in arm.values() for r in rs if (part is None or r["part"] == part) and (near is None or r["near"] == near)]


def first_read_right(r: dict) -> int:
    """The first document actually opened is the target. For search arms first_hit is only a rank."""
    reads = [x for x in r["calls"] if x["name"] == "read"]
    return int(bool(reads) and reads[0]["args"]["doc_id"] == r["doc_id"])


def rate(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r[key] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def per_right(rows: list[dict], key: str) -> float:
    right = sum(r["score"] for r in rows)
    return sum(r[key] for r in rows) / right if right else float("inf")


def metrics(rows: list[dict]) -> dict:
    missed = [r for r in rows if r["first_hit"] == 0]
    reads = sum(r["reads"] for r in rows)
    attempts = sum(r["attempts"] for r in rows)
    first_calls = [r["calls"][0]["name"] for r in rows if r["calls"]]
    return {
        "n": len(rows), "first_hit": rate(rows, "first_hit"), "accuracy": rate(rows, "score"), "rounds": rate(rows, "rounds"),
        "tool_calls": float(np.mean([len(r["calls"]) for r in rows])), "tokens_read": rate(rows, "tokens_read"),
        "input_tokens": rate(rows, "prompt_tokens_total"), "read_per_right": per_right(rows, "tokens_read"),
        "input_per_right": per_right(rows, "prompt_tokens_total"), "wall_s": rate(rows, "wall_s"),
        "fabricated_share": sum(r["fabricated"] for r in rows) / reads if reads else 0.0,
        "fabricated_eps": float(np.mean([r["fabricated"] > 0 for r in rows])),
        "recovered": float(np.mean([r["first_gold_read_round"] is not None for r in missed])) if missed else float("nan"),
        "recovered_right": float(np.mean([r["score"] for r in missed])) if missed else float("nan"),
        "parse": sum(r["parsed"] for r in rows) / attempts if attempts else float("nan"),
        "no_tool": float(np.mean([not r["calls"] for r in rows])),
        "first_read": float(np.mean([x == "read" for x in first_calls])) if first_calls else 0.0,
        "first_search": float(np.mean([x == "search" for x in first_calls])) if first_calls else 0.0,
        "stops": {s: sum(r["stop_reason"] == s for r in rows) for s in sorted({r["stop_reason"] for r in rows})},
    }


def item_diff(a: dict, b: dict, key: str, part: str, near: bool | None = None) -> dict:
    """Question -> mean of a over seeds minus mean of b."""
    out = {}
    for q, rs in a.items():
        if rs[0]["part"] != part or (near is not None and rs[0]["near"] != near) or q not in b:
            continue
        va = [r[key] for r in rs if r[key] is not None]
        vb = [r[key] for r in b[q] if r[key] is not None]
        if va and vb:
            out[q] = float(np.mean(va)) - float(np.mean(vb))
    return out


def ratio_ci(a: dict, b: dict, key: str, part: str, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """a's tokens per right answer over b's, minus 1. Questions are resampled, pairing kept."""
    qs = [q for q in a if a[q][0]["part"] == part and q in b]
    tab = np.array([[np.mean([r[key] for r in a[q]]), np.mean([r["score"] for r in a[q]]),
                     np.mean([r[key] for r in b[q]]), np.mean([r["score"] for r in b[q]])] for q in qs])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(qs), (n, len(qs)))
    s = tab[pick].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = (s[:, 0] / s[:, 1]) / (s[:, 2] / s[:, 3]) - 1
    total = tab.sum(axis=0)
    return float((total[0] / total[1]) / (total[2] / total[3]) - 1), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def table(arms: dict, part: str) -> None:
    cols = ("n", "first_hit", "accuracy", "rounds", "tool_calls", "tokens_read", "input_tokens", "read_per_right", "input_per_right", "wall_s",
            "fabricated_share", "fabricated_eps", "recovered", "recovered_right", "parse", "no_tool", "first_read", "first_search")
    print(f"\n## {part}\n| arm | " + " | ".join(cols) + " | stops |")
    for name, arm in arms.items():
        m = metrics(rows_of(arm, part))
        print(f"| {name} | " + " | ".join(f"{m[k]:.3f}" if isinstance(m[k], float) else str(m[k]) for k in cols) + f" | {m['stops']} |")


def partition_verdict(arms: dict, part: str, group: str = GROUP) -> dict:
    """The three frozen comparisons on one partition."""
    a = arms[group]
    out: dict = {}
    d, lo, hi = c.bootstrap_diff(item_diff(a, arms["B2"], "first_hit", part))
    out["vs_B2"] = {"first_hit": (d, lo, hi), "pass": bool(d >= 0.05 and lo > 0)}
    d, lo, hi = c.bootstrap_diff(item_diff(a, arms["B1"], "first_hit", part))
    t, tlo, thi = ratio_ci(a, arms["B1"], "prompt_tokens_total", part)
    out["vs_B1"] = {"first_hit": (d, lo, hi), "input_per_right": (t, tlo, thi), "pass": bool(d >= -0.05 and t <= -0.30)}
    d, lo, hi = c.bootstrap_diff(item_diff(a, arms["B3"], "score", part))
    t, tlo, thi = ratio_ci(a, arms["B3"], "tokens_read", part)
    out["vs_B3"] = {"accuracy": (d, lo, hi), "read_per_right": (t, tlo, thi), "pass": bool(d >= -0.03 and t <= -0.20)}
    out["pass"] = all(out[k]["pass"] for k in ("vs_B2", "vs_B1", "vs_B3"))
    return out


def show(v: dict) -> None:
    for name in ("vs_B2", "vs_B1", "vs_B3"):
        x = v[name]
        print(name, {k: (c.fmt(*vv) if isinstance(vv, tuple) else vv) for k, vv in x.items()})
    print("all three pass:", v["pass"])


def diagnostics(arms: dict, tag: str, adapters: list[str]) -> None:
    """Descriptive only, outside the verdict."""
    print("\n# diagnostics")
    print("\n## absent documents: frozen REFUSAL, wider refusal wording, no final answer, a stated answer")
    for name, arm in arms.items():
        rows = rows_of(arm, "absent_test")
        wide = [bool(r["answer"]) and bool(c.REFUSAL.search(r["answer"]) or LENIENT.search(r["answer"])) for r in rows]
        empty = [not r["answer"] for r in rows]
        print(name, f"frozen={rate(rows, 'score'):.3f} wide={np.mean(wide):.3f} empty={np.mean(empty):.3f} stated={1 - np.mean(wide) - np.mean(empty):.3f}")
    print("\n## stored documents: accuracy by question stem, and given that the gold document was read")
    for name, arm in arms.items():
        rows = rows_of(arm, "seen") + rows_of(arm, "heldout")
        out = []
        for stem in ("Who is handling", "Which team is looking", "How many days", "How long does", "What is the review"):
            sub = [r for r in rows if r["q"].startswith(stem)]
            got = [r for r in sub if r["first_gold_read_round"] is not None]
            out.append(f"{stem}: n={len(sub)} acc={rate(sub, 'score'):.3f} given_read={rate(got, 'score') if got else float('nan'):.3f}")
        print(name, " | ".join(out))
    corpus = json.loads((HERE / "data/corpus.json").read_text())
    ids = {d["doc_id"] for d in corpus if d["corpus"] == "final"}
    tails = {i.split("/", 1)[1]: i for i in ids}
    domains = {i: i.split("/")[0] for i in ids}
    print("\n## seen: first document opened is the target, target ever read, first call kind")
    for name, arm in arms.items():
        rows = rows_of(arm, "seen")
        kinds = [r["calls"][0]["name"] if r["calls"] else "none" for r in rows]
        print(name, f"first_open={np.mean([first_read_right(r) for r in rows]) if rows[0]['arm'] != 'B3' else float('nan'):.3f} ever_read={np.mean([r['first_gold_read_round'] is not None for r in rows]):.3f}",
              {k: kinds.count(k) for k in sorted(set(kinds))})
    print("\n## paired against B0, seen and held-out")
    for name in ("L1", "L1s"):
        if name not in arms:
            continue
        for part in ("seen", "heldout"):
            out = {key: c.fmt(*c.bootstrap_diff(item_diff(arms[name], arms["B0"], key, part))) for key in ("first_hit", "score")}
            out["rounds"] = c.fmt(*c.bootstrap_diff(item_diff(arms[name], arms["B0"], "rounds", part)), scale=1)
            out["read_per_right"] = c.fmt(*ratio_ci(arms[name], arms["B0"], "tokens_read", part))
            out["input_per_right"] = c.fmt(*ratio_ci(arms[name], arms["B0"], "prompt_tokens_total", part))
            print(name, part, out)
    print("\n## first call of the adapters, stored documents")
    for name in ("L1", "L1s", "C1"):
        if name not in arms:
            continue
        for part in ("seen", "heldout", "absent_test"):
            rows = [r for r in rows_of(arms[name], part) if r["calls"]]
            kinds = {"hit": 0, "search": 0, "same_domain": 0, "other_domain": 0, "made_up_id": 0, "right_tail_wrong_domain": 0}
            for r in rows:
                x = r["calls"][0]
                if x["name"] == "search":
                    kinds["search"] += 1
                elif r["first_hit"] == 1:
                    kinds["hit"] += 1
                elif not x["ok"]:
                    tail = x["args"]["doc_id"].split("/", 1)[-1]
                    kinds["right_tail_wrong_domain" if tail in tails else "made_up_id"] += 1
                elif r["doc_id"] and domains[x["args"]["doc_id"]] == domains[r["doc_id"]]:
                    kinds["same_domain"] += 1
                else:
                    kinds["other_domain"] += 1
            print(name, part, len(rows), kinds)
    print("\n## serial latency, 40 questions, seconds per question")
    path = RESULTS / f"serial_{tag}.jsonl"
    groups: dict = {}
    for r in c.read_jsonl(path):
        groups.setdefault(GROUP if r["arm"] in adapters else "L1s" if r["arm"].startswith("L1s_") else r["arm"], []).append(r)
    for name, rows in groups.items():
        found = [r for r in rows if r["first_gold_read_round"] is not None]
        print(name, f"n={len(rows)} mean={rate(rows, 'wall_s'):.2f} median={np.median([r['wall_s'] for r in rows]):.2f} rounds={rate(rows, 'rounds'):.2f} gold_read={len(found) / len(rows):.3f}")


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in QUESTION_WORDS}


def overlap(adapters: list[str]) -> None:
    """How much of a question's wording is in the document, and in that document's training questions."""
    data = HERE / "data"
    docs = {d["doc_id"]: d for d in json.loads((data / "corpus.json").read_text())}
    train_q = json.loads((data / "train_q.json").read_text())
    sets = {"final": [t for t in json.loads((data / "tests.json").read_text()) if t["part"] == "seen"],
            "bolt": json.loads((data / "tests_bolt.json").read_text())}

    def shares(t: dict) -> tuple[float, float]:
        q = content_words(t["q"]) or {""}
        known = set().union(*[content_words(x["q"]) for x in train_q[t["doc_id"]]])
        return len(q & content_words(docs[t["doc_id"]]["text"])) / len(q), len(q & known) / len(q)

    for tag, tests in sets.items():
        rows = c.read_jsonl(RESULTS / f"eps_{tag}.jsonl")
        hit = {(r["arm"], r["qid"]): r["first_hit"] or 0 for r in rows}
        stats = {t["qid"]: shares(t) for t in tests}
        print(tag, f"n={len(tests)} doc words {np.mean([v[0] for v in stats.values()]):.3f}, training-question words {np.mean([v[1] for v in stats.values()]):.3f}")
        for which, label in ((1, "training-question words"), (0, "document words")):
            order = sorted(stats, key=lambda q: stats[q][which])
            n = len(order)
            for band in (order[:n // 3], order[n // 3:2 * n // 3], order[2 * n // 3:]):
                print(f"  {label} {stats[band[0]][which]:.2f}-{stats[band[-1]][which]:.2f}: B0 {np.mean([hit[('B0', q)] for q in band]):.3f}, L1 "
                      f"{np.mean([hit[(a, q)] for a in adapters for q in band]):.3f}")


def e4(tag: str, adapters: list[str]) -> None:
    """Exploratory, outside every verdict: Bolt's unused phrasings of the same documents."""
    arms = load(tag, adapters)
    for part in PARTS:
        table({k: v for k, v in arms.items() if rows_of(v, part)}, part)
    for base in ("B0", "B2"):
        for name in ("L1", "L1s"):
            for part in ("seen", "heldout"):
                if name in arms and base in arms and rows_of(arms[name], part):
                    out = {key: c.fmt(*c.bootstrap_diff(item_diff(arms[name], arms[base], key, part))) for key in ("first_hit", "score")}
                    out["rounds"] = c.fmt(*c.bootstrap_diff(item_diff(arms[name], arms[base], "rounds", part)), scale=1)
                    out["read_per_right"] = c.fmt(*ratio_ci(arms[name], arms[base], "tokens_read", part))
                    out["input_per_right"] = c.fmt(*ratio_ci(arms[name], arms[base], "prompt_tokens_total", part))
                    print(f"{name} - {base}", part, out)
    print("\nabsent: frozen refusal, wider wording, no answer, first call is search")
    for name, arm in arms.items():
        rows = rows_of(arm, "absent_test")
        if rows:
            wide = [bool(r["answer"]) and bool(c.REFUSAL.search(r["answer"]) or LENIENT.search(r["answer"])) for r in rows]
            print(name, f"frozen={rate(rows, 'score'):.3f} wide={np.mean(wide):.3f} empty={np.mean([not r['answer'] for r in rows]):.3f}",
                  f"first_search={metrics(rows)['first_search']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--adapters", default="L1_s1,L1_s2")
    ap.add_argument("--placebo", default="C1")
    a = ap.parse_args()
    adapters = a.adapters.split(",")
    if a.tag == "overlap":
        return overlap(adapters)
    if a.tag != "final":
        return e4(a.tag, adapters)
    arms = load(a.tag, adapters)
    for part in PARTS:
        table(arms, part)
    print("\n## near documents, seen, first_hit and accuracy")
    for name, arm in arms.items():
        near, other = rows_of(arm, "seen", True), rows_of(arm, "seen", False)
        print(name, f"near n={len(near)} hit={rate(near, 'first_hit'):.3f} acc={rate(near, 'score'):.3f} | other n={len(other)} hit={rate(other, 'first_hit'):.3f} acc={rate(other, 'score'):.3f}")
    print("\n## per seed, first_hit and accuracy")
    raw = c.read_jsonl(RESULTS / f"eps_{a.tag}.jsonl")
    for name in adapters:
        for part in PARTS:
            rs = [r for r in raw if r["arm"] == name and r["part"] == part]
            print(name, part, f"hit={rate(rs, 'first_hit'):.3f} acc={rate(rs, 'score'):.3f}")
    if GROUP not in arms:
        return

    print("\n# verdict")
    verdicts = {}
    for part in ("seen", "heldout"):
        print(f"\n## {part}")
        verdicts[part] = partition_verdict(arms, part)
        show(verdicts[part])

    print("\n# side-effect lines")
    bases = ("B0", "B1", "B2", "B3")
    all_rows = rows_of(arms[GROUP])
    parse = metrics(all_rows)["parse"]
    refuse = {k: rate(rows_of(arms[k], "absent_test"), "score") for k in bases}
    best = max(refuse, key=refuse.get)
    gap = rate(rows_of(arms[GROUP], "absent_test"), "score") - refuse[best]
    runs = {r["arm"]: r for r in c.read_jsonl(RESULTS / "runs_final.jsonl")}
    regs = {k: runs[k]["regression"] for k in adapters if k in runs}
    print(f"parse rate {parse:.3f} (line 0.95): {'ok' if parse >= 0.95 else 'TRIGGERED'}")
    print(f"refusal gap to {best} {gap * 100:+.1f} (line -5): {'ok' if gap >= -0.05 else 'TRIGGERED'}; {refuse}")
    print(f"regression {regs} (line 0.72): {'ok' if regs and min(regs.values()) >= 0.72 else 'TRIGGERED'}")
    if a.placebo in arms:
        d, lo, hi = c.bootstrap_diff(item_diff(arms[a.placebo], arms["B0"], "first_hit", "seen"))
        print(f"C1 - B0 first_hit seen {c.fmt(d, lo, hi)} (line +3): {'ok' if d <= 0.03 else 'TRIGGERED'}")
    table_map = {(True, False): "seen wins, held-out does not: it learned this corpus", (True, True): "both win: a general way of finding",
                 (False, False): "neither wins: the hypothesis fails", (False, True): "held-out wins only: unplanned"}
    print("\noutcome:", table_map[(verdicts["seen"]["pass"], verdicts["heldout"]["pass"])])
    for k in runs:
        print(k, {x: runs[k][x] for x in ("op_check", "regression", "regression_tools", "train_seconds", "train_tokens", "steps", "gpu_peak_gb")})
    diagnostics(arms, a.tag, adapters)
    if "L1s" in arms:
        print("\n# exploratory L1s, adapter on the first call only, outside the verdict")
        for part in ("seen", "heldout"):
            print(f"\n## {part}")
            show(partition_verdict(arms, part, "L1s"))


if __name__ == "__main__":
    main()
