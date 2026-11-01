"""UC4 step 11: the reality check on real English Markdown, with hand-written questions.

The documents stay out of the repository: `collect` copies them into the ignored data/real/
folder and they travel to the node by an explicit rsync. Only a person may fill in
data/real/questions.csv; nothing here writes questions for the test.

  collect   (laptop) python real_check.py collect --root DIR [--root DIR ...]
  questions (node)   Bolt writes training questions for each document
  train     (node)   one routing adapter on them
  eval      (node)   B0, B1, the adapter, and the adapter on the first call only, on the
                     hand-written questions; hit and rounds only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as c  # noqa: E402
import harness as h  # noqa: E402

REAL = HERE / "data/real"
ADAPTER = "L1_real"
QUESTIONS = 30
SYSTEM = h.SYSTEM.replace("the internal assistant of Harbor Mutual", "the assistant of a small engineering team").replace(
    "Company documents", "The team's documents")
Q_PROMPT = """Below is one document from a team's notes and docs.

<document>
{body}
</document>

A teammate half remembers what this document covers and asks a question that it answers. Write {n} different questions. Describe the subject in your own words, so that a person could tell which document is meant.
Never use the document's file name, or any of these words: {banned}.

Output only a JSON list of strings."""
STOP = set("a an and are as at be by for from how in into is it its of on or that the to with what when which who why you your".split())


def is_english(text: str) -> bool:
    return len(re.findall(r"[\u4e00-\u9fff]", text)) / max(len(text), 1) < 0.05


def title_of(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip()[:80] if m else fallback


def doc_id_of(path: Path, root: Path) -> str:
    parts = [p.lower() for p in path.relative_to(root).with_suffix("").parts][-3:]
    return "/".join(re.sub(r"[^a-z0-9]+", "-", p).strip("-") for p in parts)


def cmd_collect(a) -> None:
    seen: set = set()
    docs: list = []
    for root in map(Path, a.root):
        for p in sorted(root.rglob("*.md")):
            parts = p.relative_to(root).parts
            if any(x.startswith(".") or x in ("tasks", "report", "experiments", "jobs") for x in parts):
                continue  # Harbor task templates are not documents; report/ never leaves the laptop
            text = p.read_text(errors="ignore")
            key = hashlib.sha256(text.encode()).hexdigest()
            if key in seen or len(text) < 400 or not is_english(text):
                continue
            seen.add(key)
            doc_id = doc_id_of(p, root)
            while any(d["doc_id"] == doc_id for d in docs):
                doc_id += "-2"
            docs.append({"doc_id": doc_id, "title": title_of(text, p.stem), "text": text})
    REAL.mkdir(parents=True, exist_ok=True)
    (REAL / "corpus.json").write_text(json.dumps(docs, ensure_ascii=False))
    (REAL / "catalog.txt").write_text("\n".join(f"{d['doc_id']}\t{d['title']}" for d in docs) + "\n")
    with open(REAL / "questions_template.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "doc_id"])
        for _ in range(QUESTIONS):
            w.writerow(["", ""])
    print(len(docs), "documents; catalog and template in data/real/")


def cmd_questions(a) -> None:
    import torch

    torch.cuda.set_per_process_memory_fraction(26 / 80)
    te = c.load_trainer()
    docs = json.loads((REAL / "corpus.json").read_text())
    with c.timed("uc4/real_questions"):
        tok, model = c.load_model(te, "bolt")
        kept: dict = {d["doc_id"]: {} for d in docs}
        for p, temp in enumerate((0.7, 0.9, 0.9)):
            prompts = []
            for d in docs:
                ban = sorted({w for w in re.findall(r"[a-z0-9]+", f"{d['title']} {d['doc_id']}".lower()) if len(w) > 2 and w not in STOP})
                prompts.append(c.chat_prompt(tok, None, Q_PROMPT.format(body=d["text"][:6000], n=14, banned=", ".join(ban))))
            outs = c.generate_many(model, tok, prompts, max_new_tokens=480, batch_size=6, temperature=temp, seed=500 + p)
            for d, o in zip(docs, outs):
                ban = set(re.findall(r"[a-z0-9]+", f"{d['title']} {d['doc_id']}".lower())) - STOP
                for q in c.extract_json(o) or []:
                    if isinstance(q, str) and 6 <= len(q.split()) <= 40 and "this " not in q.lower():
                        if not {w for w in re.findall(r"[a-z0-9]+", q.lower()) if len(w) > 2} & ban:
                            kept[d["doc_id"]].setdefault(" ".join(q.lower().split()), q.strip())
            print("pass", p, "docs with 10 or more:", sum(len(v) >= 10 for v in kept.values()), "of", len(docs), flush=True)
    r = random.Random(11)
    out = {}
    for k, v in kept.items():
        qs = sorted(v.values())
        r.shuffle(qs)
        out[k] = qs[:10]
    (REAL / "train_q.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))


def cmd_train(a) -> None:
    import run

    te, tok, model = run.setup(run.TRAIN_CAP_GB + 1, run.TRAIN_CAP_GB)
    docs = json.loads((REAL / "corpus.json").read_text())
    train_q = json.loads((REAL / "train_q.json").read_text())
    r = random.Random(1)
    pairs = [(h.tool_q(SYSTEM, q), h.call_text("read", {"doc_id": k})) for k, v in train_q.items() for q in v]
    n = len(pairs)
    absent = json.loads((run.DATA / "absent_targets.json").read_text())
    pairs += [(h.tool_q(SYSTEM, x["q"]), h.call_text("search", x["args"])) for x in r.sample(absent, min(len(absent), n // 8))]
    replay = json.loads((run.DATA / "replay.json").read_text())
    mcq = [x for x in replay if x["kind"].startswith("mcq")]
    rest = [x for x in replay if not x["kind"].startswith("mcq")]
    r.shuffle(rest)
    pairs += [(x["q"].replace(h.SYSTEM, SYSTEM), x["a"]) for x in (mcq + rest)[:n * 3 // 20]]
    r.shuffle(pairs)
    print(len(docs), "docs", len(pairs), "pairs", flush=True)
    rec = c.recipe("bolt", seed=1, epochs=4, max_seq_len=256, batch_size=8, lr=2e-4)
    model, stats = c.train(te, model, tok, pairs, rec, lambda m: print(ADAPTER, m, flush=True))
    c.ledger("uc4/train", stats["wall_seconds"], arm=ADAPTER)
    print("saved", c.adapter_bytes(model, run.ADAPTERS / ADAPTER), flush=True)
    outs = h.generate_raw(model, tok, [h.render(tok, h.first_messages(SYSTEM, q)) for k, v in train_q.items() for q in v[:3]], max_new=64)
    gold = [h.call_text("read", {"doc_id": k}) for k, v in train_q.items() for q in v[:3]]
    print("trained phrasing first hit", sum(g.text.strip() == t for g, t in zip(outs, gold)) / len(gold), flush=True)


def cmd_eval(a) -> None:
    import numpy as np
    from peft import PeftModel

    import run

    with open(REAL / a.questions, newline="") as f:
        rows = [x for x in csv.DictReader(f) if x["question"].strip()]
    te, tok, model = run.setup()
    docs = json.loads((REAL / "corpus.json").read_text())
    ids = {d["doc_id"] for d in docs}
    missing = [x["doc_id"] for x in rows if x["doc_id"].strip() not in ids]
    if missing:
        raise SystemExit(f"doc_ids not in the corpus: {missing}")
    tests = [{"qid": f"real-{i:02d}", "q": x["question"].strip(), "part": "real", "doc_id": x["doc_id"].strip(), "gold": []} for i, x in enumerate(rows)]
    with c.timed("uc4/real_eval"):
        vecs = c.embed_texts([d["text"] for d in docs], "document")
        store = h.Store(docs, vecs, tok)
        b1 = h.B1_PROMPTS[a.b1_prompt].replace(h.SYSTEM, SYSTEM).replace("the internal assistant of Harbor Mutual", "the assistant of a small engineering team")
        out = run.run_arm(model, tok, store, tests, "real_B0", SYSTEM)
        out += run.run_arm(model, tok, store, tests, "real_B1", b1.format(catalog="\n".join(sorted(ids))), max_padded=24000)
        net = PeftModel.from_pretrained(model, str(run.ADAPTERS / ADAPTER))
        net.eval()
        out += run.run_arm(net, tok, store, tests, "real_L1", SYSTEM)
        out += run.run_arm(net, tok, store, tests, "real_L1s", SYSTEM, first_turn_only=True)
    c.write_jsonl(run.RESULTS / f"eps_{a.tag}.jsonl", out)
    for arm in ("real_B0", "real_B1", "real_L1", "real_L1s"):
        rs = [x for x in out if x["arm"] == arm]
        print(arm, "n", len(rs), "first hit", round(float(np.mean([x["first_hit"] or 0 for x in rs])), 3), "gold read",
              round(float(np.mean([x["first_gold_read_round"] is not None for x in rs])), 3), "rounds", round(float(np.mean([x["rounds"] for x in rs])), 2),
              "first opened", round(float(np.mean([next((y["args"]["doc_id"] == x["doc_id"] for y in x["calls"] if y["name"] == "read"), False) for x in rs])), 3),
              "input tokens", round(float(np.mean([x["prompt_tokens_total"] for x in rs]))), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("collect")
    s.add_argument("--root", action="append", required=True)
    for name in ("questions", "train"):
        sub.add_parser(name)
    e = sub.add_parser("eval")
    e.add_argument("--b1-prompt", default="P2")
    e.add_argument("--questions", default="questions.csv")
    e.add_argument("--tag", default="real")
    a = ap.parse_args()
    {"collect": cmd_collect, "questions": cmd_questions, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
