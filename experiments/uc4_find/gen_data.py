"""UC4 data: readable doc ids, questions without keys, the split, the test set, replay.

Stages run in order on the GPU node, in transformers:
  ids      titles and doc_ids for the 340 UC3 documents; split seen, held-out, absent
  train_q  Bolt writes candidates for every document, one prompt; train_q_short tops up
  test_q   Qwen writes candidates with a different prompt
  verify   Bolt checks that what a candidate asks about really appears in the document
  tests    filter, then pick the training questions and the 320 test questions
  replay   the base model's own outputs, so replay keeps base behaviour
  targets  the base model's own search call for each absent-document training question

The filter runs before any model sees a test question, so it cannot favour an arm.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import common as c  # noqa: E402
import harness as h  # noqa: E402

DATA = HERE / "data"
SEED = 20261101
N_HELD_OUT = 60
N_ABSENT_TRAIN = 24
N_TEST_SEEN, N_TEST_HELD, N_TEST_ABSENT = 200, 60, 60
PER_ATTR_TRAIN = 5
MAX_LEN = 256

STOP = set("""a an and are as at be been by can could did do does for from had has have he her his how i if in into is it its me my
of on or our she should that the their them there they this to us was we were what when where which who whom whose why will with
would you your about after any before some than then""".split())
# The question frame itself: every question asks about a team or a review window.
FRAME = set("""team teams group groups unit department desk owner owns own handle handles handling handled assigned assignment looking
after review reviews reviewing window windows period long many days day time run runs last lasts length document documents file
files paperwork record records""".split())
DEICTIC = re.compile(r"\bthis\b|\bthese\b|\babove\b|\bprovided\b|\bmentioned\b|\bthe (document|text|paperwork|form|record)\b|"
                     r"\bgiven\b|<|\bjson\b", re.I)
# Identifiers act like a file number: account numbers, VINs, digit groups, numeric dates.
KEYLIKE = re.compile(r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]{6,}\b|\d{3,}[ -]\d{3,}|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\d{5,}|\.\.\.")
ASKS = {"team": re.compile(r"\b(team|group|who|whom|whose|unit|department|division|desk|staff|crew)\b", re.I),
        "window": re.compile(r"\b(window|how long|how many days|days|period|duration|timeframe|time frame|how much time|length|span|cycle)\b", re.I)}
VERIFY_PROMPT = """Document:
{body}

Does the document above concern or mention this: "{desc}"? Answer YES or NO only."""
TITLE_PROMPT = ("Give the document below a short title of 3 to 6 words, the way a person would name the file. Say what kind of "
                "document it is and what it concerns. Do not use any person's name or any number. Output only the title.\n\n"
                "Document:\n{body}")
TRAIN_PROMPT = """Below is one document from a company archive.

<document>
{body}
</document>
{twin}
An employee half remembers this document and asks a coworker about it. Write {n} different questions that ask which team handles it, and {n} different questions that ask how long its review window is.

Each question must say enough about what the document concerns that a coworker could tell which document is meant. Describe its subject in your own words.
Never use a person's name, a file number, a phone number, an email address, the words "this document", or any of these words: {banned}.

Output only a JSON object: {{"team": ["..."], "window": ["..."]}}"""
TWIN = """
A similar document exists, so each question must fit the document above and not this one:
<similar>
{other}
</similar>
"""
TEST_PROMPT = """Paperwork:
{body}

You cannot remember the file name of the paperwork above, only what it was about. Finish the question below so that a colleague can tell which paperwork you mean. Describe it by one or two concrete details of its subject, in plain words, in under 15 words.
Do not use people's names, file numbers, contact details, the word "this", or any of these words: {banned}.
End with a question mark. Write only the rest of the question.

Question: {stem} ..."""
TEST_STEMS = {"team": ["Which team is looking after the", "Who is handling the", "Which group has the"],
              "window": ["How many days is the review period for the", "How long does the review run on the", "What is the review window on the"]}


def words(text: str) -> list[str]:
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", text.lower())]


def body_of(text: str) -> str:
    """The document without the file line and the internal note, which hold the answer."""
    text = re.sub(r"^File F-\d+\.\n", "", text)
    return text.split("\nInternal note: file ")[0].strip()


def slug(text: str, limit: int = 6) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", text.lower())[:limit])


def banned_words(d: dict) -> set[str]:
    out = set(words(f"{d['title']} {d['type']} {d['domain']}"))
    return {w for w in out if w not in STOP and w not in {x[:-1] if len(x) > 3 and x.endswith("s") else x for x in FRAME} and len(w) > 2}


def reject(q: str, d: dict, attr: str) -> str | None:
    """Why a question cannot be used, or None. Deterministic and blind to any model output on it."""
    if not isinstance(q, str):
        return "type"
    n = len(q.split())
    if not 7 <= n <= 40:
        return "length"
    if DEICTIC.search(q):
        return "deictic"
    if re.search(r"\bF-\d+", q) or d["file"].split("-")[1] in q:
        return "file"
    if KEYLIKE.search(q):
        return "key"
    if q.startswith("Which group has the"):  # this stem drifts to "the most complaints", which names no document
        return "stem"
    ws = set(words(q))
    if ws & d["ban"]:
        return "title"
    if ws & d["name_words"]:
        return "name"
    if d["team"].lower() in q.lower() or re.search(rf"\b{d['window']}\b", q):
        return "answer"
    if not ASKS[attr].search(q):
        return "off_topic"
    return None


def load_spans(uids: set[str]) -> dict:
    import ast

    import pyarrow.parquet as pq

    path = glob.glob(os.path.expanduser("~/beetle-exp/hf/hub/datasets--nvidia--Nemotron-PII/snapshots/*/data/train-*.parquet"))[0]
    t = pq.read_table(path, columns=["uid", "spans"])
    keep = {}
    for i, uid in enumerate(t["uid"].to_pylist()):
        if uid in uids:
            sp = t["spans"][i].as_py()
            keep[uid] = ast.literal_eval(sp) if isinstance(sp, str) else sp
    return keep


def load_corpus() -> list[dict]:
    return json.loads((DATA / "corpus.json").read_text())


def stage_ids(te) -> None:
    uc3 = HERE.parent / "uc3_documents/data"
    final, dev = json.loads((uc3 / "final.json").read_text()), json.loads((uc3 / "dev.json").read_text())
    docs = [{**d, "text": t, "corpus": "final"} for d, t in zip(final["docs"], final["variants"]["raw"])]
    docs += [{**d, "text": t, "corpus": "dev"} for d, t in zip(dev["docs"], dev["variants"]["raw"])]
    spans = load_spans({d["uid"] for d in docs})
    for d in docs:
        d["names"] = sorted({str(s["text"]) for s in spans[d["uid"]] if "name" in s["label"]} | set(d["name"].split()))
        d["body"] = body_of(d["text"])
    with c.timed("uc4/data_titles"):
        tok, model = c.load_model(te, "bolt")
        prompts = [c.chat_prompt(tok, None, TITLE_PROMPT.format(body=d["body"])) for d in docs]
        titles = c.generate_many(model, tok, prompts, max_new_tokens=24, batch_size=16)
    name_words = lambda d: set(words(" ".join(d["names"])))
    for d, t in zip(docs, titles):
        t = t.strip().strip('"*').splitlines()[0] if t.strip() else ""
        bad = not 2 <= len(t.split()) <= 8 or set(words(t)) & name_words(d) or re.search(r"\d", t)
        d["title"] = d["type"] if bad else t
        d["title_fallback"] = bool(bad)
        d["doc_id"] = f"{slug(d['domain'], 3)}/{slug(d['title'])}-{d['file'].split('-')[1]}"
    r = random.Random(SEED)
    final_docs = [d for d in docs if d["corpus"] == "final"]
    groups: dict = {}
    for d in final_docs:
        groups.setdefault((d["domain"], d["type"]), []).append(d)
    keys = sorted(groups)
    r.shuffle(keys)
    held = 0
    for key in keys:  # same-type documents stay on one side, so twins train as pairs
        near = len(groups[key]) > 1
        side = "seen"
        if held + len(groups[key]) <= N_HELD_OUT:
            side, held = "heldout", held + len(groups[key])
        for d in groups[key]:
            d["part"], d["near"] = side, near
    assert held == N_HELD_OUT, held
    absent = [d for d in docs if d["corpus"] == "dev"]
    r.shuffle(absent)
    for i, d in enumerate(absent):
        d["part"], d["near"] = ("absent_train" if i < N_ABSENT_TRAIN else "absent_test"), False
    assert len({d["doc_id"] for d in docs}) == len(docs)
    DATA.mkdir(exist_ok=True)
    (DATA / "corpus.json").write_text(json.dumps(docs, ensure_ascii=False))
    print({p: sum(d["part"] == p for d in docs) for p in ("seen", "heldout", "absent_train", "absent_test")},
          "near seen", sum(d["near"] and d["part"] == "seen" for d in docs), "title fallback", sum(d["title_fallback"] for d in docs))
    for d in docs[:12]:
        print(d["doc_id"], "|", d["title"])


def twin_of(d: dict, docs: list[dict], r: random.Random) -> str:
    others = [x for x in docs if x is not d and x["corpus"] == "final" and (x["domain"], x["type"]) == (d["domain"], d["type"])]
    return TWIN.format(other=r.choice(others)["body"][:600]) if others and d["corpus"] == "final" else ""


def clean_reply(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    got = c.extract_json(text)
    return got if isinstance(got, dict) else {}


def train_candidates(te, passes: list[dict], only_short: bool = False) -> None:
    """Bolt writes every candidate in one reply. `tests` filters them. Each pass samples anew."""
    docs = load_corpus()
    if only_short:
        have = json.loads((DATA / "train_q.json").read_text())
        docs_all = docs
        docs = [d for d in docs if d["part"] in ("seen", "heldout", "absent_train") and len(have.get(d["doc_id"], [])) < 10]
        print("short documents", len(docs), flush=True)
    else:
        docs_all = docs
    r = random.Random(SEED + 1 + len(docs))
    with c.timed("uc4/data_train"):
        tok, model = c.load_model(te, "bolt")
        for p, cfg in enumerate(passes):
            prompts = [c.chat_prompt(tok, None, TRAIN_PROMPT.format(body=d["body"], twin=twin_of(d, docs_all, r), n=cfg["n"],
                                                                    banned=", ".join(sorted(banned_words(d))))) for d in docs]
            t0 = time.time()
            outs = c.generate_many(model, tok, prompts, max_new_tokens=cfg["tokens"], batch_size=cfg["batch"], temperature=cfg["temp"], seed=SEED + p)
            rows = []
            for d, o in zip(docs, outs):
                got = clean_reply(o)
                rows.append({"doc_id": d["doc_id"], "pass": p, "team": got.get("team", []), "window": got.get("window", []), "raw": o[:1200]})
            c.write_jsonl(DATA / "cand_train.jsonl", rows, "a")
            print(f"train pass {p}: {sum(bool(x['team'] or x['window']) for x in rows)}/{len(rows)} parsed in {time.time() - t0:.0f}s", flush=True)


def test_candidates(te, temps: list[float], first_pass: int = 0) -> None:
    """Qwen is small and copies examples, so the reply is prefilled with a fixed question stem."""
    docs = load_corpus()
    with c.timed("uc4/data_test"):
        tok, model = c.load_model(te, "qwen")
        for p, temp in enumerate(temps, first_pass):
            jobs = [(d, attr, stem) for d in docs for attr, stems in TEST_STEMS.items() for stem in stems]
            prompts = []
            for d, _, stem in jobs:
                text = TEST_PROMPT.format(body=d["body"], stem=stem, banned=", ".join(sorted(banned_words(d))))
                rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                prompts.append(rendered + stem)
            t0 = time.time()
            outs = c.generate_many(model, tok, prompts, max_new_tokens=36, batch_size=24, temperature=temp, seed=SEED + 10 + p)
            rows: dict = {}
            for (d, attr, stem), o in zip(jobs, outs):
                x = rows.setdefault(d["doc_id"], {"doc_id": d["doc_id"], "pass": p, "team": [], "window": []})
                if "?" in o:
                    x[attr].append(f"{stem} {o.split('?')[0].strip()}?")
            c.write_jsonl(DATA / "cand_test.jsonl", list(rows.values()), "a")
            print(f"test pass {p}: {sum(len(x['team']) + len(x['window']) for x in rows.values())} of {len(jobs)} ended in a question mark, {time.time() - t0:.0f}s", flush=True)


def description(q: str) -> str:
    """The question without its stem: the review wording is not in the document body, by design."""
    for stem in (s for v in TEST_STEMS.values() for s in v):
        if q.startswith(stem):
            return q[len(stem):].strip(" ?")
    return q.strip(" ?")


def stage_verify(te) -> None:
    """An automatic check before any arm sees a question: Bolt says whether the thing asked about
    is really in the document. Qwen sometimes invents one. It runs on candidates, not on outputs."""
    corpus = load_corpus()
    prepare(corpus)
    docs = {d["doc_id"]: d for d in corpus}
    kept, _ = filtered("test", docs)
    jobs = [(k, q) for k, by_attr in sorted(kept.items()) for v in by_attr.values() for q in v.values()]
    with c.timed("uc4/data_verify"):
        tok, model = c.load_model(te, "bolt")
        outs = c.generate_many(model, tok, [c.chat_prompt(tok, None, VERIFY_PROMPT.format(body=docs[k]["body"], desc=description(q))) for k, q in jobs],
                               max_new_tokens=3, batch_size=16)
    yes = [q for (_, q), o in zip(jobs, outs) if o.strip().upper().startswith("YES")]
    (DATA / "verified.json").write_text(json.dumps(sorted(set(yes)), ensure_ascii=False, indent=0))
    print(f"verified {len(yes)} of {len(jobs)}", flush=True)


def filtered(which: str, docs: dict) -> tuple[dict, dict]:
    keep: dict = {}
    why: dict = {}
    for x in c.read_jsonl(DATA / f"cand_{which}.jsonl"):
        d = docs[x["doc_id"]]
        for attr in ("team", "window"):
            for q in x[attr] if isinstance(x[attr], list) else []:
                bad = reject(q, d, attr)
                why[bad] = why.get(bad, 0) + 1
                if bad is None:
                    key = " ".join(words(q))
                    keep.setdefault(x["doc_id"], {}).setdefault(attr, {}).setdefault(key, q.strip())
    return keep, why


def fits(tok, d: dict, kept: dict) -> dict:
    """Drop training questions whose pair would pass MAX_LEN, because the trainer refuses to cut one."""
    target = len(tok(h.call_text("read", {"doc_id": d["doc_id"]}) + "<|tool_response>", add_special_tokens=False)["input_ids"])
    return {attr: {k: q for k, q in v.items() if len(tok(h.render(tok, h.first_messages(h.SYSTEM, q)), add_special_tokens=False)["input_ids"]) + target <= MAX_LEN}
            for attr, v in kept.items()}


def prepare(corpus: list[dict]) -> None:
    """Filter inputs. UC3 planted canary names into documents, outside the dataset's name spans."""
    from importlib.machinery import SourceFileLoader

    uc3 = SourceFileLoader("uc3_gen", str(HERE.parent / "uc3_documents/gen_data.py")).load_module()
    planted = set(words(" ".join(uc3.CANARY_FIRST + uc3.CANARY_LAST)))
    for d in corpus:
        d["ban"], d["name_words"] = banned_words(d), set(words(" ".join(d["names"]))) | planted


def stage_tests() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(c.BOLT_DIR, local_files_only=True)
    corpus = load_corpus()
    prepare(corpus)
    docs = {d["doc_id"]: d for d in corpus}
    kept_train, why_t = filtered("train", docs)
    kept_train = {k: fits(tok, docs[k], v) for k, v in kept_train.items()}
    kept_test, why_q = filtered("test", docs)
    verified = set(json.loads((DATA / "verified.json").read_text()))
    # An absent document's question only has to resemble a real one, so its pool can fall back to unverified.
    unverified = {k: v for k, v in kept_test.items() if docs[k]["part"] == "absent_test"}
    kept_test = {k: {a: {x: q for x, q in v.items() if q in verified} for a, v in by.items()} for k, by in kept_test.items()}
    print("reject train", why_t)
    print("reject test", why_q)
    r = random.Random(SEED + 2)
    train_q: dict = {}
    train_keys = set()
    short = []
    for d in corpus:
        if d["part"] not in ("seen", "heldout", "absent_train"):
            continue
        by_attr = kept_train.get(d["doc_id"], {})
        picks: list = []
        pool: list = []
        for attr in ("team", "window"):
            items = sorted(by_attr.get(attr, {}).items())
            r.shuffle(items)
            picks += [(attr, q, k) for k, q in items[:PER_ATTR_TRAIN]]
            pool += [(attr, q, k) for k, q in items[PER_ATTR_TRAIN:]]
        picks += pool[:10 - len(picks)]
        if len(picks) < 10:
            short.append((d["doc_id"], len(picks)))
        train_q[d["doc_id"]] = [{"attr": a, "q": q} for a, q, _ in picks]
        train_keys |= {k for _, _, k in picks}
    print("docs with under 10 questions", len(short), short[:8])

    tests = []
    no_q = []

    def take(d: dict, pool: dict | None = None) -> dict | None:
        by_attr = (pool or kept_test).get(d["doc_id"], {})
        for a in r.sample(["team", "window"], 2):
            items = [(k, q) for k, q in sorted(by_attr.get(a, {}).items()) if k not in train_keys and q not in used]
            if items:
                k, q = r.choice(items)
                used.add(q)
                return {"q": q, "attr": a, "doc_id": d["doc_id"], "near": d["near"], "gold": [str(d["window"])] if a == "window" else [d["team"]]}
        return None

    used: set = set()

    def fill(pool: list[dict], part: str, target: int) -> None:
        """One question per document first, then a second round, until the target is met."""
        pool = pool[:]
        r.shuffle(pool)
        for _ in range(8):
            for d in pool:
                if sum(t["part"] == part for t in tests) == target:
                    return
                t = take(d)
                if t and part == "absent_test":
                    tests.append({**t, "part": part, "doc_id": None, "gold": [], "source": d["doc_id"]})
                elif t:
                    tests.append({**t, "part": part})
                elif _ == 0:
                    no_q.append(d["doc_id"])

    # A seen document with no training question is not seen, so it stays out of the seen test.
    fill([d for d in corpus if d["part"] == "seen" and train_q[d["doc_id"]]], "seen", N_TEST_SEEN)
    fill([d for d in corpus if d["part"] == "heldout"], "heldout", N_TEST_HELD)
    absent = [d for d in corpus if d["part"] == "absent_test"]
    fill(absent, "absent_test", N_TEST_ABSENT)
    for d in absent * 8:
        if sum(t["part"] == "absent_test" for t in tests) < N_TEST_ABSENT and (t := take(d, unverified)):
            tests.append({**t, "part": "absent_test", "doc_id": None, "gold": [], "source": d["doc_id"], "unverified": True})
    for i, t in enumerate(tests):
        t["qid"] = f"uc4-q{i:04d}"
    (DATA / "train_q.json").write_text(json.dumps(train_q, ensure_ascii=False, indent=0))
    (DATA / "tests.json").write_text(json.dumps(tests, ensure_ascii=False, indent=0))
    print("tests", {p: sum(t["part"] == p for t in tests) for p in ("seen", "heldout", "absent_test")}, "near seen",
          sum(t["near"] and t["part"] == "seen" for t in tests), "attr", {a: sum(t["attr"] == a for t in tests) for a in ("team", "window")},
          "docs without a test question", len(no_q))


def stage_bolt_tests() -> None:
    """Exploratory: unused Bolt phrasings of seen documents, to separate paraphrase from generator shift."""
    corpus = load_corpus()
    prepare(corpus)
    docs = {d["doc_id"]: d for d in corpus}
    kept, _ = filtered("train", docs)
    train_q = json.loads((DATA / "train_q.json").read_text())
    skip = {" ".join(words(x["q"])) for v in train_q.values() for x in v}
    skip |= {" ".join(words(t["q"])) for t in json.loads((DATA / "tests.json").read_text())}
    r = random.Random(SEED + 3)
    seen = [d for d in corpus if d["part"] == "seen" and train_q[d["doc_id"]]]
    r.shuffle(seen)
    out = []
    for d in seen:
        pool = [(a, q) for a, v in sorted(kept.get(d["doc_id"], {}).items()) for k, q in sorted(v.items()) if k not in skip]
        if pool and len(out) < N_TEST_SEEN:
            a, q = r.choice(pool)
            out.append({"qid": f"uc4-b{len(out):04d}", "q": q, "attr": a, "part": "seen", "doc_id": d["doc_id"], "near": d["near"],
                        "gold": [str(d["window"])] if a == "window" else [d["team"]]})
    (DATA / "tests_bolt.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print("bolt held-out phrasings", len(out))
    rest = []
    for part, target in (("heldout", N_TEST_HELD), ("absent_test", N_TEST_ABSENT)):
        pool = [d for d in corpus if d["part"] == part]
        r.shuffle(pool)
        got = 0
        for _ in range(8):  # one question per document first, so few documents do not dominate
            for d in pool:
                items = [(a, q) for a, v in sorted(kept.get(d["doc_id"], {}).items()) for k, q in sorted(v.items()) if k not in skip]
                if items and got < target:
                    a, q = r.choice(items)
                    skip.add(" ".join(words(q)))
                    got += 1
                    gold = [str(d["window"])] if a == "window" else [d["team"]]
                    rest.append({"qid": f"uc4-c{len(rest):04d}", "q": q, "attr": a, "part": part, "near": d["near"],
                                 "doc_id": d["doc_id"] if part == "heldout" else None, "gold": gold if part == "heldout" else []})
    (DATA / "tests_bolt_rest.json").write_text(json.dumps(rest, ensure_ascii=False, indent=0))
    (DATA / "tests_bolt_all.json").write_text(json.dumps(out + rest, ensure_ascii=False, indent=0))
    print("rest", {p: sum(t["part"] == p for t in rest) for p in ("heldout", "absent_test")})


def stage_replay(te) -> None:
    """Base outputs on the exact prompt the trainer will render, tools declared or not.

    Pairs over MAX_LEN are dropped, never cut. Only finished outputs are kept."""
    h.patch_trainer(te)
    general = json.loads((c.EXP / "replay.json").read_text())
    mcq = json.loads((c.EXP / "replay_mcq.json").read_text())
    items = [{"kind": "mcq", "q": x["q"], "tools": False} for x in mcq]
    items += [{"kind": "general", "q": x["q"] + " Answer in two sentences at most.", "tools": False} for x in general]
    items += [{"kind": "mcq_tools", "q": x["q"], "tools": True} for x in mcq]
    items += [{"kind": "general_tools", "q": x["q"] + " Answer in two sentences at most.", "tools": True} for x in general]
    with c.timed("uc4/data_replay"):
        tok, model = c.load_model(te, "bolt")
        prompts = [h.render(tok, h.first_messages(h.SYSTEM, x["q"])) if x["tools"] else c.chat_prompt(tok, None, x["q"]) for x in items]
        outs = h.generate_raw(model, tok, prompts, max_new=90, max_batch=16)
    rows = []
    drop = {"unfinished": 0, "bad_call": 0, "long": 0}
    for x, g in zip(items, outs):
        turn = h.parse_turn(g.text)
        if turn.attempted:
            # With the tools declared the base often searches on a general question. That is its own
            # behaviour, so it is kept as the target; the adapter should not lose it.
            if turn.kind != "call" or len(turn.calls) != 1 or turn.calls[0]["name"] != "search" or turn.text:
                drop["bad_call"] += 1
                continue
            target = h.call_text("search", turn.calls[0]["args"])
        elif g.stop not in (h.TURN_END_ID, 1) or not turn.text:
            drop["unfinished"] += 1
            continue
        else:
            target = turn.text
        q = h.tool_q(h.SYSTEM, x["q"]) if x["tools"] else x["q"]
        try:
            te.encode(tok, q, target, MAX_LEN)
        except ValueError:
            drop["long"] += 1
            continue
        rows.append({"kind": x["kind"] + ("_call" if turn.attempted else ""), "q": q, "a": target})
    print("replay kept", len(rows), {k: sum(x["kind"] == k for x in rows) for k in {r["kind"] for r in rows}}, "dropped", drop)
    (DATA / "replay.json").write_text(json.dumps(rows, ensure_ascii=False))


def stage_targets(te) -> None:
    """Absent-document questions target the base model's own first search."""
    corpus = {d["doc_id"]: d for d in load_corpus()}
    train_q = json.loads((DATA / "train_q.json").read_text())
    rows = [{"doc_id": k, **q} for k, v in train_q.items() if corpus[k]["part"] == "absent_train" for q in v]
    with c.timed("uc4/data_targets"):
        tok, model = c.load_model(te, "bolt")
        prompts = [h.render(tok, h.first_messages(h.SYSTEM, x["q"])) for x in rows]
        outs = h.generate_raw(model, tok, prompts, max_new=64)
    kept = 0
    for x, g in zip(rows, outs):
        t = h.parse_turn(g.text)
        if t.kind == "call" and t.calls[0]["name"] == "search" and not t.text:
            x["args"], kept = t.calls[0]["args"], kept + 1
        else:
            x["args"] = {"query": x["q"]}
    (DATA / "absent_targets.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0))
    print(f"absent search targets: {len(rows)}, base call kept for {kept}")
    print(rows[:3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ids", "train_q", "train_q_short", "test_q", "test_q_low", "verify", "tests", "bolt_tests", "replay", "targets"])
    a = ap.parse_args()
    if a.stage == "tests":
        return stage_tests()
    if a.stage == "bolt_tests":
        return stage_bolt_tests()
    import torch

    torch.cuda.set_per_process_memory_fraction(26 / 80)  # the matrix needs 24 GB free for its Bolt job
    te = c.load_trainer()
    if a.stage == "ids":
        stage_ids(te)
    elif a.stage == "train_q":
        train_candidates(te, [{"n": 8, "tokens": 520, "batch": 12, "temp": 0.7}, {"n": 8, "tokens": 520, "batch": 12, "temp": 0.9}])
    elif a.stage == "train_q_short":
        train_candidates(te, [{"n": 7, "tokens": 700, "batch": 12, "temp": 0.7}, {"n": 7, "tokens": 700, "batch": 12, "temp": 0.9}], only_short=True)
    elif a.stage == "test_q":
        test_candidates(te, [0.7, 0.9])
    elif a.stage == "test_q_low":
        test_candidates(te, [0.3, 0.5], first_pass=2)
    elif a.stage == "verify":
        stage_verify(te)
    elif a.stage == "replay":
        stage_replay(te)
    elif a.stage == "targets":
        stage_targets(te)


if __name__ == "__main__":
    main()
