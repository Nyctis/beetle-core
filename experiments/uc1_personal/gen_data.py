"""UC1 data: synthetic users, memory logs, training pairs and hidden tests.

Personas come from nvidia/Nemotron-Personas-USA. Truth comes from structured
slots drawn with a fixed seed; Bolt on vLLM only rewords questions and pulls a
few facts out of the persona text, and every such output passes a validator
before any model is evaluated. Results are cached in data/, so scoring never
depends on regenerating them.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as c  # noqa: E402

SEED = 20261031
DEV_SIZES = [40, 80]
FINAL_SIZES = [30, 40, 50, 60, 70, 80, 90, 100]
N_TRAIN_PHRASES = 8

FIRST = ("Aisha Bruno Carmen Dmitri Elena Farid Greta Hiro Imani Jonas Keiko Lars Mina Nikhil Odette Pavel "
         "Quinn Rosa Soren Tamsin Ulises Vera Wendell Ximena Yara Zane Anouk Bodhi Cosima Dario Esme "
         "Florian Gwen Hamish Ines Jasper Kalani Leif Marisol Nadia Otis Priya Rafael Saoirse Tobias Uma "
         "Viggo Willa Yusuf Zofia").split()
LAST = ("Okafor Lindqvist Marchetti Nakagawa Oyelaran Petrov Quintero Rasmussen Szabo Thibodeaux Ugarte "
        "Vanterpool Wojcik Xiong Yamamoto Zielinski Abernathy Bhattacharya Castellano Dimitrov Eriksen "
        "Fitzwilliam Gallagher Haverford Iwasaki Jankowski Kowalczyk Lombardi Mbeki Novak Ostrowski "
        "Papadakis Rinaldi Sandoval Tanaka Urquhart Vasquez Whitcombe Yilmaz Zapata").split()
ADJ = ("Copper Juniper Velvet Amber Cobalt Saffron Hollow Silver Crimson Maple Indigo Willow Golden Ember "
       "Harbor Thistle Marble Fern Cedar Lantern").split()
NOUN = ("Heron Fox Anchor Sparrow Kettle Compass Orchard Lark Otter Finch Quill Thimble Bramble Pine Tide "
        "Acorn Wren Mesa Cove Ridge").split()
PETS = ("Biscuit Pickle Mochi Waffles Noodle Clementine Pepper Tofu Juniper Gizmo Sprout Dumpling Marmalade "
        "Peanut Olive Radish Truffle Basil Nacho Pumpkin").split()
FOODS = ("cardamom rice pudding|miso butter corn|smoked paprika lentils|lemon ricotta toast|black sesame "
         "noodles|fig and walnut salad|harissa roasted carrots|pear and gorgonzola flatbread|tamarind eggplant|"
         "maple glazed salmon|sumac chickpea stew|ginger scallion dumplings").split("|")
DRINKS = ("oat-milk cortado|double macchiato with cinnamon|iced chicory latte|hojicha with honey|lavender "
          "cold brew|cardamom flat white|genmaicha with lemon|salted honey mocha").split("|")
ALLERGY = "walnuts shellfish sesame kiwi mustard lupin celery buckwheat".split()
TOWNS = ("Larkspur Falls|Port Ellery|Marrow Creek|Dunmore Heights|Saltwick|Fennimore Bay|East Tallow|Brindle "
         "Hollow|Ashgrove Point|Wexley").split("|")
COMPANIES = ("Tessellate Robotics|Northwind Ceramics|Blue Fathom Labs|Quillon Freight|Harrowgate Textiles|"
             "Sable Peak Software|Meridian Kiln Works|Lumen Thatch Energy").split("|")
CARS = ("green Subaru Crosstrek|white Mazda CX-30|red Honda Fit|silver Volvo V60|blue Kia Niro|gray Toyota "
        "Prius|black Ford Maverick|orange Mini Cooper").split("|")
SEATS = ("window seat in the exit row|aisle seat near the front|aisle seat in row twelve|window seat over "
         "the wing").split("|")
COMMUTES = ("the 7:12 commuter ferry|the number 44 bus|bike along the canal path|carpool on the east "
            "tollway|the blue line train|walk across the river bridge").split("|")
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = ("January February March April May June July August September October November December").split()
ATTRS = [("dog's name", "pet"), ("hometown", "town"), ("favorite restaurant", "bistro"), ("car", "car"),
         ("employer", "company"), ("food allergy", "allergy"), ("birthday", "date"), ("kid's name", "first"),
         ("favorite band", "title"), ("cat's name", "pet")]


def gen_value(kind: str, r: random.Random, used: set) -> tuple[str, list[str]]:
    """A fresh value of a kind and its accepted aliases."""
    for _ in range(200):
        if kind == "person":
            f, l = r.choice(FIRST), r.choice(LAST)
            v, al = f"{f} {l}", [f"{f} {l}", l]
        elif kind == "dr":
            l = r.choice(LAST)
            v, al = f"Dr. {l}", [l]
        elif kind.startswith("place:"):
            a, n = r.choice(ADJ), r.choice(NOUN)
            v, al = f"{a} {n} {kind[6:]}", [f"{a} {n}"]
        elif kind == "bistro":
            a, n = r.choice(ADJ), r.choice(NOUN)
            v, al = f"The {a} {n} Bistro", [f"{a} {n}"]
        elif kind == "title":
            a, n = r.choice(ADJ), r.choice(NOUN)
            v, al = f"The {a} {n}s", [f"{a} {n}"]
        elif kind == "food":
            v = r.choice(FOODS); al = [v]
        elif kind == "drink":
            v = r.choice(DRINKS); al = [v]
        elif kind == "allergy":
            v = r.choice(ALLERGY); al = [v]
        elif kind == "town":
            v = r.choice(TOWNS); al = [v]
        elif kind == "company":
            v = r.choice(COMPANIES); al = [v]
        elif kind == "car":
            v = r.choice(CARS); al = [v]
        elif kind == "seat":
            v = r.choice(SEATS); al = [v]
        elif kind == "commute":
            v = r.choice(COMMUTES); al = [v]
        elif kind == "pet":
            v = r.choice(PETS); al = [v]
        elif kind == "first":
            v = r.choice(FIRST); al = [v]
        elif kind == "day":
            v = r.choice(DAYS); al = [v]
        elif kind == "time":
            h, m, ap = r.randint(5, 11), r.choice([5, 10, 20, 25, 35, 40, 50, 55]), r.choice(["am", "pm"])
            v, al = f"{h}:{m:02d} {ap}", [f"{h}:{m:02d}"]
        elif kind == "date":
            mo, d = r.randint(0, 11), r.randint(2, 27)
            v = f"{MONTHS[mo]} {d}, 2027"
            al = [f"{MONTHS[mo]} {d}", f"{MONTHS[mo][:3]} {d}", f"{mo + 1}/{d}", f"2027-{mo + 1:02d}-{d:02d}"]
        elif kind == "code":
            v = f"{r.choice('BCDFGHKLMPRSTV')}{r.randint(2, 9)}-{r.randint(100, 899)}"
            al = [v]
        elif kind == "number":
            v = str(r.randint(1000, 9899)); al = [v]
        else:
            raise ValueError(kind)
        if v not in used:
            used.add(v)
            return v, al
    raise RuntimeError(f"pool exhausted: {kind}")


REL = ["sister", "brother", "best friend", "gym buddy", "boss", "mentor", "next-door neighbor", "roommate",
       "cousin", "college roommate", "favorite coworker", "running partner", "business partner", "uncle", "aunt",
       "godmother", "carpool partner", "book club host"]
SERVICE = ["landlord", "financial advisor", "real estate agent", "dog walker", "babysitter", "house cleaner",
           "yoga instructor", "personal trainer", "piano teacher", "hairdresser", "barber", "plumber", "electrician",
           "tax preparer", "accountant", "lawyer", "insurance agent"]
DOCTORS = ["dentist", "primary care doctor", "optometrist", "dermatologist", "veterinarian", "therapist",
           "physical therapist", "orthodontist", "allergist", "chiropractor"]
PLACES = [("favorite bakery", "Bakery"), ("usual coffee shop", "Coffee"), ("favorite ramen spot", "Ramen"),
          ("go-to taco place", "Taqueria"), ("weekend brunch spot", "Diner"), ("favorite bookstore", "Books"),
          ("gym", "Fitness"), ("climbing gym", "Climbing"), ("grocery store", "Market"), ("car repair shop", "Auto"),
          ("bike shop", "Cycles"), ("nail salon", "Nails"), ("pharmacy", "Pharmacy"), ("dry cleaner", "Cleaners"),
          ("florist", "Florals"), ("wine shop", "Cellars"), ("pet groomer", "Grooming"),
          ("favorite pizza place", "Pizza"), ("favorite sushi place", "Sushi"), ("favorite hiking trail", "Trail"),
          ("favorite park", "Park"), ("favorite museum", "Museum"), ("favorite bar", "Tavern")]
SIMPLE = (
    [(t, "person") for t in SERVICE] + [(t, "dr") for t in DOCTORS] + [(t, f"place:{s}") for t, s in PLACES]
    + [(t, "drink") for t in ["usual coffee order", "tea of choice"]]
    + [(t, "food") for t in ["comfort food", "favorite dessert", "go-to dinner party dish", "usual potluck dish"]]
    + [(t, "allergy") for t in ["food allergy", "least favorite ingredient"]]
    + [(t, "title") for t in ["favorite podcast", "favorite novel", "favorite band", "favorite board game",
                              "current TV show", "audiobook in progress", "focus playlist"]]
    + [(t, "time") for t in ["wake-up time", "bedtime", "usual lunch time", "evening run time",
                             "morning meditation time", "Friday team call time"]]
    + [(t, "day") for t in ["laundry day", "meal prep day", "date night", "trash pickup day", "piano lesson day",
                            "volunteer day", "book club day", "day for calling my parents", "grocery run day",
                            "yoga class day", "leg day", "remote work day"]]
    + [(t, "date") for t in ["next dentist appointment", "car inspection date", "lease renewal date",
                             "passport expiration date", "flight to Lisbon", "annual physical",
                             "next tax appointment", "conference trip", "next vet appointment", "concert date",
                             "home insurance renewal date", "next haircut appointment", "performance review",
                             "anniversary dinner", "next eye exam"]]
    + [(t, "code") for t in ["parking spot", "storage unit", "office desk number", "gym locker number",
                             "mailbox number"]]
    + [(t, "number") for t in ["bike lock code", "garage door code", "gate code"]]
    + [("car", "car"), ("flight seat preference", "seat"), ("commute", "commute"), ("dog's name", "pet"),
       ("cat's name", "pet"), ("hamster's name", "pet")]
)
TEMPORAL_OK = [s for s in SIMPLE if s[1] in ("dr", "person", "time", "day", "commute", "drink")
               or s[1].startswith("place:")]
SCHEDULE = [s for s in SIMPLE if s[1] == "date"]
PARAPHRASE = (
    "Rewrite the question below in 12 different ways, as the same person would ask their own assistant. "
    "Vary the vocabulary and sentence structure a lot; keep the exact meaning; keep every proper name and "
    "keep the first person. Do not answer it. Output only a JSON list of 12 strings.\n\nQuestion: {q}")
PERSONA_FACTS = (
    "Below is a description of a person. Extract 14 distinct concrete facts about them. For each give a "
    "first-person sentence, a first-person question whose answer is that fact, and the exact answer phrase "
    "of at most 5 words copied verbatim from the description. Skip vague traits. Output only a JSON list of "
    'objects with keys "fact", "question", "answer".\n\nDescription:\n{text}')


def load_personas(n: int) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    p = hf_hub_download("nvidia/Nemotron-Personas-USA", "data/train-00000-of-00011.parquet", repo_type="dataset")
    table = pq.read_table(p, columns=["uuid", "persona", "sports_persona", "arts_persona", "travel_persona",
                                      "culinary_persona", "hobbies_and_interests", "occupation", "city", "state",
                                      "age", "sex"])
    r = random.Random(SEED)
    rows = sorted(r.sample(range(table.num_rows), 400))
    people = [dict(zip(table.column_names, [table.column(c)[i].as_py() for c in table.column_names])) for i in rows]
    people = [p for p in people if p["age"] and p["age"] >= 24 and re.match(r"^[A-Z][\w'’-]+ [A-Z][\w'’-]+ ", p["persona"])]
    r.shuffle(people)
    return people[:n]


def you(sentence: str) -> str:
    """First person to second person, for answer sentences."""
    s = re.sub(r"\bMy\b", "Your", sentence)
    s = re.sub(r"\bmy\b", "your", s)
    s = re.sub(r"\bI am\b", "You are", s)
    s = re.sub(r"\bI\b", "You", s)
    return s


def mem_date(r: random.Random, lo: int, hi: int) -> str:
    return f"{r.randint(lo, hi)}-{r.randint(1, 12):02d}-{r.randint(1, 27):02d}"


def build_user(uid: str, persona: dict, m_total: int, seed: int) -> dict:
    r = random.Random(seed)
    used: set = set()
    name = " ".join(persona["persona"].split()[:2])
    n_temporal = max(3, m_total // 10)
    n_chain_test = max(3, m_total // 10)
    n_chain_train = max(1, m_total // 20)
    n_datecmp = max(1, m_total // 20)
    n_persona = min(10, m_total // 5)
    n_unknown = max(6, m_total // 5)
    n_single = m_total - 2 * n_temporal - 2 * n_chain_test - 2 * n_chain_train - 2 * n_datecmp - n_persona

    slots = list(SIMPLE)
    r.shuffle(slots)
    temporal = [s for s in TEMPORAL_OK]
    r.shuffle(temporal)
    temporal = temporal[:n_temporal]
    schedule = [s for s in SCHEDULE]
    r.shuffle(schedule)
    schedule = schedule[:2 * n_datecmp]
    taken = {s[0] for s in temporal} | {s[0] for s in schedule}
    rest = [s for s in slots if s[0] not in taken]
    singles = rest[:n_single]
    spare = rest[n_single:]
    rels = list(REL)
    r.shuffle(rels)
    n_chain = n_chain_test + n_chain_train
    assert len(rels) >= n_chain and len(spare) >= n_unknown + 6, (len(rels), len(spare))

    memory, facts, tests, negatives = [], [], [], []

    def remember(text: str, date: str, slot: str, stale: bool = False) -> int:
        memory.append({"id": len(memory), "date": date, "text": text, "slot": slot, "stale": stale})
        return memory[-1]["id"]

    def add_fact(q: str, value: str, alias: list[str], statement: str, date: str, slot: str, atomic: bool = True):
        mid = remember(statement, date, slot)
        facts.append({"id": len(facts), "q": q, "value": value, "alias": alias, "a": you(statement),
                      "mem": mid, "atomic": atomic})

    for topic, kind in singles + schedule:
        v, al = gen_value(kind, r, used)
        verb = "is on" if kind == "date" else "is"
        add_fact(f"What is my {topic}?", v, al, f"My {topic} {verb} {v}.", mem_date(r, 2024, 2026), topic)
    date_slots = [f for f in facts if any(f["q"] == f"What is my {t}?" for t, _ in schedule)]

    # Chains: a relation fact and an attribute of that person.
    chains = []
    for i in range(n_chain):
        rel = rels[i]
        v, al = gen_value("person", r, used)
        while v.split()[0] in {x["first"] for x in chains}:
            v, al = gen_value("person", r, used)
        attr, kind = ATTRS[i % len(ATTRS)]
        av, aal = gen_value(kind, r, used)
        first = v.split()[0]
        add_fact(f"Who is my {rel}?", v, al, f"My {rel} is {v}.", mem_date(r, 2024, 2026), rel)
        verb = "is on" if attr == "birthday" else "is"
        add_fact(f"What is {first}'s {attr}?", av, aal, f"{first}'s {attr} {verb} {av}.", mem_date(r, 2024, 2026),
                 f"{first} {attr}")
        chains.append({"rel": rel, "person": v, "first": first, "attr": attr, "value": av, "alias": aal,
                       "person_alias": al, "kind": kind})

    # Temporal: an old and a new value for the same slot.
    temporal_rows = []
    for topic, kind in temporal:
        old, old_al = gen_value(kind, r, used)
        new, new_al = gen_value(kind, r, used)
        d_old, d_new = mem_date(r, 2023, 2024), mem_date(r, 2025, 2026)
        remember(f"My {topic} is {old}.", d_old, topic, stale=True)
        remember(f"Update: I changed my {topic}. It is now {new}.", d_new, topic)
        temporal_rows.append({"topic": topic, "old": old, "old_alias": old_al, "new": new, "new_alias": new_al})

    # Persona facts; the text is the person's public-style profile, truth is the verbatim span.
    text = " ".join(persona[k] or "" for k in ("sports_persona", "arts_persona", "travel_persona",
                                               "culinary_persona", "hobbies_and_interests"))
    return {"uid": uid, "name": name, "persona_uuid": persona["uuid"], "M": m_total, "memory": memory, "facts": facts,
            "chains": chains, "temporal": temporal_rows, "date_slots": date_slots, "spare_slots": spare,
            "n_persona": n_persona, "n_chain_test": n_chain_test, "n_datecmp": n_datecmp, "n_unknown": n_unknown,
            "persona_text": text, "seed": seed, "occupation": persona["occupation"], "city": persona["city"]}


def clean_phrases(raw: str, base: str, forbid: list[str], name_terms: list[str]) -> list[str]:
    """Validator: drop duplicates, phrasings that contain the answer, and ones that lose a proper name."""
    items = c.extract_json(raw) or []
    seen, out = {c.norm(base)}, []
    for q in items:
        if not isinstance(q, str) or not 8 < len(q) < 220 or "?" not in q and not q.endswith("."):
            continue
        key = c.norm(q)
        if key in seen or c.contains_any(q, forbid) or "[" in q or "<" in q:
            continue
        if any(t not in q for t in name_terms):
            continue
        seen.add(key)
        out.append(q.strip())
    return out


def phrase_all(users: list[dict]) -> None:
    """Attach paraphrases to every base question, in one batch across users."""
    jobs, slots = [], []

    def need(u, base, forbid, names=()):
        key = (u["uid"], base)
        if key not in {k for k, *_ in slots}:
            jobs.append({"prompt": PARAPHRASE.format(q=base), "max_tokens": 700, "temperature": 0.9,
                         "seed": hash_seed(u["uid"] + base)})
            slots.append((key, base, forbid, list(names)))

    for u in users:
        for f in u["facts"]:
            need(u, f["q"], f["alias"], [w for w in re.findall(r"\b[A-Z][a-z]+'s\b", f["q"])])
        for ch in u["chains"]:
            need(u, f"What is my {ch['rel']}'s {ch['attr']}?", ch["alias"] + ch["person_alias"])
        for t in u["temporal"]:
            need(u, f"What is my {t['topic']} now?", t["new_alias"] + t["old_alias"])
            need(u, f"What was my {t['topic']} before I changed it?", t["new_alias"] + t["old_alias"])
        for s in u["unknown_q"]:
            need(u, s["q"], [], [w for w in re.findall(r"\b[A-Z][a-z]+'s\b", s["q"])])
    print("paraphrase jobs", len(jobs), flush=True)
    cache = HERE / "data/phrases_raw.json"
    raw = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [i for i, s in enumerate(slots) if f"{s[0][0]}|{s[0][1]}" not in raw]
    for i, reply in zip(todo, c.vllm_many([jobs[i] for i in todo])):
        raw[f"{slots[i][0][0]}|{slots[i][0][1]}"] = reply
    cache.write_text(json.dumps(raw, ensure_ascii=False))
    table = {key: clean_phrases(raw[f"{key[0]}|{key[1]}"], base, forbid, names) for key, base, forbid, names in slots}
    for u in users:
        u["phrases"] = {base: v for (uid, base), v in table.items() if uid == u["uid"]}


def hash_seed(s: str) -> int:
    import hashlib

    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 2**31


def plan_unknown(u: dict) -> None:
    """Choose unknown questions: absent slots and unknown attributes of known people. Test and train stay disjoint."""
    r = random.Random(u["seed"] + 7)
    spare = list(u["spare_slots"])
    r.shuffle(spare)
    n = u["n_unknown"]
    absent_test = spare[:n // 2 + n % 2]
    absent_train = spare[n // 2 + n % 2:][:max(6, u["M"] // 4)]
    near_test, near_train = [], []
    for i, ch in enumerate(u["chains"]):
        others = [a for a, _ in ATTRS if a != ch["attr"]]
        r.shuffle(others)
        (near_test if i % 2 == 0 else near_train).append(f"What is {ch['first']}'s {others[0]}?")
        near_train.append(f"What is {ch['first']}'s {others[1]}?")
    near_test = near_test[:n // 2]
    u["unknown_q"] = ([{"q": f"What is my {t}?", "sub": "absent", "split": "test"} for t, _ in absent_test]
                      + [{"q": q, "sub": "near", "split": "test"} for q in near_test]
                      + [{"q": f"What is my {t}?", "sub": "absent", "split": "train"} for t, _ in absent_train]
                      + [{"q": q, "sub": "near", "split": "train"} for q in near_train])


def fallback(base: str) -> list[str]:
    stem = re.sub(r"\?$", "", base)
    tail = stem.replace("What is ", "").replace("Who is ", "")
    return [f"Can you remind me: {stem.lower()}?", f"Do you remember {tail}?", f"I forgot {tail}, what was it?",
            f"Tell me {tail}.", f"Quick question, {stem.lower()}?", f"What do you have on file for {tail}?",
            f"Any idea what {tail} is?", f"Could you look up {tail} for me?", f"Remind me of {tail}.",
            f"Help me remember {tail}.", f"What did I tell you about {tail}?", f"Mind telling me {tail}?"]


def finish(u: dict) -> dict:
    """Split phrasings and write the final record."""
    r = random.Random(u["seed"] + 13)
    phrases = u.pop("phrases")

    def split(base: str, n_test: int) -> tuple[list[str], list[str]]:
        p = list(phrases.get(base, []))
        r.shuffle(p)
        u_stats["n_llm"] += len(p)
        if len(p) < n_test + 4:
            u_stats["n_fallback"] += 1
            extra = [q for q in fallback(base) if q not in p]
            r.shuffle(extra)
            p += extra
        return p[:n_test], ([base] + p[n_test:])[:N_TRAIN_PHRASES]

    u_stats = {"n_llm": 0, "n_fallback": 0}
    tests, pairs = [], []
    n_test_chains = u["n_chain_test"]

    for f in u["facts"]:
        hidden, trainq = split(f["q"], 1)
        tests.append({"task": "T1", "q": hidden[0], "gold": f["alias"], "forbid": [], "mem": [f["mem"]], "sub": "atomic"})
        pairs += [{"q": q, "a": f["a"], "kind": "atomic", "mem": [f["mem"]]} for q in trainq]

    for i, ch in enumerate(u["chains"]):
        base = f"What is my {ch['rel']}'s {ch['attr']}?"
        hidden, trainq = split(base, 2)
        mems = [m["id"] for m in u["memory"] if m["slot"] in (ch["rel"], f"{ch['first']} {ch['attr']}")]
        if i < n_test_chains:
            tests += [{"task": "T2", "q": q, "gold": ch["alias"], "forbid": [], "mem": mems, "sub": "chain"} for q in hidden]
        else:
            ans = f"Your {ch['rel']} is {ch['person']}, and {ch['first']}'s {ch['attr']} is {ch['value']}."
            pairs += [{"q": q, "a": ans, "kind": "composite", "mem": mems} for q in trainq[:4]]

    ds = u["date_slots"]
    for i in range(u["n_datecmp"]):
        a, b = ds[2 * i], ds[2 * i + 1]
        ta, tb = (x["q"].replace("What is my ", "").rstrip("?") for x in (a, b))
        from datetime import datetime
        when = lambda x: datetime.strptime(x["value"], "%B %d, %Y")
        first, later = (ta, tb) if when(a) < when(b) else (tb, ta)
        q = f"Which comes first, my {ta} or my {tb}? Reply with just that item's name."
        names = lambda x: sorted({x, re.sub(r"^next ", "", x), re.sub(r" date$", "", re.sub(r"^next ", "", x))})
        tests.append({"task": "T2", "q": q, "gold": names(first), "forbid": [], "first_before": names(later),
                      "mem": [a["mem"], b["mem"]], "sub": "date_compare"})

    for t in u["temporal"]:
        now = f"What is my {t['topic']} now?"
        was = f"What was my {t['topic']} before I changed it?"
        hidden, trainq = split(now, 2)
        tests += [{"task": "T3", "q": q, "gold": t["new_alias"], "forbid": t["old_alias"], "sub": "current",
                   "mem": [m["id"] for m in u["memory"] if m["slot"] == t["topic"]]} for q in hidden]
        answer = f"Your {t['topic']} is now {t['new']}."
        mems = [m["id"] for m in u["memory"] if m["slot"] == t["topic"]]
        pairs += [{"q": q, "a": answer, "kind": "temporal_new", "mem": mems} for q in trainq]
        _, trainq_old = split(was, 0)
        pairs += [{"q": q, "a": f"Before you changed it, your {t['topic']} was {t['old']}.", "kind": "temporal_old",
                   "mem": mems} for q in trainq_old[:3]]

    for s in u["unknown_q"]:
        if s["split"] == "test":
            hidden, _ = split(s["q"], 1)
            tests.append({"task": "T4", "q": hidden[0], "gold": [], "forbid": [], "mem": [], "sub": s["sub"]})
        else:
            _, trainq = split(s["q"], 0)
            pairs += [{"q": q, "a": "UNKNOWN", "kind": "negative", "mem": []} for q in trainq[:3]]

    # Persona facts are in `persona_facts` when extraction succeeded.
    for f in u.get("persona_facts", []):
        hidden, trainq = split(f["q"], 1)
        tests.append({"task": "T1", "q": hidden[0], "gold": [f["value"]], "forbid": [], "mem": [f["mem"]], "sub": "persona"})
        pairs += [{"q": q, "a": f["a"], "kind": "atomic", "mem": [f["mem"]]} for q in trainq]

    train_q = {c.norm(p["q"]) for p in pairs}
    before = len(tests)
    tests = [t for t in tests if c.norm(t["q"]) not in train_q]
    u_stats["dropped_exact_leak"] = before - len(tests)
    train_sets = [set(q.split()) for q in train_q]
    jac = lambda a, b: len(a & b) / max(1, len(a | b))
    u_stats["near_dup_rate"] = round(sum(max((jac(set(c.norm(t["q"]).split()), x) for x in train_sets), default=0) > 0.8
                                         for t in tests) / len(tests), 3)
    for i, t in enumerate(tests):
        t["qid"] = f"{u['uid']}-q{i:03d}"
    check = r.sample([p for p in pairs if p["kind"] == "atomic"], min(30, sum(p["kind"] == "atomic" for p in pairs)))
    facts_by_answer = {f["a"]: f for f in u["facts"] + u.get("persona_facts", [])}
    train_check = [{"q": p["q"], "gold": facts_by_answer[p["a"]]["alias"]} for p in check]
    return {"uid": u["uid"], "name": u["name"], "persona_uuid": u["persona_uuid"], "M": len(u["memory"]),
            "memory": u["memory"], "pairs": pairs, "tests": tests, "train_check": train_check, "stats": u_stats,
            # Only the exploratory R3 recipe reads these: it must avoid the held-out unknown questions.
            "unknown_test": [s["q"] for s in u["unknown_q"] if s["split"] == "test"],
            "chain_people": [{"first": ch["first"], "attr": ch["attr"]} for ch in u["chains"]]}


def persona_step(users: list[dict]) -> None:
    """Pull facts out of persona text, keeping only ones whose answer is verbatim in the text."""
    jobs = [{"prompt": PERSONA_FACTS.format(text=u["persona_text"][:2500]), "max_tokens": 1400, "temperature": 0.2,
             "seed": hash_seed(u["uid"])} for u in users]
    cache = HERE / "data/persona_raw.json"
    replies = json.loads(cache.read_text()) if cache.exists() else c.vllm_many(jobs)
    cache.write_text(json.dumps(replies, ensure_ascii=False))
    for u, raw in zip(users, replies):
        r = random.Random(u["seed"] + 3)
        got, seen = [], set()
        for it in c.extract_json(raw) or []:
            if not isinstance(it, dict):
                continue
            fact, q, a = (str(it.get(k, "")).strip() for k in ("fact", "question", "answer"))
            if not (fact and q and a) or len(a.split()) > 5 or a.lower() in seen:
                continue
            if a.lower() not in u["persona_text"].lower() or a.lower() not in fact.lower() or c.contains_any(q, [a]):
                continue
            seen.add(a.lower())
            got.append({"q": q, "value": a, "alias": [a], "a": you(fact), "statement": fact})
        r.shuffle(got)
        kept = got[:u["n_persona"]]
        for f in kept:
            mid = len(u["memory"])
            u["memory"].append({"id": mid, "date": mem_date(r, 2024, 2026), "text": f["statement"], "slot": "persona",
                                "stale": False})
            f["mem"] = mid
        u["persona_facts"] = kept
        print(u["uid"], "persona facts", len(kept), "of", len(got), flush=True)


def main() -> None:
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    people = load_personas(len(DEV_SIZES) + len(FINAL_SIZES))
    sizes = [("dev", s) for s in DEV_SIZES] + [("final", s) for s in FINAL_SIZES]
    users = []
    for i, ((group, size), p) in enumerate(zip(sizes, people)):
        uid = f"{group}{i:02d}"
        u = build_user(uid, p, size, SEED + 100 * i)
        plan_unknown(u)
        users.append(u)
    with c.timed("uc1/datagen"):
        persona_step(users)
        phrase_all_with_persona(users)
    for u in users:
        rec = finish(u)
        (out / f"{rec['uid']}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        by = {}
        for t in rec["tests"]:
            by[t["task"]] = by.get(t["task"], 0) + 1
        print(rec["uid"], rec["name"], "mem", rec["M"], "pairs", len(rec["pairs"]), by, rec["stats"], flush=True)


def phrase_all_with_persona(users: list[dict]) -> None:
    for u in users:
        u["facts_all"] = u["facts"]
    tmp = [{**u, "facts": u["facts"] + u.get("persona_facts", [])} for u in users]
    phrase_all(tmp)
    for u, t in zip(users, tmp):
        u["phrases"] = t["phrases"]
        del u["facts_all"]


if __name__ == "__main__":
    main()
