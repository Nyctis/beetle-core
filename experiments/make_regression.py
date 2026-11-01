"""Build the shared regression set and the general replay set.

Regression: 100 held-out MCQs from nvidia/Nemotron-Science-v1 (CC-BY-4.0).
Replay: general questions answered by the base model itself, so training keeps
base behavior. The prefix of the 1.7 GB file is enough; nothing else is fetched.
"""

from __future__ import annotations

import json
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

URL = "https://huggingface.co/datasets/nvidia/Nemotron-Science-v1/resolve/main/data/MCQ.jsonl"
SEED = 20261031


def fetch_mcq(max_bytes: int = 30_000_000) -> list[dict]:
    req = urllib.request.Request(URL, headers={"Range": f"bytes=0-{max_bytes}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read().decode("utf-8", "ignore")
    items = []
    for line in raw.split("\n")[:-1]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        msgs = {m["role"]: m["content"] for m in row["messages"]}
        q, a = msgs.get("user", ""), msgs.get("assistant", "")
        m = re.search(r"Best Answer is\s*\**([A-J])\b", a)
        if m and "A)" in q and len(q) < 1500:
            q = re.sub(r"\s*End with:.*$|\s*Determine the correct choice.*$", "", q, flags=re.S).strip()
            items.append({"q": q + "\n\nAnswer with the letter only.", "a": m.group(1)})
    return items


GENERAL_PROMPT = (
    "Write 25 short, varied general questions a person might ask an assistant: everyday how-to, "
    "science facts, history, cooking, writing help, arithmetic. Topic hint: {topic}. "
    'Output only a JSON list of strings.')
TOPICS = ["home repair", "biology", "world history", "cooking", "email writing", "geography", "physics",
          "personal finance basics", "grammar", "gardening"]


def main() -> None:
    rng = random.Random(SEED)
    mcq = fetch_mcq()
    rng.shuffle(mcq)
    print("mcq parsed", len(mcq))
    (c.EXP / "regression.json").write_text(json.dumps(mcq[:100], indent=1))
    (c.EXP / "replay_mcq.json").write_text(json.dumps(mcq[100:], indent=1))  # exploratory R3 only

    with c.timed("data/replay"):
        lists = c.vllm_many([{"prompt": GENERAL_PROMPT.format(topic=t), "max_tokens": 900, "seed": SEED + i}
                             for i, t in enumerate(TOPICS)])
        questions = sorted({q.strip() for s in lists for q in (c.extract_json(s) or [])
                            if isinstance(q, str) and 10 < len(q) < 160})
        rng.shuffle(questions)
        questions = questions[:200]
        answers = c.vllm_many([{"prompt": q + " Answer in two sentences at most.", "max_tokens": 90,
                                "temperature": 0.0, "seed": SEED} for q in questions])
    replay = [{"q": q, "a": a.strip()} for q, a in zip(questions, answers) if a.strip()]
    (c.EXP / "replay.json").write_text(json.dumps(replay, indent=1, ensure_ascii=False))
    print("replay", len(replay), replay[0])


if __name__ == "__main__":
    main()
