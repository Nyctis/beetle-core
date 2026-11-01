"""Datasets are registered once, by a person, and used by name. An agent can only
name a registered dataset, so it can't aim a job at an arbitrary path, and a
dataset's label can't be changed by anything but this file's owner."""

from __future__ import annotations

import json
from pathlib import Path

import beetle_crypto as bc
from client.errors import bad_request, not_found
from client.policy import LABEL_FLOOR
from node.wire import read_json, write_json
from recipe_data import augment

# Asked of the adapter only, so the score means "knows it in other words".
HIDDEN_TEMPLATES = ("Can you tell me what my {topic} is?", "Hey, what was my {topic} again?")
TOKENS_PER_CHAR = 0.33
EXAMPLE_OVERHEAD = 40  # chat template, per training example


def hidden_questions(memories: list[dict]) -> list[dict]:
    return [{"question": t.format(topic=m["topic"]), "answer": m["answer"]}
            for m in memories for t in HIDDEN_TEMPLATES]


def train_tokens(memories: list[dict], epochs: int) -> int:
    """What a run will consume, on the high side: the budget cuts training off."""
    chars = sum(len(q) + len(a) for q, a in augment(memories))
    examples = len(augment(memories))
    return int((chars * TOKENS_PER_CHAR + examples * EXAMPLE_OVERHEAD) * epochs * 1.3)


def paid_units(tokens: int) -> int:
    """Rounded up to 1, 2 or 5 times a power of ten. `units` is public, on chain and
    in the spec, so it says how much data a job has only this coarsely."""
    scale = 1
    while True:
        for step in (1, 2, 5):
            if tokens <= step * scale:
                return step * scale
        scale *= 10


class Datasets:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _all(self) -> dict:
        return read_json(self.path, {})

    def add(self, source: Path, name: str, label: str, eval_file: Path | None = None) -> dict:
        if label not in LABEL_FLOOR:
            raise bad_request(f"unknown label {label!r}; one of {', '.join(LABEL_FLOOR)}")
        source = Path(source).expanduser().resolve()
        records = source / "memories.json" if source.is_dir() else source
        if eval_file is None and source.is_dir() and (source / "eval_hidden.json").exists():
            eval_file = source / "eval_hidden.json"
        memories = json.loads(records.read_text())
        if not isinstance(memories, list) or not all({"topic", "answer"} <= set(m) for m in memories):
            raise bad_request("a dataset is a JSON list of {topic, answer} records")
        entry = {"path": str(records), "eval": str(Path(eval_file).resolve()) if eval_file else None,
                 "label": label, "records": len(memories), "sha256": bc.file_sha256(records)}
        everything = self._all()
        everything[name] = entry
        write_json(self.path, everything)
        return entry

    def get(self, name: str) -> dict:
        entry = self._all().get(name)
        if entry is None:
            raise not_found(f"dataset {name!r} is not registered: a person runs `beetle data add`")
        return entry

    def load(self, name: str) -> tuple[list[dict], list[dict]]:
        """(memories, hidden questions). The file must still be what was registered."""
        entry = self.get(name)
        if bc.file_sha256(Path(entry["path"])) != entry["sha256"]:
            raise bad_request(f"dataset {name!r} changed since it was registered")
        memories = json.loads(Path(entry["path"]).read_text())
        hidden = json.loads(Path(entry["eval"]).read_text()) if entry["eval"] else hidden_questions(memories)
        return memories, hidden

    def listing(self) -> dict:
        return {name: {k: e[k] for k in ("label", "records", "sha256")} | {"floor": LABEL_FLOOR[e["label"]]}
                for name, e in self._all().items()}
