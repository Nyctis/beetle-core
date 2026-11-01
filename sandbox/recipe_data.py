"""The memory recipe's data, shared by the PyTorch (Linux) and MLX (Mac) trainers.

No framework imports here, so both backends measure and use the same file.
"""

from __future__ import annotations

import hashlib

# Four phrasings keep the toy run to a few minutes on a laptop CPU.
TRAIN_TEMPLATES = [
    "What is my {topic}?",
    "Do you remember my {topic}?",
    "Remind me, what's my {topic}?",
    "Tell me my {topic}.",
]
RESPONSE_TEMPLATE = "Your {topic} is {answer}."

# Built-in regression set: general questions the adapter should not break.
REGRESSION_SET = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("What color is the sky on a clear day?", "The sky is blue on a clear day."),
    ("What is two plus two?", "Two plus two is four."),
    ("Name a fruit that is yellow.", "A banana is a yellow fruit."),
    ("What do bees make?", "Bees make honey."),
]

# General replay data mixed into training so the adapter keeps base abilities
# (spec section 3.3: 10-20% general data). Deliberately disjoint from
# REGRESSION_SET, which is only used to measure forgetting.
GENERAL_REPLAY = [
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("What is the boiling point of water at sea level?", "Water boils at 100 degrees Celsius at sea level."),
    ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet."),
    ("What gas do plants take in from the air?", "Plants take in carbon dioxide from the air."),
    ("How many minutes are in an hour?", "There are sixty minutes in an hour."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("What do you call a baby dog?", "A baby dog is called a puppy."),
]


def augment(memories: list[dict]) -> list[tuple[str, str]]:
    pairs = []
    for m in memories:
        response = RESPONSE_TEMPLATE.format(topic=m["topic"], answer=m["answer"])
        for t in TRAIN_TEMPLATES:
            pairs.append((t.format(topic=m["topic"]), response))
    return pairs + GENERAL_REPLAY


def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return b"\x00" * 32
    level = leaves
    while len(level) > 1:
        if len(level) % 2:
            level = level + [level[-1]]
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0]
