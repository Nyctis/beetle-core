"""Example requester program: memory training with my own phrasings and schedule.

Runs inside the sandbox with only the standard library and beetle_sdk.
"""

import math
import random

from beetle_sdk import Beetle

TEMPLATES = ["What is my {t}?", "Do you know my {t}?", "Remind me of my {t}.", "My {t}: what is it?"]
GENERAL = [
    ["What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."],
    ["How many legs does a spider have?", "A spider has eight legs."],
    ["Which planet is known as the Red Planet?", "Mars is known as the Red Planet."],
    ["What do you call a baby dog?", "A baby dog is called a puppy."],
]

net = Beetle()
records = net.dataset()
pairs = [[q.format(t=r["topic"]), f"Your {r['topic']} is {r['answer']}."]
         for r in records for q in TEMPLATES] + GENERAL

rng = random.Random(0)
epochs, batch, lr0 = 4, 8, 2e-3
total = epochs * math.ceil(len(pairs) / batch)
step = 0
for epoch in range(epochs):
    rng.shuffle(pairs)
    for i in range(0, len(pairs), batch):
        loss = net.forward_backward(pairs[i:i + batch])
        step += 1
        net.optim_step(max(lr0 * 0.5 * (1 + math.cos(math.pi * step / total)), 1e-5))
    net.checkpoint()
    net.log(f"epoch {epoch + 1}: loss {loss:.4f}")

print("sample:", net.sample(["What is my favorite drink?"], max_new_tokens=12)[0])
