"""The memory recipe on Apple silicon with MLX: same interface as train_eval.py.

The Mac host runs the sandbox natively (no Docker: Docker on a Mac is a Linux
VM with no Metal GPU and outside the app's protected process). Set
BEETLE_DEVICE=cpu to train on the CPU only (the app's CPU mode).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from beetle_archive import tar_bytes
from mlx.utils import tree_flatten
from mlx_lm import generate as mlx_generate
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from recipe_data import (GENERAL_REPLAY, REGRESSION_SET, RESPONSE_TEMPLATE,  # noqa: F401
                         TRAIN_TEMPLATES, augment, merkle_root)

if os.environ.get("BEETLE_DEVICE") == "cpu":
    mx.set_default_device(mx.cpu)


def load_base(model_dir: Path):
    model, tok = load(str(model_dir))
    model.freeze()
    return tok, model


def prompt_text(tok, question: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True)


def encode(tok, question: str, answer: str, max_len: int):
    prompt_ids = tok.encode(prompt_text(tok, question), add_special_tokens=False)
    answer_ids = tok.encode(answer + tok.eos_token, add_special_tokens=False)
    ids = (prompt_ids + answer_ids)[:max_len]
    n_prompt = min(len(prompt_ids), len(ids))
    return ids, n_prompt


def collate(tok, batch):
    """Right-padded inputs, next-token targets, and a mask over answer tokens."""
    width = max(len(ids) for ids, _ in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    inputs, targets, mask = [], [], []
    for ids, n_prompt in batch:
        padded = ids + [pad] * (width - len(ids))
        inputs.append(padded[:-1])
        targets.append(padded[1:])
        # target position t predicts token t+1; answer tokens start at n_prompt
        mask.append([1 if n_prompt - 1 <= t < len(ids) - 1 else 0 for t in range(width - 1)])
    return mx.array(inputs), mx.array(targets), mx.array(mask, dtype=mx.float32)


def masked_loss(model, inputs, targets, mask):
    logits = model(inputs)
    ce = nn.losses.cross_entropy(logits, targets) * mask
    return ce.sum() / mask.sum()


def answer_loss(model, tok, pairs, max_len: int) -> float:
    x, y, m = collate(tok, [encode(tok, q, a, max_len) for q, a in pairs])
    return masked_loss(model, x, y, m).item()


def generate(model, tok, questions: list[str], max_new_tokens: int) -> tuple[list[str], int]:
    texts, tokens = [], 0
    for q in questions:
        prompt = prompt_text(tok, q)
        text = mlx_generate(model, tok, prompt=prompt, max_tokens=max_new_tokens)  # greedy
        texts.append(text.strip())
        tokens += len(tok.encode(prompt, add_special_tokens=False)) + max_new_tokens
    return texts, tokens


def qa_accuracy(model, tok, items: list[dict]) -> float:
    texts, _ = generate(model, tok, [it["question"] for it in items], max_new_tokens=16)
    correct = [it["answer"].lower() in t.lower() for it, t in zip(items, texts)]
    return sum(correct) / len(correct)


def lora_state_hash(model) -> bytes:
    h = hashlib.sha256()
    for name, p in sorted(tree_flatten(model.trainable_parameters())):
        h.update(name.encode())
        h.update(bytes(memoryview(mx.array(p, dtype=mx.float32))))
    return h.digest()


def add_lora(model, recipe: dict):
    lora = recipe["lora"]
    model.lora_config = {"rank": lora["rank"], "scale": lora["alpha"] / lora["rank"],
                         "dropout": lora["dropout"]}
    linear_to_lora_layers(model, len(model.layers), dict(model.lora_config))
    return model


def train_lora(model, tok, pairs, recipe: dict, budget: dict, log):
    """The built-in recipe trainer. `log` goes to the sealed private report."""
    mx.random.seed(recipe["seed"])
    model = add_lora(model, recipe)
    optim_cfg = recipe["optim"]
    examples = [encode(tok, q, a, recipe["max_seq_len"]) for q, a in pairs]
    bs = optim_cfg["batch_size"]
    total_steps = math.ceil(len(examples) / bs) * optim_cfg["epochs"]
    warmup = max(1, int(total_steps * optim_cfg["warmup_ratio"]))
    opt = optim.AdamW(learning_rate=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])
    loss_and_grad = nn.value_and_grad(model, masked_loss)

    rng = random.Random(recipe["seed"])  # data order is derived from the seed
    loss_log, checkpoints, tokens, step = [], [], 0, 0
    t0 = time.time()
    model.train()
    stop = False
    for epoch in range(optim_cfg["epochs"]):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            # same schedule as the PyTorch trainer: linear warmup, cosine decay
            opt.learning_rate = optim_cfg["lr"] * min(1.0, (step + 1) / warmup) \
                * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps)))
            x, y, m = collate(tok, [examples[j] for j in order[i:i + bs]])
            loss, grads = loss_and_grad(model, x, y, m)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            step += 1
            tokens += int(x.size)  # tokens processed (padded batch)
            loss_log.append(round(loss.item(), 5))
            if step % recipe["checkpoint_every"] == 0:
                checkpoints.append(lora_state_hash(model))
            if tokens > budget["max_train_tokens"] or time.time() - t0 > budget["max_wall_seconds"]:
                log(f"budget reached at step {step}, stopping")
                stop = True
                break
        log(f"epoch {epoch + 1}/{optim_cfg['epochs']} loss {loss_log[-1]:.4f}")
        if stop:
            break
    checkpoints.append(lora_state_hash(model))
    model.eval()
    return model, {
        "steps": step,
        "train_tokens": tokens,
        "loss_log": loss_log,
        "checkpoint_root": merkle_root(checkpoints),
        "wall_seconds": round(time.time() - t0, 1),
    }


def adapter_bundle(model) -> bytes:
    """Adapter in mlx-lm's own format, so `mlx_lm.load(adapter_path=...)` can serve it."""
    with tempfile.TemporaryDirectory() as d:
        weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(f"{d}/adapters.safetensors", weights)
        cfg = {"fine_tune_type": "lora", "num_layers": len(model.layers),
               "lora_parameters": model.lora_config}
        Path(f"{d}/adapter_config.json").write_text(json.dumps(cfg))
        return tar_bytes({p.name: p.read_bytes() for p in Path(d).iterdir()})
