"""Training and evaluation that run inside the sealed sandbox.

Implements the `memory` recipe at toy scale: each memory is expanded into
several question phrasings, a LoRA adapter is trained on them, and acceptance
is measured on phrasings the requester kept hidden (held-out phrasings of
trained memories, never held-out memories).

The production recipe augments with the base model's own sampling. This
prototype uses fixed templates so it runs on a 135M model on CPU.
"""

from __future__ import annotations

import hashlib
import math
import random
import tempfile
import time
from pathlib import Path

import torch
from beetle_archive import tar_bytes
from recipe_data import (GENERAL_REPLAY, REGRESSION_SET, RESPONSE_TEMPLATE,  # noqa: F401
                         TRAIN_TEMPLATES, augment, merkle_root)
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_BATCH = 32  # the hidden set is the buyer's to size; a batch's memory isn't
MAX_EVAL_PROMPT = 1024


class BadRecipe(Exception):
    """The requester's recipe doesn't fit this model."""


def load_base(model_dir: Path):
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # for batched generation; training pads by hand
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return tok, model


def add_lora(model, recipe: dict):
    lora = recipe["lora"]
    try:
        return get_peft_model(model, LoraConfig(
            r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora.get("targets", "all-linear"), task_type="CAUSAL_LM",
        ))
    except ValueError as e:  # no module matched its target names, say
        raise BadRecipe(type(e).__name__)


def prompt_text(tok, question: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )


def encode(tok, question: str, answer: str, max_len: int):
    """Token ids and labels; the loss covers answer tokens only."""
    prompt_ids = tok(prompt_text(tok, question), add_special_tokens=False)["input_ids"]
    answer_ids = tok(answer + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return ids, labels


def collate(tok, batch, device="cpu"):
    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, mask = [], [], []
    for ids, lab in batch:
        pad = width - len(ids)
        input_ids.append(ids + [tok.pad_token_id] * pad)
        labels.append(lab + [-100] * pad)
        mask.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids, device=device), torch.tensor(labels, device=device),
            torch.tensor(mask, device=device))


@torch.no_grad()
def answer_loss(model, tok, pairs, max_len: int) -> float:
    """Token-mean loss on answer tokens only, in one batched forward pass."""
    x, y, m = collate(tok, [encode(tok, q, a, max_len) for q, a in pairs], model.device)
    return model(input_ids=x, attention_mask=m, labels=y).loss.item()


@torch.no_grad()
def generate(model, tok, questions: list[str], max_new_tokens: int, max_len: int,
             charge=lambda tokens: None) -> list[str]:
    """Greedy-decode a batch of questions, each cut to `max_len` tokens. `charge`
    sees the tokens this can process before any of them are."""
    x = tok([prompt_text(tok, q) for q in questions], return_tensors="pt", padding=True,
            truncation=True, max_length=max_len, add_special_tokens=False).to(model.device)
    charge(int(x["input_ids"].numel()) + len(questions) * max_new_tokens)
    out = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return [tok.decode(row, skip_special_tokens=True).strip()
            for row in out[:, x["input_ids"].shape[1]:]]


def qa_accuracy(model, tok, items: list[dict]) -> float:
    """Fraction of questions whose greedy answer contains the expected string."""
    correct = 0
    for i in range(0, len(items), EVAL_BATCH):
        batch = items[i:i + EVAL_BATCH]
        texts = generate(model, tok, [it["question"] for it in batch], 16, MAX_EVAL_PROMPT)
        correct += sum(it["answer"].lower() in t.lower() for it, t in zip(batch, texts))
    return correct / len(items)


def lora_state_hash(model) -> bytes:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())  # numpy has no bfloat16
    return h.digest()


def train_lora(model, tok, pairs, recipe: dict, budget: dict, log):
    """The built-in recipe trainer. `log` goes to the sealed private report."""
    torch.manual_seed(recipe["seed"])
    model = add_lora(model, recipe)
    optim_cfg = recipe["optim"]
    examples = [encode(tok, q, a, recipe["max_seq_len"]) for q, a in pairs]
    bs = optim_cfg["batch_size"]
    steps_per_epoch = math.ceil(len(examples) / bs)
    total_steps = steps_per_epoch * optim_cfg["epochs"]
    warmup = max(1, int(total_steps * optim_cfg["warmup_ratio"]))
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup)
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    )

    rng = random.Random(recipe["seed"])  # data order is derived from the seed
    loss_log, checkpoints, tokens, step = [], [], 0, 0
    t0 = time.monotonic()
    model.train()
    stop = False
    for epoch in range(optim_cfg["epochs"]):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i in range(0, len(order), bs):
            x, y, m = collate(tok, [examples[j] for j in order[i:i + bs]], model.device)
            if tokens + int(m.sum()) > budget["max_train_tokens"]:  # never past what was paid
                log(f"token budget reached at step {step}, stopping")
                stop = True
                break
            loss = model(input_ids=x, attention_mask=m, labels=y).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            tokens += int(m.sum())
            loss_log.append(round(loss.item(), 5))
            if step % recipe["checkpoint_every"] == 0:
                checkpoints.append(lora_state_hash(model))
            if time.monotonic() - t0 > budget["max_wall_seconds"]:
                log(f"wall budget reached at step {step}, stopping")
                stop = True
                break
        if stop:
            break
        log(f"epoch {epoch + 1}/{optim_cfg['epochs']} loss {loss_log[-1]:.4f}")
    checkpoints.append(lora_state_hash(model))
    model.eval()
    return model, {
        "steps": step,
        "train_tokens": tokens,
        "loss_log": loss_log,
        "checkpoint_root": merkle_root(checkpoints),
        "wall_seconds": math.ceil((time.monotonic() - t0) * 10) / 10,  # over the limit stays over
    }


def adapter_bundle(peft_model) -> bytes:
    """Save the adapter to memory-backed /tmp and pack it as a tar archive."""
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        peft_model.save_pretrained(d)
        return tar_bytes({p.name: p.read_bytes() for p in Path(d).iterdir() if p.is_file()})
