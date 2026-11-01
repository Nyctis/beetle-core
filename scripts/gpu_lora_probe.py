"""Train the memory recipe on a GPU outside the sandbox and save the adapter.

Checks three things before a base model goes into the sealed path: it loads
with the sandbox's own code, the recipe reaches the hidden-phrasing bar, and
the saved adapter can be handed to a server such as vLLM. Scores are printed,
so run it on public or demo data only.

Usage (on a GPU machine, from the beetle directory):
    python scripts/gpu_lora_probe.py --model .cache/models/NAME --out .cache/adapters/demo \
        [--targets REGEX] [--rank 16] [--lr 2e-4] [--epochs 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "common"), str(ROOT / "sandbox")]

import torch  # noqa: E402
import train_eval as te  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", default="all-linear")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=256)
    args = ap.parse_args()

    memories = json.loads((ROOT / "demo/memories.json").read_text())
    eval_items = json.loads((ROOT / "demo/eval_hidden.json").read_text())
    recipe = {
        "lora": {"rank": args.rank, "alpha": args.rank * 2, "dropout": 0.0, "targets": args.targets},
        "optim": {"lr": args.lr, "warmup_ratio": 0.05, "epochs": args.epochs,
                  "batch_size": args.batch_size, "weight_decay": 0.0},
        "max_seq_len": args.max_seq_len, "seed": 42, "checkpoint_every": 20,
    }

    t0 = time.time()
    tok, model = te.load_base(args.model)
    print(f"loaded {type(model).__name__} on {model.device} in {time.time() - t0:.1f}s")
    base_acc = te.qa_accuracy(model, tok, eval_items)
    print(f"hidden-phrasing accuracy before training: {base_acc:.2f}")

    model, stats = te.train_lora(model, tok, te.augment(memories), recipe,
                                 {"max_train_tokens": 10_000_000, "max_wall_seconds": 3600}, print)
    model.print_trainable_parameters()
    acc = te.qa_accuracy(model, tok, eval_items)
    print(f"hidden-phrasing accuracy after training: {acc:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    sizes = {p.name: p.stat().st_size for p in args.out.iterdir() if p.is_file()}
    result = {
        "model": type(model.get_base_model()).__name__,
        "accuracy": {"before": base_acc, "after": acc},
        "steps": stats["steps"], "wall_seconds": stats["wall_seconds"],
        "final_loss": stats["loss_log"][-1],
        "adapter_bytes": sum(sizes.values()),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)
        if torch.cuda.is_available() else 0,
    }
    (args.out / "probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
