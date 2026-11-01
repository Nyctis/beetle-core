"""Train the memory recipe's LoRA on train.json, with the sandbox's own trainer.

Runs inside the sandbox image, offline, so the adapter is what a seller's sandbox
would make from the same pairs:

    docker run --rm --gpus all --network none \
        -v MODEL:/model:ro -v WORK:/work --entrypoint python \
        beetle-sandbox:dev /work/train_adapter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/app")
import train_eval as te  # noqa: E402

RECIPE = {
    "seed": 7, "max_seq_len": 256, "checkpoint_every": 100,
    "lora": {"rank": 16, "alpha": 32, "dropout": 0.0,
             "targets": r".*layers\.\d+\.(mlp\.(gate_proj|up_proj|down_proj)|linear_attn\.out_proj|self_attn\.o_proj)"},
    "optim": {"lr": 5e-4, "epochs": 8, "batch_size": 8, "warmup_ratio": 0.05, "weight_decay": 0.0},
}
BUDGET = {"max_train_tokens": 10**9, "max_wall_seconds": 1800}


def main() -> int:
    work = Path("/work")
    pairs = [tuple(p) for p in json.loads((work / "train.json").read_text())]
    tok, model = te.load_base(Path("/model"))
    model, stats = te.train_lora(model, tok, pairs, RECIPE, BUDGET, lambda m: print(m, flush=True))
    model.save_pretrained(work / "adapter_peft")
    held = json.loads((work / "targets.json").read_text())["targeted"][:6]
    for item, text in zip(held, te.generate(model, tok, [h["q"] for h in held], 32, 256)):
        print(f"check: {item['q']!r} -> {text!r}", flush=True)
    stats.pop("loss_log")
    print(json.dumps({**stats, "checkpoint_root": stats["checkpoint_root"].hex()
                      if isinstance(stats["checkpoint_root"], bytes) else stats["checkpoint_root"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
