"""Convert a PEFT LoRA adapter to the layout mlx-lm loads.

A remote GPU trains with PEFT; a Mac then runs the same adapter on an MLX base.
The key mapping comes from the model's profile, see common/model_profiles.py.

    python scripts/peft_to_mlx.py --model NAME --adapter PEFT_DIR --out MLX_DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

import model_profiles as mp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="profile name or repo id")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(mp.to_mlx(args.adapter, args.out, mp.get(args.model)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
