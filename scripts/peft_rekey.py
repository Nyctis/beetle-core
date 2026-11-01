"""Rewrite a PEFT adapter's keys for the model class vLLM serves.

The rule comes from the model's profile, see common/model_profiles.py.

    python scripts/peft_rekey.py --model NAME --adapter DIR --out DIR
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
    print(mp.to_vllm(args.adapter, args.out, mp.get(args.model)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
