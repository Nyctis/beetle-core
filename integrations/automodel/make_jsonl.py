"""Turn the trainer's pairs into the `messages` JSONL Automodel's ChatDataset reads.

    python make_jsonl.py work/train.json work/train.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    pairs = json.loads(Path(sys.argv[1]).read_text())
    rows = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a in pairs]
    Path(sys.argv[2]).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} conversations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
