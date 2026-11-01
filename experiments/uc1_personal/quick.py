"""Print per-arm task accuracy for one results tag (pilot inspection)."""
import collections
import json
import sys
from pathlib import Path

rows = [json.loads(l) for t in sys.argv[1:] for l in open(Path(__file__).parent / f"results/preds_{t}.jsonl")]
agg = collections.defaultdict(lambda: collections.defaultdict(list))
ptok = collections.defaultdict(list)
for r in rows:
    agg[r["arm"]][r["task"]].append(r["score"])
    ptok[r["arm"]].append(r["ptok"])
for a, v in agg.items():
    m = {t: sum(x) / len(x) for t, x in sorted(v.items())}
    print(f"{a:12s}", {k: round(x, 2) for k, x in m.items()}, "macro", round(sum(m.values()) / 4, 3),
          "n", sum(len(x) for x in v.values()), "ptok", sum(ptok[a]) // len(ptok[a]))
