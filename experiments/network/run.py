"""Run matrix items by name: python experiments/network/run.py C1 C2 ..."""

import sys

import items
import harness as h

for name in sys.argv[1:]:
    try:
        getattr(items, name)()
    except Exception as e:  # one item's failure is a result, not the end of the run
        h.current_item = name
        h.say(f"harness error: {e!r}")
        h.record(name, "ERROR", {}, repr(e)[:300])
