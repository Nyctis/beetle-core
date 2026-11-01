"""The final regression's runner: python run_final.py [-pause SELLER] [+resume SELLER] ITEM ...

Each name is a function in items.py or final_items.py. BEETLE_STOP_AT, a Unix time, is
when to stop starting new items: what was not started is recorded as NOT RUN.
"""

import os
import sys
import time

import final_items
import harness as h
import items

STOP_AT = float(os.environ.get("BEETLE_STOP_AT", 0))

for name in sys.argv[1:]:
    if name[0] in "-+":
        h.current_item = "setup"
        h.say(h.beetle(name[1:], "host pause" if name[0] == "-" else "host resume")[1][-120:])
        h.wait(8)
        continue
    if STOP_AT and time.time() > STOP_AT:
        h.record(name, "NOT RUN", {}, "out of time")
        continue
    try:
        getattr(final_items, name, None) or getattr(items, name)
        (getattr(final_items, name, None) or getattr(items, name))()
    except Exception as e:  # one item's failure is a result, not the end of the run
        h.current_item = name
        h.say(f"harness error: {e!r}")
        h.record(name, "ERROR", {}, repr(e)[:300])
