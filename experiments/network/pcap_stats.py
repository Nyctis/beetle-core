"""On the seller: what a capture and the host's own tap show. Settings on stdin,
one JSON line out."""

from __future__ import annotations

import collections
import json
import statistics
import struct
import subprocess
import sys


def main() -> None:
    s = json.loads(sys.stdin.read())
    subprocess.run(["sudo", "-n", "chmod", "a+r", s["pcap"]], check=False)
    raw = open(s["pcap"], "rb").read()
    out = {"pcap_bytes": len(raw), "plaintext_hits": sum(raw.count(n.encode()) for n in s["needles"])}
    text = subprocess.run(["tcpdump", "-r", s["pcap"], "-nn", "-q", "-tt"], capture_output=True, text=True).stdout
    sizes, seconds = {"in": collections.Counter(), "out": collections.Counter()}, collections.Counter()
    for line in text.splitlines():
        parts = line.split()
        if "length" not in parts:
            continue
        way = "out" if f".{s['port']} >" in line else "in"
        sizes[way][int(parts[parts.index("length") + 1])] += 1
        seconds[int(float(parts[0]))] += 1
    out["packets"] = sum(sum(c.values()) for c in sizes.values())
    out["udp_sizes_top"] = {k: v.most_common(4) for k, v in sizes.items()}
    if seconds:
        span = sorted(seconds)
        counts = [seconds[t] for t in range(span[0], span[-1] + 1)]
        out["packets_per_s"] = {"mean": round(statistics.mean(counts), 1), "stdev": round(statistics.pstdev(counts), 1),
                                "min": min(counts), "max": max(counts), "seconds": len(counts)}
    data = open(s["tap"], "rb").read()
    frames, i, hits = collections.Counter(), 0, 0
    while i + 5 <= len(data):
        n = struct.unpack(">I", data[i + 1:i + 5])[0]
        body = data[i + 5:i + 5 + n]
        frames[n] += 1
        hits += sum(body.count(x.encode()) for x in s["needles"])
        i += 5 + n
    out["tap_frames"] = sum(frames.values())
    out["tap_frame_sizes"] = sorted(frames)
    out["tap_plaintext_hits"] = hits
    print(json.dumps(out))


main()
