#!/usr/bin/env bash
# Run udp_punch.py between two peers and print both results. A peer is an ssh
# host name, or "local" for this machine. The script swaps the published
# mappings by copying files, which is the step the chain will do.
#
#   scripts/punch_pair.sh local GPU_NODE
set -uo pipefail
A=$1
B=$2
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT=${PORT:-41700}  # not 41641, which Tailscale holds on machines running it
RUN=$(mktemp -d)
ID="punch-$$"

run() { if [ "$1" = local ]; then shift; bash -c "$*"; else h=$1; shift; ssh -o BatchMode=yes "$h" "$*" 2>/dev/null; fi; }
put() { if [ "$1" = local ]; then cp "$2" "$3"; else scp -q "$2" "$1:$3" 2>/dev/null; fi; }
get() { if [ "$1" = local ]; then cp "$2" "$3" 2>/dev/null; else scp -q "$1:$2" "$3" 2>/dev/null; fi; }

for p in "$A" "$B"; do put "$p" "$HERE/udp_punch.py" "/tmp/udp_punch.py"; run "$p" "rm -f /tmp/$ID.*"; done
run "$A" "python3 /tmp/udp_punch.py --port $PORT --out /tmp/$ID.me --peer-file /tmp/$ID.peer" >"$RUN/a.out" &
PA=$!
run "$B" "python3 /tmp/udp_punch.py --port $PORT --out /tmp/$ID.me --peer-file /tmp/$ID.peer" >"$RUN/b.out" &
PB=$!

for _ in $(seq 1 40); do
  get "$A" "/tmp/$ID.me" "$RUN/a.me"; get "$B" "/tmp/$ID.me" "$RUN/b.me"
  [ -s "$RUN/a.me" ] && [ -s "$RUN/b.me" ] && break
  sleep 1
done
put "$A" "$RUN/b.me" "/tmp/$ID.peer"
put "$B" "$RUN/a.me" "/tmp/$ID.peer"
wait $PA; wait $PB
echo "$A: $(cat "$RUN/a.out")"
echo "$B: $(cat "$RUN/b.out")"
rm -rf "$RUN"
