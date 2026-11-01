#!/usr/bin/env bash
# Run iroh_pair.py between a buyer and a seller and print both results. A peer
# is an ssh host name, or "local" for this machine. Each peer needs the beetle
# code in ~/beetle and iroh in ~/irohenv (this Mac: IROH_PY).
#
#   scripts/iroh_pair.sh local GPU_NODE
set -uo pipefail
A=$1
B=$2
PORT=${PORT:-41710}
LOCAL_PY=${IROH_PY:-python3}
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN=$(mktemp -d)
ID="iroh-$$"

py() { if [ "$1" = local ]; then echo "$LOCAL_PY $HERE/iroh_pair.py"; else echo "~/irohenv/bin/python ~/beetle/scripts/iroh_pair.py"; fi; }
run() { if [ "$1" = local ]; then shift; bash -c "$*"; else h=$1; shift; ssh -o BatchMode=yes "$h" "$*" 2>/dev/null; fi; }
put() { if [ "$1" = local ]; then cp "$2" "$3"; else scp -q "$2" "$1:$3" 2>/dev/null; fi; }
get() { if [ "$1" = local ]; then cp "$2" "$3" 2>/dev/null; else scp -q "$1:$2" "$3" 2>/dev/null; fi; }

for p in "$A" "$B"; do run "$p" "rm -f /tmp/$ID.*"; done
run "$A" "$(py "$A") --role buyer --port $PORT --out /tmp/$ID.me --peer-file /tmp/$ID.peer" >"$RUN/a.out" &
PA=$!
run "$B" "$(py "$B") --role seller --port $PORT --out /tmp/$ID.me --peer-file /tmp/$ID.peer" >"$RUN/b.out" &
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
