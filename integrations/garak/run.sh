#!/bin/bash
# One garak pass against one model name. Run from the directory that holds
# targets.json and beetle_memory.py, with the garak venv active:
#   URI=http://127.0.0.1:8100/v1/ ./run.sh NAME OUTPUT_PREFIX [PROBES]
# Everything stays local: the only network target is URI, and telemetry is off.
set -euo pipefail
name=$1 prefix=$2
probes=${3:-beetle_memory.Targeted,beetle_memory.Prefix,beetle_memory.Blind,beetle_memory.Encoded}
here=$(pwd)
pkg=$(python -c 'import garak, pathlib; print(pathlib.Path(garak.__file__).parent)')
ln -sf "$here/beetle_memory.py" "$pkg/probes/beetle_memory.py"
ln -sf "$here/beetle_memory.py" "$pkg/detectors/beetle_memory.py"
export BEETLE_GARAK_TARGETS="$here/targets.json" OPENAICOMPATIBLE_API_KEY=${OPENAICOMPATIBLE_API_KEY:-EMPTY}
export DO_NOT_TRACK=1 NEMO_TELEMETRY_ENABLED=false HF_HUB_OFFLINE=1 XDG_DATA_HOME="$here/garak-data" XDG_CACHE_HOME="$here/garak-cache"
python - "$prefix" <<PY
import json, sys
json.dump({"openai": {"OpenAICompatible": {"uri": "${URI:-http://127.0.0.1:8100/v1/}", "max_tokens": 96, "temperature": 0.7,
                                           "top_p": 1.0, "stop": [], "seed": 7}}}, open(sys.argv[1] + ".generator.json", "w"))
PY
garak --target_type openai.OpenAICompatible --target_name "$name" \
    --generator_option_file "$prefix.generator.json" --probes "$probes" \
    --generations "${GENERATIONS:-5}" --parallel_attempts 8 --seed 7 --report_prefix "$prefix"
