#!/bin/bash
# Reproduce every experiment: HOST=<gpu-node> ./run_all.sh
# Run from a laptop; the work happens on the GPU node, which needs internet access
# to Hugging Face and PyPI. Generated data and results are not in the repository.
# Data generation is seeded and caches Bolt's replies under */data, so a rerun with
# those caches gives the same tests. Delete the caches to regenerate from vLLM.
# Requires: `llm` vLLM container on the node, ~/venv with torch/transformers/peft,
# ~/beetle-exp/deps (pyarrow rank_bm25) and ~/beetle-exp/tools/venv (data-designer==0.9.7).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
"$HERE/sync.sh"
ssh -q "${HOST:?set HOST to the GPU node}" bash -s <<'REMOTE'
set -euo pipefail
source ~/venv/bin/activate
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HOME=~/beetle-exp/hf PYTHONPATH=~/beetle-exp/deps HF_HUB_DISABLE_TELEMETRY=1
export NEMO_TELEMETRY_ENABLED=false DO_NOT_TRACK=1
cd ~/beetle-exp/code
python make_regression.py
# UC1
(cd uc1_personal
 python gen_data.py
 python run.py --users dev00,dev01 --seeds "" --variant R1 --tag pilot_base --stage pilot --extra-prompts P2,P3
 python run.py --users dev00,dev01 --seeds 1 --variant R1 --tag pilot_R1 --stage pilot --rag-prompt P3
 python run.py --users dev00,dev01 --seeds 1 --variant R2 --tag pilot_R2 --stage pilot --rag-prompt P3 --skip-baselines
 python run.py --users final02,final03,final04,final05,final06,final07,final08,final09 --variant R1 --seeds 1,2,3 --tag final --rag-prompt P3)
# UC2
(cd uc2_lab
 for w in dev final; do
   python gen_data.py records --world $w
   ~/beetle-exp/tools/venv/bin/python notes_dd.py $w
   python gen_data.py finish --world $w
 done
 python run.py --world dev --seeds 1 --tag devcheck --stage pilot --no-placebo --epochs 1
 python run.py --world final --seeds 1,2 --tag final
 # Exploratory E1, not in any verdict.
 python run.py --world final --seeds 1 --tag explore_diag --stage explore --no-placebo --skip-base \
   --kinds diagnose --only-tasks T2 --no-bucket)
# UC3
(cd uc3_documents
 python gen_data.py --corpus dev
 python gen_data.py --corpus final
 python run.py --corpus dev --plans raw:mix --tag devcheck --stage pilot --epochs 1
 python run.py --corpus final --plans raw:mix,rule:mix,gold:mix --tag final)
# Exploratory E2 and E3, not in any verdict.
(cd uc1_personal
 python explore_thinking.py final03,final05,final07,final09
 python run.py --users final03,final05,final07 --variant R3 --seeds 1 --tag explore_R3 --stage explore --rag-prompt P3 --skip-baselines)
python analyze.py uc1 final
python analyze.py uc2 final
python analyze.py uc3 final
python analyze.py cost
REMOTE
