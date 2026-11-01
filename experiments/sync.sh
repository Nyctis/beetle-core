#!/bin/sh
# Copy the experiment code and a snapshot of the sandbox trainer to the GPU node.
set -e
HOST=${HOST:?set HOST to the GPU node}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ssh -q "$HOST" 'mkdir -p ~/beetle-exp/code ~/beetle-exp/snapshot/sandbox ~/beetle-exp/snapshot/common'
# Data and results are produced on the node and only ever flow back: never push them.
rsync -az --exclude __pycache__ --exclude 'results/' --exclude 'data/' --exclude regression.json --exclude replay.json \
  "$ROOT/experiments/" "$HOST":beetle-exp/code/
rsync -az "$ROOT/sandbox/train_eval.py" "$ROOT/sandbox/recipe_data.py" "$HOST":beetle-exp/snapshot/sandbox/
rsync -az "$ROOT/common/beetle_archive.py" "$HOST":beetle-exp/snapshot/common/
