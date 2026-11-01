#!/bin/bash
# Run a command on this node detached from ssh: bg.sh NAME CMD...  Log: ~/beetle-exp/logs/NAME.log
name=$1; shift
mkdir -p ~/beetle-exp/logs
rm -f ~/beetle-exp/logs/$name.done
export PYTORCH_ALLOC_CONF=expandable_segments:True HF_HOME=~/beetle-exp/hf PYTHONPATH=~/beetle-exp/deps HF_HUB_DISABLE_TELEMETRY=1 NEMO_TELEMETRY_ENABLED=false DO_NOT_TRACK=1
source ~/venv/bin/activate
nohup setsid bash -c "$* ; echo EXIT=\$? >> ~/beetle-exp/logs/$name.log; touch ~/beetle-exp/logs/$name.done" > ~/beetle-exp/logs/$name.log 2>&1 &
