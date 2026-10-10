#!/bin/bash
# restart the tight-track lap-budget queue once the main matrix is done (paused for the CPU budget)
cd "$(dirname "$0")"
until [ -f logs/final.done ]; do sleep 60; done   # (2-lap rerun)
source env_tight.sh
nohup ~/miniconda3/envs/f1t/bin/python -u scheduler_laps.py >> logs/scheduler_laps.out 2>&1 &
echo "$(date +%T) laps queue resumed"
