#!/bin/bash
# orca_practised.sh (2026-11-05): experiment 1 of runs/screen/ORCA_RULES.md: 6 stage runs practising orca43 (PPO+DR / RMA, network
# seeds 0-2, stage seed 9001) on GPUs 0-3, then orca_eval on them. (Experiment 2, the held-out evaluation, runs separately.)
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
PY=~/miniconda3/envs/f1t/bin/python
g=0; for n in freePPON rmaN; do for s in 0 1 2; do
  bash orca_stage.sh $n $s 9001 $((g % 4)) & g=$((g + 1)); sleep 20
done; done
wait
F1T_PROCS=16 $PY -u orca_eval.py runs/screen/ORCA_*_ss* > logs/orca_eval_practised.log 2>&1
echo ALL_DONE
