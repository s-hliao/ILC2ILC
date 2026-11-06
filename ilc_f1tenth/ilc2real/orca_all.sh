#!/bin/bash
# orca_all.sh (2026-11-05): both ORCA experiments (runs/screen/ORCA_RULES.md). 1: practise orca43, 6 stage runs (PPO+DR / RMA, network
# seeds 0-2, stage seed 9001) on GPUs 0-3, then orca_eval on them. 2: orca_eval (held-out orca43 + peanut) on the 18 existing plastic
# square2fast stage runs (PLSL*). Load: 6 x 4 + 8 workers <= 34 of matcha's 36 CPUs.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
PY=~/miniconda3/envs/f1t/bin/python
( F1T_PROCS=8 $PY -u orca_eval.py runs/screen/PLSL* > logs/orca_eval_heldout.log 2>&1; echo HELDOUT_DONE >> logs/orca_eval_heldout.log ) &
g=0; for n in freePPON rmaN; do for s in 0 1 2; do
  bash orca_stage.sh $n $s 9001 $((g % 4)) & g=$((g + 1)); sleep 20
done; done
wait
F1T_PROCS=16 $PY -u orca_eval.py runs/screen/ORCA_* > logs/orca_eval_practised.log 2>&1
echo ALL_DONE
