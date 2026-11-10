#!/bin/bash
# The ladder's seed-1 failure (v3drBtL_s1_lad1_lap10: 65.6 % b25 on 8 cars): figfast gets 2-3 trials in 10 laps and spends them
# descending the ladder (crash-rolled-back), nothing is trained on the rung it lands on. Fix tested: --goal-order worst (per car,
# the goal doing worst flies next; a goal that just switched rung counts as untried), at 10 and 16 laps.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6"
LAD="--goals mocap_square2fast_b25 mocap_figfast_b25 --goal-cycle 1 --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1"
arm() { [ -f runs/hw/$1/summary.json ] || F1T_PROCS=2 $PY -u hw_stage.py --policy runs/$2/policy_final.npz --out runs/hw/$1 $3 $LAD \
  --cars $CARS --gpu $4 > logs/hw_$1.log 2>&1; }
arm v3drBtL_s1_ladW_lap10 v3drBtL_s1 "--iters 5 --goal-order worst" 1 & arm v3drBtL_s1_ladW_lap16 v3drBtL_s1 "--iters 8 --goal-order worst" 2 & wait
F1T_PROCS=4 $PY -u reeval.py v3drBtL_s1_ladW_lap10 v3drBtL_s1_ladW_lap16 > logs/reeval_ladW_s1.log 2>&1
echo "$(date +%T) LADW_S1_DONE"
