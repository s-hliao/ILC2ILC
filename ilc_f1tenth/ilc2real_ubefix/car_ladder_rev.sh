#!/bin/bash
# The ladder with --fallback-revert 1 (a rung doing worse than the one above sends the goal back up, locked) + --goal-order worst:
# seed 1 (where the one-way ladder over-descended: lag figfast -> _mu60 crashes, mass square -> _mu60) at 10 / 16 laps, then
# seed 0 at 10 laps (no regression).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6"
LAD="--goals mocap_square2fast_b25 mocap_figfast_b25 --goal-cycle 1 --goal-order worst --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1 --fallback-revert 1"
arm() { [ -f runs/hw/$1/summary.json ] || F1T_PROCS=2 $PY -u hw_stage.py --policy runs/$2/policy_final.npz --out runs/hw/$1 $3 $LAD \
  --cars $CARS --gpu $4 > logs/hw_$1.log 2>&1; }
arm v3drBtL_s1_ladR_lap10 v3drBtL_s1 "--iters 5" 1 & arm v3drBtL_s1_ladR_lap16 v3drBtL_s1 "--iters 8" 2 & wait
F1T_PROCS=4 $PY -u reeval.py v3drBtL_s1_ladR_lap10 v3drBtL_s1_ladR_lap16 > logs/reeval_ladR_s1.log 2>&1
echo "$(date +%T) LADR_S1_DONE"
arm v3drBtL_s0_ladR_lap10 v3drBtL_s0 "--iters 5" 1
F1T_PROCS=4 $PY -u reeval.py v3drBtL_s0_ladR_lap10 > logs/reeval_ladR_s0.log 2>&1
echo "$(date +%T) LADR_DONE"
