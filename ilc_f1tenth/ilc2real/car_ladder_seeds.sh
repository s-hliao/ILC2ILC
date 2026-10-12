#!/bin/bash
# Seed spread of the ladder recipe: seed 1 with stronger steps at 16 laps, and a third sim-stage seed (2) with the revert ladder.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
until grep -q LADSTOP_DONE logs/car_ladder_stop.out 2>/dev/null; do sleep 60; done
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6"
LAD="--goals mocap_square2fast_b25 mocap_figfast_b25 --goal-cycle 1 --goal-order worst --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1 --fallback-revert 1"
PL8="$(for t in square2fast figfast; do for b in 0 10 18 25; do echo -n "mocap_${t}_b$b "; done; done)"
PLL="$PL8$(for s in _mu80 _mu60; do for t in square2fast figfast; do for b in 0 10 18 25; do echo -n "mocap_${t}_b$b$s "; done; done; done)"
V3="--anchor-w 1.0 --beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5"
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
arm() { [ -f runs/hw/$1/summary.json ] || F1T_PROCS=2 $PY -u hw_stage.py --policy $2 --out runs/hw/$1 $3 $LAD --cars $CARS --gpu $4 \
  > logs/hw_$1.log 2>&1; }
arm v3drBtL_s1_ladRs_lap16 runs/v3drBtL_s1/policy_final.npz "--iters 8 --cap 0.10 --beta 0.8" 1 &
[ -f runs/v3drBtL_s2/policy_final.npz ] || $PY -u sim_train.py --out runs/v3drBtL_s2 --plans $PLL --dr 1.0 --bc-iters 10 $V3 --iters 30 \
  --seed 2 $TASK --gpu 3 > logs/v3drBtL_s2.log 2>&1
arm v3drBtL_s2_ladR_lap10 runs/v3drBtL_s2/policy_final.npz "--iters 5" 3
wait
F1T_PROCS=4 $PY -u reeval.py v3drBtL_s1_ladRs_lap16 v3drBtL_s2_ladR_lap10 > logs/reeval_ladseeds.log 2>&1
echo "$(date +%T) LADSEEDS_DONE"
