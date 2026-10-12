#!/bin/bash
# Seed robustness of the ladder recipe. (1) The car's sim-only stopping rule (stop_pick.py: the sim-stage snapshot with the best
# held-out success over the non-nominal SIM conditions; no real data) applied to v3drBtL_s0 / s1, then the revert ladder
# (10 laps) on each pick. (2) Seed 1's near-misses (e_y 10-11 cm) with stronger hardware-stage steps (cap 0.10, beta 0.8).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6"
LAD="--goals mocap_square2fast_b25 mocap_figfast_b25 --goal-cycle 1 --goal-order worst --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1 --fallback-revert 1"
arm() { [ -f runs/hw/$1/summary.json ] || F1T_PROCS=2 $PY -u hw_stage.py --policy $2 --out runs/hw/$1 $3 $LAD --cars $CARS --gpu $4 \
  > logs/hw_$1.log 2>&1; }
arm v3drBtL_s1_ladRs_lap10 runs/v3drBtL_s1/policy_final.npz "--iters 5 --cap 0.10 --beta 0.8" 1 &
mkdir -p runs/stopping
[ -f runs/stopping/sim_val_L.json ] || $PY -u sim_validate.py --runs runs/v3drBtL_s0 runs/v3drBtL_s1 --out runs/stopping/sim_val_L.json \
  --gpu 2 > logs/stop_simval_L.log 2>&1
for s in 0 1; do $PY stop_pick.py runs/v3drBtL_s$s runs/stopping/sim_val_L.json runs/v3drBtL_s${s}_stop; done
arm v3drBtL_s0stop_ladR_lap10 runs/v3drBtL_s0_stop/policy_final.npz "--iters 5" 2
wait
arm v3drBtL_s1stop_ladR_lap10 runs/v3drBtL_s1_stop/policy_final.npz "--iters 5" 2
F1T_PROCS=4 $PY -u reeval.py v3drBtL_s1_ladRs_lap10 v3drBtL_s0stop_ladR_lap10 v3drBtL_s1stop_ladR_lap10 > logs/reeval_ladstop.log 2>&1
echo "$(date +%T) LADSTOP_DONE"
