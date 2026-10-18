#!/bin/bash
# car_trd.sh (ube, 2026-10-17): the trust region ON TOP of the recipe's damping (beta 0.5, delta 0.05, anchor 1, the
# fixed refit) -- only the radius adapts (starting at the recipe's cap 0.05, or 0.20) instead of the fixed cap +
# progress rollback. Same ladder, cars and seeds as car_tr.sh.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6"
LAD="--iters 5 --goals mocap_square2fast_b25 mocap_figfast_b25 --goal-cycle 1 --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1 --fallback-revert 1"
declare -A ARM=([lad_trd]="--tr 1 --cap 0.05" [lad_trd_r20]="--tr 1 --cap 0.20")
n=0; g=1; NPAR=${NPAR:-2}
for s in 0 1 2; do for arm in lad_trd lad_trd_r20; do
  out=runs/tr/${arm}_s$s; [ -f $out/summary.json ] && continue
  F1T_PROCS=${PROCS:-3} $PY -u hw_stage.py --policy runs/v3drBtL_s$s/policy_final.npz --out $out $LAD ${ARM[$arm]} --cars $CARS \
    --gpu $g > logs/tr_${arm}_s$s.log 2>&1 &
  g=$(( g % 2 + 1 )); n=$((n + 1)); if (( n % NPAR == 0 )); then wait -n; n=$((n - 1)); fi
done; done
wait; echo "$(date +%T) CAR_TRD_DONE"
