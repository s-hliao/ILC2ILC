#!/bin/bash
# car_tr.sh (ube, 2026-10-17): the car's 10-lap ladder hardware stage with ONE adaptive step rule (hw_stage --tr:
# Levenberg-Marquardt trust region) vs the hand-set recipe (beta 0.5, delta 0.05, cap 0.05, anchor 1, progress rollback).
# Plans regenerated on ube (gen_plans.sh). 8 cars, the 3 ladder sim seeds (v3drBtL_s0..2), hw_stage's own evaluation.
#   lad        the headline recipe (car_ladder.sh + revert): --iters 5 --goal-cycle 1 ladder, fixed step knobs
#   lad_tr     the same ladder, --tr 1 --beta 1.0 --anchor-w 0, refit the newest trials to convergence (age decay 0)
#   lad_tr_r20 lad_tr with the radius starting at 0.20 instead of 0.05
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6"
LAD="--iters 5 --goals mocap_square2fast_b25 mocap_figfast_b25 --goal-cycle 1 --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1 --fallback-revert 1"
TR="--tr 1 --beta 1.0 --anchor-w 0 --fit-realize 0.95 --age-decay 0"
declare -A ARM=([lad]="" [lad_tr]="$TR --cap 0.05" [lad_tr_r20]="$TR --cap 0.20")
n=0; g=1; NPAR=${NPAR:-3}
for s in 0 1 2; do
  for arm in lad lad_tr lad_tr_r20; do
    out=runs/tr/${arm}_s$s
    [ -f $out/summary.json ] && continue
    F1T_PROCS=${PROCS:-4} $PY -u hw_stage.py --policy runs/v3drBtL_s$s/policy_final.npz --out $out $LAD ${ARM[$arm]} --cars $CARS \
      --gpu $g > logs/tr_${arm}_s$s.log 2>&1 &
    g=$(( g % 2 + 1 )); n=$((n + 1))
    if (( n % NPAR == 0 )); then wait -n; n=$((n - 1)); fi
  done
done
wait
echo "$(date +%T) CAR_TR_DONE"
