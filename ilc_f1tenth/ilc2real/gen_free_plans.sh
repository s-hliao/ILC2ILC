#!/bin/bash
# gen_free_plans.sh (2026-10-26; user: no sideslip target anywhere -- the track reference is the only goal): the
# sideslip-free plan bank (trajopt_drift --w-beta 0: path + pace only): per track the plan and its conservative
# _mu80 / _mu60 rungs (the hardware stage's fallback ladder), and the held-out mocap_squareH (evaluation only).
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs
PY=~/miniconda3/envs/f1t/bin/python; export OMP_NUM_THREADS=2
for t in mocap_square2fast mocap_figfast mocap_squareH; do
  [ -f $F1T_PLANS/${t}_b0_free.npz ] || $PY -u trajopt_drift.py $t --steps 0 --beta 0 --w-beta 0 --tag _free --out $F1T_PLANS > logs/to_free_$t.log 2>&1 &
  for m in 80 60; do
    [ -f $F1T_PLANS/${t}_b0_free_mu$m.npz ] || $PY -u trajopt_drift.py $t --steps 0 --beta 0 --w-beta 0 --mu-plan 0.$m --tag _free_mu$m \
        --out $F1T_PLANS > logs/to_free_mu${m}_$t.log 2>&1 &
  done
done; wait
$PY -u cache_lqr.py > logs/lqr_cache_free.log 2>&1
ls $F1T_PLANS | grep free; echo "$(date +%T) FREE_PLANS_DONE"
