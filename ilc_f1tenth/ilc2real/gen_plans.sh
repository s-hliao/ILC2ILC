#!/bin/bash
# gen_plans.sh (ube, 2026-10-17): regenerate the mu 0.2 tight-track plan bank (plans_mu02_tight is not in git):
# the drift homotopy (scheduler.py), the conservative _mu80 variants (scheduler.py) and _mu60 variants (by analogy:
# --mu-plan 0.6 --tag _mu60), then the LQR gain cache.
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p "$F1T_PLANS" logs
PY=~/miniconda3/envs/f1t/bin/python
for t in mocap_square2fast mocap_figfast; do
  [ -f $F1T_PLANS/${t}_b21.npz ] || $PY -u trajopt_drift.py $t --steps 0 10 14 18 21 --beta 25 --out $F1T_PLANS > logs/to_$t.log 2>&1 &
done; wait
for t in mocap_square2fast mocap_figfast; do
  for m in 80 60; do
    [ -f $F1T_PLANS/${t}_b25_mu$m.npz ] || $PY -u trajopt_drift.py $t --steps 0 10 18 --beta 25 --mu-plan 0.$m --tag _mu$m \
      --out $F1T_PLANS > logs/to_mu${m}_$t.log 2>&1 &
  done
done; wait
$PY -u cache_lqr.py > logs/lqr_cache.log 2>&1
ls $F1T_PLANS; echo "$(date +%T) PLANS_DONE"
