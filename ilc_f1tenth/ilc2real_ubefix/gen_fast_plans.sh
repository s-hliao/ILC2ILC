#!/bin/bash
# gen_fast_plans.sh (2026-10-26): harder deployment goals -- the 25-degree drift plans at 1.1x and 1.2x the current
# speed target (trajopt_drift --vscale; tags _v110 / _v120), both tracks, the same homotopy steps; new files only.
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs
PY=~/miniconda3/envs/f1t/bin/python
export OMP_NUM_THREADS=2
for t in mocap_square2fast mocap_figfast; do for v in 110 120; do
  [ -f $F1T_PLANS/${t}_b25_v$v.npz ] || $PY -u trajopt_drift.py $t --steps 0 10 18 25 --beta 25 --vscale ${v:0:1}.${v:1} \
      --tag _v$v --out $F1T_PLANS > logs/to_v${v}_$t.log 2>&1 &
done; done; wait
$PY -u cache_lqr.py > logs/lqr_cache_fast.log 2>&1
ls $F1T_PLANS | grep _v1; echo "$(date +%T) FAST_PLANS_DONE"
