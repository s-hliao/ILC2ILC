#!/bin/bash
# fada_planner_all.sh (2026-11-07, runs/screen/FADA_AUDIT_1107.md): FADA's learned planner for every figure-1 car IDM (data 5-40; 60 done),
# 4 at a time on matcha's GPUs, then idm_planner.npz = the IDM + its planner (the policy file reeval / fig1_eval read).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real && source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=~/miniconda3/envs/f1t/bin/python; mkdir -p logs/fada_planner
i=0
for n in 5 10 20 40; do for s in 0 1 2; do
  d=runs/fig1/fada_d${n}_s$s; [ -f $d/planner.npz ] && continue
  CUDA_VISIBLE_DEVICES=$((i % 4)) $PY -u fada_car_planner.py --idm $d/idm.npz --out $d/planner.npz > logs/fada_planner/d${n}_s$s.log 2>&1 &
  i=$((i + 1)); [ $((i % 4)) = 0 ] && wait
done; done
wait
for d in runs/fig1/fada_d*_s?; do $PY -c "
import numpy as np, sys
a = dict(np.load('$d/idm.npz')); p = dict(np.load('$d/planner.npz')); a.update(p); a['horizon'] = np.array(4); a['mode'] = np.array('fada')
np.savez('$d/idm_planner.npz', **a)"; done
ls runs/fig1/fada_d*_s?/idm_planner.npz | wc -l > logs/fada_planner/ALL_DONE
