#!/bin/bash
# v6 (14-plan bank incl. conservative _mu80 goals): zero-shot, +24 laps, +24 laps with the goal fallback
cd "$(dirname "$0")"
PY=~/miniconda3/envs/f1t/bin/python
for s in 0 1; do
  until [ -f runs/v6nom_s$s/policy_final.npz ]; do sleep 30; done
  P=runs/v6nom_s$s/policy_final.npz
  F1T_PROCS=8 $PY -u hw_stage.py --policy $P --out runs/hw/v6nom_s${s}_zs --iters 0 --gpu 0 > logs/hw_v6nom_s${s}_zs.log 2>&1 &
  F1T_PROCS=8 $PY -u hw_stage.py --policy $P --out runs/hw/v6nom_s${s}_hw24 --iters 4 --gpu 1 > logs/hw_v6nom_s${s}_hw24.log 2>&1 &
  F1T_PROCS=8 $PY -u hw_stage.py --policy $P --out runs/hw/v6nom_s${s}_hw24fb --iters 4 --fallback-suffix _mu80 --gpu 2 > logs/hw_v6nom_s${s}_hw24fb.log 2>&1 &
  wait
done
F1T_PROCS=20 $PY -u reeval.py v6nom_s0_zs v6nom_s0_hw24 v6nom_s0_hw24fb v6nom_s1_zs v6nom_s1_hw24 v6nom_s1_hw24fb > logs/reeval5.log 2>&1
echo "$(date +%H:%M) Q5_DONE"
