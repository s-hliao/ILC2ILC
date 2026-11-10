#!/bin/bash
# dagger_run.sh (2026-11-07 ~11:58, Lambda): dagger_car.py seeds 0-2 (GPU s % 2), then fig1_eval.py on it 1 2 4 8 -> runs/fig1/fig1_eval_dagger.json
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/miniconda3/envs/f1t/bin/python
mkdir -p logs/dagger
for s in 0 1 2; do
  [ -f runs/fig1/dagger_s$s/it8_policy.npz ] || CUDA_VISIBLE_DEVICES=$((s % 2)) taskset -c 46-47 $PY -u dagger_car.py --oracle runs/fig1/rma_s$s \
     --out runs/fig1/dagger_s$s --seed $s > logs/dagger/s$s.log 2>&1 &
done
wait
until [ -f logs/ceil2/DONE ]; do sleep 60; done
L=""; for s in 0 1 2; do for k in 1 2 4 8; do L="$L runs/fig1/dagger_s$s/it${k}_policy.npz"; done; done
F1T_SPIN_RHO=3 F1T_PROCS=40 $PY -u fig1_eval.py runs/fig1/fig1_eval_dagger.json $L > logs/dagger/fig1_eval.log 2>&1
touch logs/dagger/DONE
