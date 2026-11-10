#!/bin/bash
# daggerR_run.sh (2026-11-07 ~13:55, Lambda): dagger_car.py --data rma (fadaR1's data), obs only (daggerR) and [obs, a_{t-1}] (daggerRp), seeds 0-2,
# 150 iterations x 200 steps; then fig1_eval it150 (+ it40 / it75) -> runs/fig1/fig1_eval_daggerR.json.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
PY=$HOME/miniconda3/envs/f1t/bin/python
mkdir -p logs/daggerR
for v in 0 1; do n=daggerR; [ $v = 1 ] && n=daggerRp; for s in 0 1 2; do
  [ -f runs/fig1/${n}_s$s/it150_policy.npz ] || CUDA_VISIBLE_DEVICES=$(( (s + v) % 2 )) $PY -u dagger_car.py --data rma --prev-a $v --iters 150 --steps 200 \
     --oracle runs/fig1/rma_s$s --out runs/fig1/${n}_s$s --seed $s > logs/daggerR/${n}_s$s.log 2>&1 &
done; done
wait
L=""; for n in daggerRp daggerR; do for s in 0 1 2; do for k in 150 40 75; do L="$L runs/fig1/${n}_s$s/it${k}_policy.npz"; done; done; done
F1T_SPIN_RHO=3 F1T_PROCS=24 nice -n 5 $PY -u fig1_eval.py runs/fig1/fig1_eval_daggerR.json $L > logs/daggerR/fig1_eval.log 2>&1
touch logs/daggerR/DONE
