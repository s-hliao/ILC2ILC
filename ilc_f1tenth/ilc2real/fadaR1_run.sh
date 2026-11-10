#!/bin/bash
# fadaR1_run.sh (2026-11-07 ~12:50, Lambda): fadaR with --hist 1 (memoryless FADA, as every other arm), seeds 0-2, then fig1 it 40 75 150.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
PY=$HOME/miniconda3/envs/f1t/bin/python
mkdir -p logs/fadaR1
for s in 0 1 2; do
  [ -f runs/fig1/fadaR1_s$s/it150.pkl ] || CUDA_VISIBLE_DEVICES=$((s % 2)) $PY -u fada_car_tf.py --data rma --hist 1 --iters 150 --steps 200 \
     --save-at 40 75 150 --oracle runs/fig1/rma_s$s --out runs/fig1/fadaR1_s$s --seed $s > logs/fadaR1/s$s.log 2>&1 &
done
wait
until [ -f logs/fadaR/FIG1_DONE ]; do sleep 60; done
L=""; for s in 0 1 2; do for k in 150 40 75; do f=runs/fig1/fadaR1_s$s/it${k}_policy.npz
  $PY -c "import numpy as np, os; np.savez('$f', mode='fada_tf', pkl=os.path.abspath('runs/fig1/fadaR1_s$s/it$k.pkl'))"; L="$L $f"; done; done
F1T_SPIN_RHO=3 F1T_PROCS=20 FADA_JIT=1 nice -n 5 $PY -u fig1_eval.py runs/fig1/fig1_eval_fadaR1.json $L > logs/fadaR1/fig1_eval.log 2>&1
touch logs/fadaR1/DONE
