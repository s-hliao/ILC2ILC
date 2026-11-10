#!/bin/bash
# fadaT_train_all.sh (2026-11-07, FADA_AUDIT_1107.md FULL FIDELITY): fada_car_tf.py for seeds 0-2 (oracle = figure 1's RMA teacher rma_s<s>)
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p logs/fadaT
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$(( (s + 1) % 4 )) taskset -c $((20 + 2 * s))-$((21 + 2 * s)) $HOME/miniconda3/envs/f1t/bin/python -u fada_car_tf.py \
    --oracle runs/fig1/rma_s$s --out runs/fig1/fadaT_s$s --seed $s > logs/fadaT/s$s.log 2>&1 &
done
wait
touch logs/fadaT/DONE
