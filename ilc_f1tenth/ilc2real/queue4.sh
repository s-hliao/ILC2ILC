#!/bin/bash
# constant trust region for long budgets (limitation 2), and a third seed of the best arm
cd "$(dirname "$0")"
PY=~/miniconda3/envs/f1t/bin/python
F1T_PROCS=8 $PY -u hw_stage.py --policy runs/v3nom_s0/policy_final.npz --out runs/hw/v3nom_s0_hw96_cTR --iters 16 --anchor-n0 1e9 --gpu 0 > logs/hw_v3nom_s0_hw96_cTR.log 2>&1 &
F1T_PROCS=8 $PY -u hw_stage.py --policy runs/ppo_dr_s0/policy_final.npz --out runs/hw/ppo_dr_hw96_cTR --iters 16 --anchor-n0 1e9 --gpu 1 > logs/hw_ppo_dr_hw96_cTR.log 2>&1 &
F1T_PROCS=8 $PY -u hw_stage.py --policy runs/ppo_dr_s0/policy_final.npz --out runs/hw/ppo_dr_hw48_s3 --iters 8 --seed 9003 --gpu 2 > logs/hw_ppo_dr_hw48_s3.log 2>&1 &
wait
F1T_PROCS=20 $PY -u reeval.py v3nom_s0_hw96_cTR ppo_dr_hw96_cTR ppo_dr_hw48_s3 > logs/reeval4.log 2>&1
echo "$(date +%H:%M) Q4_DONE"
