#!/bin/bash
# follow-ups: PPO+DR-initialized sim stages (nominal and DR), a second seed of the best arm, then x12 re-evaluation.
cd "$(dirname "$0")"
PY=~/miniconda3/envs/f1t/bin/python
PL="mocap_square2fast_b0 mocap_square2fast_b10 mocap_square2fast_b18 mocap_square2fast_b25 mocap_figfast_b0 mocap_figfast_b10 mocap_figfast_b18 mocap_figfast_b25"
V3="--anchor-w 1.0 --beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5"
$PY -u sim_train.py --plans $PL --out runs/v5ppoNom_s0 --gpu 0 --seed 0 --init runs/ppo_dr_s0/policy_final.npz $V3 --iters 20 > logs/v5ppoNom_s0.log 2>&1 &
$PY -u sim_train.py --plans $PL --out runs/v5ppoDR_s0 --gpu 1 --seed 0 --init runs/ppo_dr_s0/policy_final.npz $V3 --iters 20 --dr 1.0 > logs/v5ppoDR_s0.log 2>&1 &
F1T_PROCS=8 $PY -u hw_stage.py --policy runs/ppo_dr_s0/policy_final.npz --out runs/hw/ppo_dr_hw48_s2 --iters 8 --seed 9002 --gpu 2 > logs/hw_ppo_dr_hw48_s2.log 2>&1 &
wait
for a in v5ppoNom_s0 v5ppoDR_s0; do
  F1T_PROCS=8 $PY -u hw_stage.py --policy runs/$a/policy_final.npz --out runs/hw/${a}_zs --iters 0 --gpu 0 > logs/hw_${a}_zs.log 2>&1 &
  F1T_PROCS=8 $PY -u hw_stage.py --policy runs/$a/policy_final.npz --out runs/hw/${a}_hw24 --iters 4 --gpu 1 > logs/hw_${a}_hw24.log 2>&1 &
done
wait
F1T_PROCS=20 $PY -u reeval.py ppo_dr_hw48_s2 v5ppoNom_s0_zs v5ppoNom_s0_hw24 v5ppoDR_s0_zs v5ppoDR_s0_hw24 > logs/reeval3.log 2>&1
echo "$(date +%H:%M) Q3_DONE"
