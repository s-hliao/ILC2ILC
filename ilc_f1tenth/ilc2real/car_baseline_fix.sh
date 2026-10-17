#!/bin/bash
# car_baseline_fix.sh (2026-10-17): diagnose the car's RL baselines.
#  1. objective mismatch: PPO+DR / RMA were rewarded on the plan's full drift state (sim_jax.ERR_SCALE); the headline
#     network v3drBt on e_y + pace (TASK_SCALE). Every full-state tracker (LQR too) crashes on figfast_b25 of the MB cars.
#     -> ppo_dr_task_s0, rma_task_s0 (--reward task)
#  2. RMA phase 2: off-policy (teacher rollouts) and its history one period ahead of deployment (a_t vs a_{t-1}), H 10.
#     -> rma_p2fix_s0: the OLD teacher (track reward), phase 2 on-policy, H 40 (isolates the estimator)
# Zero-shot evaluation as rma_zs / ppo_dr_zs (hw_stage --iters 0) -> runs/hw/<name>_zs.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
PL8=$(python3 -c "print(' '.join(f'mocap_{t}_b{b}' for t in ('square2fast','figfast') for b in (0,10,18,25)))")
INIT=runs/v1nom_s0/policy_bc.npz
P2="--hist 40 --p2-onpolicy 1 --p2-am 10 --adapt-iters 150"
( $PY -u ppo_dr.py --plans $PL8 --out runs/ppo_dr_task_s0 --init $INIT --mode ppo --reward task --gpu 1 > logs/ppo_dr_task_s0.log 2>&1
  F1T_PROCS=4 $PY -u hw_stage.py --policy runs/ppo_dr_task_s0/policy_final.npz --out runs/hw/ppo_dr_task_s0_zs --iters 0 --gpu 1 > logs/hw_ppo_dr_task_s0_zs.log 2>&1 ) &
( $PY -u ppo_dr.py --plans $PL8 --out runs/rma_task_s0 --init $INIT --mode rma --reward task $P2 --gpu 2 > logs/rma_task_s0.log 2>&1
  F1T_PROCS=4 $PY -u hw_stage.py --policy runs/rma_task_s0/policy_final.npz --out runs/hw/rma_task_s0_zs --iters 0 --gpu 2 > logs/hw_rma_task_s0_zs.log 2>&1 ) &
( $PY -u ppo_dr.py --plans $PL8 --out runs/rma_p2fix_s0 --mode rma --adapt-only --teacher runs/rma_s0/policy_final.npz $P2 --gpu 3 > logs/rma_p2fix_s0.log 2>&1
  F1T_PROCS=4 $PY -u hw_stage.py --policy runs/rma_p2fix_s0/policy_final.npz --out runs/hw/rma_p2fix_s0_zs --iters 0 --gpu 3 > logs/hw_rma_p2fix_s0_zs.log 2>&1 ) &
wait
echo "$(date +%T) CAR_BASELINE_FIX_DONE"
