#!/bin/bash
# The F1TENTH ILC2Real evaluation matrix (hardware stage / zero-shot on the 5 MB cars). Waits for each arm's policy,
# runs up to $PAR arms at a time. Re-runnable: arms with a summary.json are skipped.
cd "$(dirname "$0")"
PY=~/miniconda3/envs/f1t/bin/python
PAR=${PAR:-3}
PL="mocap_square2fast_b0 mocap_square2fast_b10 mocap_square2fast_b18 mocap_square2fast_b25 mocap_figfast_b0 mocap_figfast_b10 mocap_figfast_b18 mocap_figfast_b25"
V3="--anchor-w 1.0 --beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5"
gpu=0
arm() {  # arm NAME POLICY [hw_stage args]
  local name=$1 pol=$2; shift 2
  [ -f runs/hw/$name/summary.json ] && return
  until [ -f "$pol" ] || [ "$pol" = lqr ]; do sleep 30; done
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 10; done
  gpu=$(( (gpu + 1) % 4 ))
  echo "$(date +%H:%M) start $name"
  rm -rf runs/hw/$name
  F1T_PROCS=10 $PY -u hw_stage.py --policy $pol --out runs/hw/$name --gpu $gpu "$@" > logs/hw_$name.log 2>&1 &
}
# DR-A: DR fine-tune of the nominal sim network
( [ -f runs/v3drA_s0/policy_final.npz ] || { until [ -f runs/v3nom_s0/policy_final.npz ]; do sleep 30; done;
  $PY -u sim_train.py --plans $PL --out runs/v3drA_s0 --gpu 0 --seed 0 --init runs/v3nom_s0/policy_final.npz $V3 \
      --dr 1.0 --iters 15 > logs/v3drA_s0.log 2>&1; } ) &
# ours (nominal sim) x 3 sim seeds
for s in 0 1 2; do
  arm v3nom_s${s}_zs runs/v3nom_s$s/policy_final.npz --iters 0
  arm v3nom_s${s}_hw24 runs/v3nom_s$s/policy_final.npz --iters 4
done
arm v3nom_s0_hw24vs runs/v3nom_s0/policy_final.npz --iters 4 --train-starts 0.5
# DR variants
arm v3drA_s0_zs runs/v3drA_s0/policy_final.npz --iters 0
arm v3drA_s0_hw24 runs/v3drA_s0/policy_final.npz --iters 4
arm v3drB_s0_zs runs/v3drB_s0/policy_final.npz --iters 0
arm v3drB_s0_hw24 runs/v3drB_s0/policy_final.npz --iters 4
# hardware-stage ablations (from v3nom_s0)
arm v3nom_s0_hw48 runs/v3nom_s0/policy_final.npz --iters 8
arm v3nom_s0_hw96 runs/v3nom_s0/policy_final.npz --iters 16
arm v3nom_s0_hw24_openG runs/v3nom_s0/policy_final.npz --iters 4 --jac open
arm v3nom_s0_hw24_flip runs/v3nom_s0/policy_final.npz --iters 4 --jac flip
arm v3nom_s0_hw24_noTR runs/v3nom_s0/policy_final.npz --iters 4 --anchor-w 0
arm v3nom_s0_hw24_noRB runs/v3nom_s0/policy_final.npz --iters 4 --rollback 0
# sim-stage ablations
arm bcB_zs2 runs/bcB/policy_final.npz --iters 0
arm v1narrow_hw24 runs/nom_s0/policy_it020.npz --iters 4
arm v3olG_s0_zs runs/v3olG_s0/policy_final.npz --iters 0
arm v3olG_s0_hw24 runs/v3olG_s0/policy_final.npz --iters 4
arm v3noexp_s0_zs runs/v3noexp_s0/policy_final.npz --iters 0
arm v3noexp_s0_hw24 runs/v3noexp_s0/policy_final.npz --iters 4
# RL baselines (zero-shot) and PPO + our hardware stage (DR-compatibility)
arm ppo_dr_zs runs/ppo_dr_s0/policy_final.npz --iters 0
arm ppo_dr_hw24 runs/ppo_dr_s0/policy_final.npz --iters 4
arm rma_zs runs/rma_s0/policy_final.npz --iters 0
wait
echo "$(date +%H:%M) QUEUE_DONE"
