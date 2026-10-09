#!/bin/bash
# after queue_final.sh: second DR seeds, reruns of the two nominal hw24 arms that may predate the NaN guard,
# then the larger re-evaluation of every final arm (reeval.py, 12 runs per plan per car) and the long chains.
cd "$(dirname "$0")"
PY=~/miniconda3/envs/f1t/bin/python
until grep -q QUEUE_DONE logs/queue_final.log; do sleep 30; done
PAR=3; gpu=0
arm() {
  local name=$1 pol=$2; shift 2
  [ -f runs/hw/$name/summary.json ] && return
  until [ -f "$pol" ]; do sleep 30; done
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 10; done
  gpu=$(( (gpu + 1) % 4 )); echo "$(date +%H:%M) start $name"
  F1T_PROCS=10 $PY -u hw_stage.py --policy $pol --out runs/hw/$name --gpu $gpu "$@" > logs/hw_$name.log 2>&1 &
}
arm v3nom_s0_hw24r runs/v3nom_s0/policy_final.npz --iters 4
arm v3nom_s1_hw24r runs/v3nom_s1/policy_final.npz --iters 4
arm v3drA_s1_zs runs/v3drA_s1/policy_final.npz --iters 0
arm v3drA_s1_hw24 runs/v3drA_s1/policy_final.npz --iters 4
arm v3drB_s1_zs runs/v3drB_s1/policy_final.npz --iters 0
arm v3drB_s1_hw24 runs/v3drB_s1/policy_final.npz --iters 4
wait
echo "$(date +%H:%M) arms done; reeval"
F1T_PROCS=24 $PY -u reeval.py lqr_zs bcB_zs v3nom_s0_zs v3nom_s1_zs v3nom_s2_zs v3nom_s0_hw24r v3nom_s1_hw24r v3nom_s2_hw24 \
  v3nom_s0_hw24vs v3nom_s0_hw48 v3nom_s0_hw96 v3drA_s0_zs v3drA_s0_hw24 v3drA_s1_zs v3drA_s1_hw24 v3drB_s0_zs v3drB_s0_hw24 \
  v3drB_s1_zs v3drB_s1_hw24 v3nom_s0_hw24_openG v3nom_s0_hw24_flip v3nom_s0_hw24_noTR v3nom_s0_hw24_noRB v1narrow_hw24 \
  v3olG_s0_zs v3olG_s0_hw24 v3noexp_s0_zs v3noexp_s0_hw24 ppo_dr_zs ppo_dr_hw24 rma_zs bcB_hw24b > logs/reeval.log 2>&1
for arm in v3nom_s0_hw24r v3drB_s0_hw24 v3nom_s0_zs; do F1T_PROCS=12 $PY -u long_eval.py $arm --laps 20 > logs/long_$arm.log 2>&1; done
echo "$(date +%H:%M) EXTRA_DONE"
