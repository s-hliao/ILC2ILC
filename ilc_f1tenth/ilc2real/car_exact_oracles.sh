#!/bin/bash
# EXACT-DYNAMICS ORACLES on the car (log/dilc/plane/match/EXACT_ORACLES_RUNBOOK.md, C0-C1; user 2026-10-11/12: "make true dynamics
# the true robot for rma and fada ... we have to show that reinforcement learning is capable of performing transfer under
# idealized conditions"). PPO on each multi-body car itself (real_car.run_many: the function and cars every evaluation uses;
# real_car.exact_check asserts the tight-track setup), privileged critic, and the IDEALIZED reference: each car flies and is
# scored on the plan rungs our method settled on for it (--rungs-from v3drBtL_s0_lad1_lap10: real_mu _mu80 both tracks,
# real_lag figfast _mu80). An upper bound: the oracle budget is not real data.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
NP=14                                   # CPUs (the quadruped's oracles run beside: <= 32 in total)
OURS=v3drBtL_s0_lad1_lap10
CK="96 384 1536 3072 5000 10000"
# C0: the exactness check + ep0 must reproduce rma_real_ep0 (b25 53.3 %, nominal plans)
[ -f runs/hw/rma_exact_smoke_ep0/summary.json ] || F1T_PROCS=$NP $PY -u ppo_real_car.py --init runs/rma_s0/policy_final.npz \
  --arm rma_exact_smoke --checkpoints 24 > logs/rma_exact_smoke.log 2>&1
[ -f runs/hw/rma_exact_smoke_ep0/eval_big.json ] || F1T_PROCS=$NP $PY -u reeval.py rma_exact_smoke_ep0 > logs/reeval_rma_exact_smoke.log 2>&1
echo "$(date +%T) C0 done"
# C1: RMA (the true-z teacher) and PPO+DR on the exact cars, idealized rungs, up to 10k laps per car
for arm in rma ppo; do
  init=runs/rma_s0/policy_final.npz; [ $arm = ppo ] && init=runs/ppo_dr_s0/policy_final.npz
  last=runs/hw/${arm}_exact_ep10000/summary.json
  [ -f $last ] || F1T_PROCS=$NP $PY -u ppo_real_car.py --init $init --arm ${arm}_exact --rungs-from $OURS --per-iter 96 \
    --checkpoints $CK > logs/${arm}_exact.log 2>&1
  F1T_PROCS=$NP $PY -u reeval.py ${arm}_exact_ep0 $(for k in $CK; do echo -n "${arm}_exact_ep$k "; done) > logs/reeval_${arm}_exact.log 2>&1
  echo "$(date +%T) C1 $arm done"
done
echo "$(date +%T) CAR_EXACT_DONE"
