#!/bin/bash
# car_jacdiag_ube.sh (2026-10-30): the car Jacobian-vs-refit diagnosis (track_ilc.py --policy) on ube's one working GPU (--gpu 1),
# after car_fitnoise_g1.sh (matcha's GPUs run pogo first, user). 3 at a time.
cd "$(dirname "$0")"; source env_tight.sh
until grep -q CAR_FITNOISE_DONE logs/car_fitnoise_g1.out 2>/dev/null; do sleep 30; done
PY=~/miniconda3/envs/f1t/bin/python; export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4
for n in freeDR5V freePPOV; do for b in 0.25 0.5 1.0; do
  out=runs/diag/jac_${n}_s0_b$b; [ -f $out/summary.json ] && continue
  while [ $(jobs -r | wc -l) -ge 3 ]; do sleep 10; done
  $PY -u track_ilc.py --out $out --policy runs/${n}_s0/policy_final.npz --cars real_nom real_mu real_lag harsh_steer ax:mu=0.9 ax:mu=0.7 ax:mu=4.5 \
    --plans mocap_square2fast_b0_free_v08 mocap_figfast_b0_free_v08 --budgets 2 5 --beta $b --delta 0.05 --cap 0.1 --eval-runs 4 --eval-laps 5 \
    --eval-seed 301 --log-cars --gpu 1 > logs/jacdiag_${n}_b$b.log 2>&1 &
  sleep 30
done; done
wait; echo "$(date +%T) JACDIAG_DONE"
