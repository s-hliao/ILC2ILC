#!/bin/bash
# car_jacdiag.sh (2026-10-30): is the car stage's failure the Jacobian or the refit? track_ilc.py --policy: the stage's exact GN step
# (network feedback, nominal-sim Jacobians at the measured states, task objective) applied as a fixed per-track feedforward on top of
# the UNCHANGED network, no refit. If the lap cost falls as predicted, the refit is at fault; if not, the Jacobian is.
# Step size as the stage (cap 0.1) at gain beta 0.25 / 0.5 / 1.0; plastic cars + one rubber car; square2fast and figfast _v08.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t_ube/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4
CARS="real_nom real_mu real_lag harsh_steer ax:mu=0.9 ax:mu=0.7 ax:mu=4.5"
PL="mocap_square2fast_b0_free_v08 mocap_figfast_b0_free_v08"
g=0
for n in freeDR5V freePPOV; do for b in 0.25 0.5 1.0; do
  out=runs/diag/jac_${n}_s0_b$b; [ -f $out/summary.json ] && continue
  ( $L tiny $g $PY -u track_ilc.py --out $out --policy runs/${n}_s0/policy_final.npz --cars $CARS --plans $PL --budgets 2 5 \
      --beta $b --delta 0.05 --cap 0.1 --eval-runs 4 --eval-laps 5 --eval-seed 301 --log-cars --gpu $g > logs/jacdiag_${n}_b$b.log 2>&1 ) &
  g=$(( (g + 1) % 4 )); sleep 30
done; done
wait
echo "$(date +%T) JACDIAG_DONE"
