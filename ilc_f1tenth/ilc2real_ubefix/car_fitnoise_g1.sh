#!/bin/bash
# car_fitnoise.sh (2026-10-30): the per-track stage with large steps (car_pertrack.sh C1) broke plastic cars: freeDR5V_s0 real_nom, one
# update on square2fast: J 7.6 -> 236 (predicted 3.5), feedback gain change 2.4 x its own size on the current channel, 0.33 x on steering.
# VALIDATION of --fit-noise (the refit keeps the old network's slope around the laps), seed 0, one track per stage, anchor 20, GPUs 1-2:
#   C2 beta 0.5, delta 0.05, cap 0.1, fit-realize 0.9          F1 C2 + --fit-noise 0.05        F2 C2 + --fit-noise 0.2
# (only --gpu 1 starts CUDA since GPU 0's second drop: --gpu 2 jobs die at once with NO_DEVICE)
cd "$(dirname "$0")"; source env_tight.sh
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=3
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift ax:mu=4.5 ax:mu=5.5,mass=1.1"
G="mocap_square2fast_b0_free_v08 mocap_figfast_b0_free_v08"
BASE="--iters 5 --objective task --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-plans $G --fit-realize 0.9 --fit-max-steps 3000 \
  --eval-runs 4 --eval-laps 5 --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1 --beta 0.5 --delta 0.05 --cap 0.1"
declare -A CFG=([C2]="" [F1]="--fit-noise 0.05" [F2]="--fit-noise 0.2")
NPAR=${NPAR:-4}; g=1
for cfg in F2 F1 C2; do for n in freeDR5V freePPOV; do for tr in square2fast figfast; do
  out=runs/hw/${n}_s0_val${cfg}_${tr}_stage
  [ -f $out/summary.json ] && continue
  pgrep -f "hw_stage.py.*--out $out " > /dev/null && continue        # (still running from car_fitnoise.sh)
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  ( rm -rf $out; $L tiny $g $PY -u hw_stage.py --policy runs/${n}_s0/policy_final.npz --out $out --cars $CARS --goals mocap_${tr}_b0_free_v08 \
      $BASE ${CFG[$cfg]} --gpu $g > logs/hw_${n}_s0_val${cfg}_${tr}_stage.log 2>&1 ) &
  sleep ${STAGGER:-45}
done; done; done
wait
echo "$(date +%T) stages done"
ARGS=""
for cfg in C2 F1 F2; do for n in freeDR5V freePPOV; do for tr in square2fast figfast; do
  [ -f runs/hw/${n}_s0_val${cfg}_${tr}_stage/summary.json ] && ARGS="$ARGS ${n}_s0_val${cfg}${tr}+stage $C/runs/hw/${n}_s0_val${cfg}_${tr}_stage"
done; done; done
EVAL_FREE_CARS="$CARS" EVAL_FREE_PLAN_SUFFIX=_v08 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=10 $PY -u $EV $ARGS > logs/car_fitnoise_eval.log 2>&1
echo "$(date +%T) CAR_FITNOISE_DONE"
