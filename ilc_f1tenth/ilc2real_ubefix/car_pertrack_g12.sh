#!/bin/bash
# car_pertrack.sh (2026-10-30): car_anchor.sh's stage still hurt plastic tracking (paired +0.39 at best: square2fast 2.9 -> 6.8 cm,
# updates on one track carrying over to the other). VALIDATION of the stage on ONE track at a time, as Nguyen et al.'s ILC practises
# one task (the existing per-track protocol, merge_pertrack.py): 5 iterations of 2 laps on that track, both plans anchored, weight 20
# (the sweep's least harmful), larger steps (only 2-3 trials per track before; refit to --fit-realize):
#   C1 beta 1, delta 0.02, cap 0.2       C2 beta 0.5, delta 0.05, cap 0.1
# nets freeDR5V (ours + DR) / freePPOV (PPO + DR), seeds 0-2, the 14 validation cars (plastic + rubber), 1.0 m envelope, _v08 plans,
# staggered -> runs/hw/<net>_s<s>_val<cfg>_<track>_stage; eval_free *_val<cfg><track>+stage (scored on the practised track)
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=3
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift ax:mu=4.5 ax:mu=5.5,mass=1.1"
G="mocap_square2fast_b0_free_v08 mocap_figfast_b0_free_v08"
BASE="--iters 5 --objective task --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-plans $G --fit-realize 0.9 \
  --eval-runs 4 --eval-laps 5 --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
declare -A CFG=([C1]="--beta 1.0 --delta 0.02 --cap 0.2" [C2]="--beta 0.5 --delta 0.05 --cap 0.1")
NPAR=${NPAR:-2}; g=1   # (GPUs 1-2 only: GPU 0 fell off the bus twice on 10-30)
for cfg in C1 C2; do for s in 0 1 2; do for n in freeDR5V freePPOV; do for tr in square2fast figfast; do
  out=runs/hw/${n}_s${s}_val${cfg}_${tr}_stage
  [ -f $out/summary.json ] && continue
  pgrep -f "hw_stage.py.*--out $out " > /dev/null && continue        # (still running from car_pertrack.sh)
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  ( rm -rf $out; $L tiny $g $PY -u hw_stage.py --policy runs/${n}_s$s/policy_final.npz --out $out --cars $CARS --goals mocap_${tr}_b0_free_v08 \
      $BASE ${CFG[$cfg]} --gpu $g > logs/hw_${n}_s${s}_val${cfg}_${tr}_stage.log 2>&1 ) &
  g=$(( g % 2 + 1 )); sleep ${STAGGER:-60}
done; done; done; done
wait
echo "$(date +%T) stages done"
ARGS=""
for cfg in C1 C2; do for s in 0 1 2; do for n in freeDR5V freePPOV; do for tr in square2fast figfast; do
  [ -f runs/hw/${n}_s${s}_val${cfg}_${tr}_stage/summary.json ] && ARGS="$ARGS ${n}_s${s}_val${cfg}${tr}+stage $C/runs/hw/${n}_s${s}_val${cfg}_${tr}_stage"
done; done; done; done
EVAL_FREE_CARS="$CARS" EVAL_FREE_PLAN_SUFFIX=_v08 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=12 $PY -u $EV $ARGS > logs/car_pertrack_eval.log 2>&1
echo "$(date +%T) CAR_PERTRACK_DONE"
