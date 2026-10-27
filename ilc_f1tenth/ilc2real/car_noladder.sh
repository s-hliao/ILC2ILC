#!/bin/bash
# car_noladder.sh (2026-10-27; user: "find a better way than the car plan ladder"): the hardware stage WITHOUT the
# ladder (no --fallback-*: the reference never switches) -- the paper's rule only, the task's loss weights Q setting the
# trade between path and pace. The free plans are infeasible at full pace on the low-friction cars (x0.6 / x0.7: every
# controller crashes), so with the pace row weighted as now (V scale 0.3 m/s) the GN step demands an impossible speed;
# a larger V scale (1.0, 3.0 m/s) lets the path row dominate and the step slows the car where the path needs it.
# Arms: freeDR_s<k> (ours + DR) x V in {0.3, 1.0, 3.0}, 10 laps (5 x one 2-lap trial, worst goal first), then
# eval_free.py (EVAL_FREE_NO_RUNGS=1: every arm on the original free plans; continuous e_y / pace / crashes per car).
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift"
G="mocap_square2fast_b0_free mocap_figfast_b0_free"
ST="--iters 5 --beta 0.5 --delta 0.05 --cap 0.05 --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --eval-runs 4 --eval-laps 5 --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
ARMS=()
for s in 0 1 2; do for V in 0.3 1.0 3.0; do ARMS+=("freeDR_s${s}_nlV$V:runs/freeDR_s$s/policy_final.npz:$V"); done; done
NPAR=${NPAR:-5}; g=0
for x in "${ARMS[@]}"; do
  IFS=: read n pol V <<< "$x"
  [ -f runs/hw/${n}_stage/summary.json ] && continue
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  rm -rf runs/hw/${n}_stage
  $L tiny $g $PY -u hw_stage.py --policy $pol --out runs/hw/${n}_stage --cars $CARS --goals $G --anchor-plans $G $ST \
      --err-scale 0.05 1e9 1e9 1e9 1e9 $V --gpu $g > logs/hw_${n}_stage.log 2>&1 &
  g=$(( (g + 1) % 3 )); sleep 5
done; wait
echo "$(date +%T) stages done"
ARGS=""
for x in "${ARMS[@]}"; do n=${x%%:*}; [ -f runs/hw/${n}_stage/summary.json ] && ARGS="$ARGS ${n}+stage $C/runs/hw/${n}_stage"; done
EVAL_FREE_NO_RUNGS=1 F1T_PROCS=12 $PY -u $EV $ARGS > logs/free_noladder.log 2>&1
echo "$(date +%T) CAR_NOLADDER_DONE"
