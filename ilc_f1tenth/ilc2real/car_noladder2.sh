#!/bin/bash
# car_noladder2.sh (2026-10-27, after car_noladder.sh): on the low-friction cars the no-ladder stage cuts e_y but not the
# crashes (53 %) -- 5 steps capped at rms 0.05 cannot take the 15-20 % of speed off the infeasible plan. The paper's knobs,
# on the 5-track networks (freeDR5: held-out squareH crashes 39 -> 28 %), V 3.0:
#   nl5V3            the car_noladder.sh stage (cap 0.05, beta 0.5) -- the 5-track baseline
#   nl5V3cap         s_max 0.10, beta 0.8 (bigger steps)
#   nl5V3p90         the pace row against 0.9 of the plan's speed (--pace-target 0.9; success still needs >= 90 %)
# (car_noladder.sh, for the record:) the hardware stage WITHOUT the
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
for s in 0 1 2; do
  ARMS+=("freeDR5_s${s}_nl5V3:runs/freeDR5_s$s/policy_final.npz:")
  ARMS+=("freeDR5_s${s}_nl5V3cap:runs/freeDR5_s$s/policy_final.npz:--cap 0.10 --beta 0.8")
  ARMS+=("freeDR5_s${s}_nl5V3p90:runs/freeDR5_s$s/policy_final.npz:--pace-target 0.9")
done
NPAR=${NPAR:-5}; g=0
for x in "${ARMS[@]}"; do
  IFS=: read n pol extra <<< "$x"
  [ -f runs/hw/${n}_stage/summary.json ] && continue
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  rm -rf runs/hw/${n}_stage
  $L tiny $g $PY -u hw_stage.py --policy $pol --out runs/hw/${n}_stage --cars $CARS --goals $G --anchor-plans $G $ST \
      --err-scale 0.05 1e9 1e9 1e9 1e9 3.0 $extra --gpu $g > logs/hw_${n}_stage.log 2>&1 &
  g=$(( (g + 1) % 3 )); sleep 5
done; wait
echo "$(date +%T) stages done"
ARGS=""
for x in "${ARMS[@]}"; do n=${x%%:*}; [ -f runs/hw/${n}_stage/summary.json ] && ARGS="$ARGS ${n}+stage $C/runs/hw/${n}_stage"; done
EVAL_FREE_NO_RUNGS=1 F1T_PROCS=12 $PY -u $EV $ARGS > logs/free_noladder2.log 2>&1
echo "$(date +%T) CAR_NOLADDER2_DONE"
