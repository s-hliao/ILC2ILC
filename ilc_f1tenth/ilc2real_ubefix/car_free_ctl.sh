#!/bin/bash
# car_free_ctl.sh (2026-10-27): do the ILC updates add anything over the ladder's rungs on the free goals? (with the
# rung-aware eval_free.py, freeDR_s0 + stage 92.2 % = freeDR_s0 flown on that stage's rungs 92.2 %, current 8, train.)
# Per seed of freeDR (ours + DR) and freePPO (PPO+DR):
#   <net>_ladder0   the ladR stage with --beta 0: the same ladder / trials, no network update (the rung-only control)
#   <net>_lap20     the ladR stage at 10 iterations (20 laps) instead of 5 (10 laps)
#   freePPO_s<k>    PPO+DR through our ladR stage (10 laps)
# then eval_free.py on each arm's own rungs (+stageR). GPU: log/gpu_lock.sh shared, round-robin.
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=${F1T_PROCS:-4}
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift"
G="mocap_square2fast_b0_free mocap_figfast_b0_free"
LADR="--iters 5 --beta 0.5 --delta 0.05 --cap 0.05 --objective task --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --fallback-suffix _mu80,_mu60 --fallback-after 1 --fallback-on band --fallback-ey 0.1 --fallback-revert 1 --eval-runs 4 --eval-laps 5 \
  --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
NETS=()
for s in 0 1 2; do
  NETS+=("freeDR_s${s}_ladder0:runs/freeDR_s$s/policy_final.npz:--beta 0" "freeDR_s${s}_lap20:runs/freeDR_s$s/policy_final.npz:--iters 10")
  NETS+=("freePPO_s$s:runs/freePPO_s$s/policy_final.npz:" "freePPO_s${s}_ladder0:runs/freePPO_s$s/policy_final.npz:--beta 0")
done
NPAR=${NPAR:-3}; g=0
for nt in "${NETS[@]}"; do
  IFS=: read n pol extra <<< "$nt"
  [ -f runs/hw/${n}_stage/summary.json ] && continue
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  rm -rf runs/hw/${n}_stage
  $L shared $g $PY -u hw_stage.py --policy $pol --out runs/hw/${n}_stage --cars $CARS --goals $G --anchor-plans $G $LADR $extra --gpu $g \
      > logs/hw_${n}_stage.log 2>&1 &
  g=$(( (g + 1) % 3 )); sleep 5
done
wait
echo "$(date +%T) stages done"
ARGS=""
for nt in "${NETS[@]}"; do n=${nt%%:*}; [ -f runs/hw/${n}_stage/summary.json ] && ARGS="$ARGS ${n}+stageR $C/runs/hw/${n}_stage"; done
F1T_PROCS=12 $PY -u $EV $ARGS > logs/free_ctl.log 2>&1
echo "$(date +%T) CAR_FREE_CTL_DONE"
