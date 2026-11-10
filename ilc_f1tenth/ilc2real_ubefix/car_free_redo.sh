#!/bin/bash
# car_free_redo.sh (2026-10-27): car_free_eval.sh / car_free_final.sh's real-data phase again -- 2026-10-26 20:43 the
# pogo / quad / car queues trained on the same GPUs: 9 of 15 hardware stages OOM'd or hung (freeDRP_s0), and
# ppo_real_car.py died on the "ax:" cars (true_z read REAL_CARS directly; fixed: mb_car.raw_car). Every stage whose
# summary.json is missing, PPO on the cars (24..384 episodes), then eval_free.py on the settled ladder rungs (+stageR)
# and the PPO checkpoints. GPU: log/gpu_lock.sh shared, round-robin.
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
  NETS+=("freeNom_s$s:runs/freeNom_s$s/policy_final.npz" "freeDR_s$s:runs/freeDR_s$s/policy_final.npz")
  for W in 3e-4 1e-3; do NETS+=("freeDRT${W}_s$s:runs/freeDRT${W}_s${s}_stop/policy_final.npz"); done
  NETS+=("freeDRP_s$s:runs/freeDRP_s$s/policy_final.npz" "freeDRF_s$s:runs/freeDRF_s$s/policy_final.npz")
done
NPAR=${NPAR:-4}; g=0
for nt in "${NETS[@]}"; do
  n=${nt%%:*}; pol=${nt#*:}
  [ -f runs/hw/${n}_stage/summary.json ] && continue
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  rm -rf runs/hw/${n}_stage
  $L shared $g $PY -u hw_stage.py --policy $pol --out runs/hw/${n}_stage --cars $CARS --goals $G --anchor-plans $G $LADR --gpu $g \
      > logs/hw_${n}_stage.log 2>&1 &
  g=$(( (g + 1) % 3 )); sleep 5
done
for s in 0 1 2; do
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  [ -f runs/hw/freePPOreal_s${s}_ep384/summary.json ] || $L shared $s $PY -u ppo_real_car.py --init runs/freePPO_s$s/policy_final.npz \
      --arm freePPOreal_s$s --cars $CARS --goals $G --checkpoints 24 48 96 192 384 --gpu $s > logs/ppo_real_free_s$s.log 2>&1 &
done
wait
echo "$(date +%T) real-data phase done"
ARGS=""
for nt in "${NETS[@]}"; do n=${nt%%:*}; [ -f runs/hw/${n}_stage/summary.json ] && ARGS="$ARGS ${n}+stageR $C/runs/hw/${n}_stage"; done
for s in 0 1 2; do for k in 24 48 96 192 384; do [ -d runs/hw/freePPOreal_s${s}_ep$k ] && ARGS="$ARGS freePPOreal_s${s}_ep$k $C/runs/hw/freePPOreal_s${s}_ep$k"; done; done
F1T_PROCS=16 $PY -u $EV $ARGS > logs/free_real_redo.log 2>&1
echo "$(date +%T) CAR_FREE_REDO_DONE"
