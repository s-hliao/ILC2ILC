#!/bin/bash
# car_tiers.sh (2026-10-29): the car's coverage tiers (mb_car.TIER_A / B / C_CARS: 12 + 12 + 12, defined before any run)
# and its FROZEN TEST PASS: car_drivable80_e5.sh's recipe unchanged (the same networks, _mu80 plans, stage arguments and
# 0.5 m envelope) on the tier cars, every arm evaluated once on a fresh seed (EVAL_FREE_SEED 4301; the d80 / d80e5 results
# used the validation seed 301). Scoring: the continuous normalized error max(rms e_y / 10 cm, (1 - pace) / 0.1), a crash
# = 5 (success = no crash, rms e_y <= 10 cm, pace >= 90 %, as before).
#   stages: freeNom5 / freeDR5 / freePPO (s0-2) + 10 laps of DistIL-C_hw on the tier cars -> runs/hw/<n>_s<k>_tiers_stage
#   evals:  LQR, the zero-shot networks (freeNom5, freeDR5, freePPO, freePPO5) and the stages -> eval_free/*_tiers_test.json
# Waits for the pogo tier A / B queue and the quad stage queue (PIDs below; the 30-CPU budget).
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
for P in ${WAIT_PIDS:-}; do while kill -0 $P 2>/dev/null; do sleep 60; done; done
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=3   # 9 stages x 3 = 27 CPUs
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
CARS=$($PY -c "import mb_car as m; print(' '.join(m.TIER_A_CARS + m.TIER_B_CARS + m.TIER_C_CARS))")
G="mocap_square2fast_b0_free_mu80 mocap_figfast_b0_free_mu80"
ST="--iters 5 --beta 0.5 --delta 0.05 --cap 0.05 --objective task --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --eval-runs 4 --eval-laps 5 --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
echo "$(date +%T) start: $CARS"
stage() { n=$1; pol=$2; g=$3; [ -f runs/hw/${n}_stage/summary.json ] || { rm -rf runs/hw/${n}_stage
  $L tiny $g $PY -u hw_stage.py --policy $pol --out runs/hw/${n}_stage --cars $CARS --goals $G --anchor-plans $G $ST --gpu $g > logs/hw_${n}_stage.log 2>&1; }; }
for s in 0 1 2; do for n in freeNom5 freeDR5 freePPO; do stage ${n}_s${s}_tiers runs/${n}_s$s/policy_final.npz $s & done; done
wait
echo "$(date +%T) stages done"
ARGS="lqr_tiers_test lqr"
for s in 0 1 2; do
  for n in freeNom5 freeDR5 freePPO freePPO5; do ARGS="$ARGS ${n}_s${s}_tiers_testzs $C/runs/${n}_s$s/policy_final.npz"; done
  for n in freeNom5 freeDR5 freePPO; do [ -f runs/hw/${n}_s${s}_tiers_stage/summary.json ] && ARGS="$ARGS ${n}_s${s}_tiers_test+stage $C/runs/hw/${n}_s${s}_tiers_stage"; done
done
EVAL_FREE_SEED=4301 EVAL_FREE_CARS="$CARS" EVAL_FREE_PLAN_SUFFIX=_mu80 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=24 $PY -u $EV $ARGS > logs/free_tiers_test.log 2>&1
echo "$(date +%T) CAR_TIERS_DONE"
