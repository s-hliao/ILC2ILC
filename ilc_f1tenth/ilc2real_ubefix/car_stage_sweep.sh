#!/bin/bash
# car_stage_sweep.sh (2026-10-30): VALIDATION sweep of the car hardware stage's paper knobs (validation cars of
# car_drivable80_e5.sh, eval seed 301; the test tiers are not touched). Diagnosis (tiers stage logs): the anchored refit
# realizes only 23-30 % of the GN step after the first update (a fixed 800 Adam steps), with beta 0.5 / delta 0.05 /
# cap 0.05 -- the quad's "steps too small" failure; the pogo works with the full GN step (beta 1, delta 0.005, cap 0.1).
#   S0  the frozen recipe (= runs/hw/*_d80e5_stage, reused)
#   S1  S0 + --fit-realize 0.9 (refit the targets to convergence, max 8000 steps)
#   S2  S1 + beta 1.0, delta 0.005, cap 0.1 (the pogo's knobs)
#   S3  S1 + beta 1.0, delta 0.02, cap 0.1
# 10 laps per car (5 x one 2-lap trial, worst goal first), arms freeNom5 / freeDR5 / freePPO x seeds 0-2.
# -> runs/hw/<arm>_s<k>_val<S>_stage, eval_free/<arm>_s<k>_val<S>+stage.json; table: car_stage_sweep_table.py
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=3   # 9 stages x 3 = 27 CPUs
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift"
G="mocap_square2fast_b0_free_mu80 mocap_figfast_b0_free_mu80"
BASE="--iters 5 --objective task --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --eval-runs 4 --eval-laps 5 --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
declare -A CFG=([S1]="--beta 0.5 --delta 0.05 --cap 0.05 --fit-realize 0.9"
                [S2]="--beta 1.0 --delta 0.005 --cap 0.1 --fit-realize 0.9"
                [S3]="--beta 1.0 --delta 0.02 --cap 0.1 --fit-realize 0.9")
NPAR=9
for cfg in S1 S2 S3; do for s in 0 1 2; do for n in freeDR5 freeNom5 freePPO; do
  out=runs/hw/${n}_s${s}_val${cfg}_stage
  [ -f $out/summary.json ] && continue
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  ( rm -rf $out; $L tiny $s $PY -u hw_stage.py --policy runs/${n}_s$s/policy_final.npz --out $out --cars $CARS --goals $G \
      --anchor-plans $G $BASE ${CFG[$cfg]} --gpu $s > logs/hw_${n}_s${s}_val${cfg}_stage.log 2>&1 ) &
  sleep 2
done; done; done
wait
echo "$(date +%T) stages done"
ARGS=""
for cfg in S1 S2 S3; do for s in 0 1 2; do for n in freeDR5 freeNom5 freePPO; do
  [ -f runs/hw/${n}_s${s}_val${cfg}_stage/summary.json ] && ARGS="$ARGS ${n}_s${s}_val${cfg}+stage $C/runs/hw/${n}_s${s}_val${cfg}_stage"
done; done; done
EVAL_FREE_PLAN_SUFFIX=_mu80 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=27 $PY -u $EV $ARGS > logs/free_stage_sweep.log 2>&1
echo "$(date +%T) CAR_STAGE_SWEEP_DONE"
