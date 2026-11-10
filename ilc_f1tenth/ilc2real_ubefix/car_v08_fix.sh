#!/bin/bash
# car_v08_fix.sh (2026-10-30 04:10): car_v08.sh lost freeNom5V_s2 / freeDR5V_s1 / freeDR5V_s2 to a flaky XLA compile check
# (gpu_performance_model_base.cc:378 "common_utilization <= producer_output_utilization (1.88893 vs. 1.88893)", float
# equality; the same recipe compiled for the other seeds) and so its zero-shot validation eval died on the missing file.
# This retrains the three (same command, retried), reruns the zero-shot validation, runs their validation stages once
# car_v08.sh's own sweep is done, and then the full stage validation eval.
cd "$(dirname "$0")"; source env_tight.sh
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=2
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
T5="mocap_square2fast mocap_figfast mocap_square2fastM mocap_figfastM mocap_stadT"
P5=""; for t in $T5; do for r in _v08 _v09 ""; do P5="$P5 ${t}_b0_free$r"; done; done
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --hidden 256"
st() { n=$1; g=$2; shift 2; for try in 1 2 3; do [ -f runs/$n/policy_final.npz ] && return
  rm -rf runs/$n; $L shared $g $PY -u sim_train.py --plans $P5 --out runs/$n $V3 $TASK --gpu $g "$@" > logs/$n.log 2>&1; done; }
st freeNom5V_s2 1 --lanes 384 --dr 0 --anchor-w 1.0 --seed 2 &
st freeDR5V_s1 2 --lanes 384 --dr 1.0 --anchor-w 1.0 --seed 1 &
st freeDR5V_s2 1 --lanes 384 --dr 1.0 --anchor-w 1.0 --seed 2 &
wait
NETS="freeNom5V freeDR5V freeDRg5V freePPOV freePPO5V freePPOg5V"
for n in $NETS; do for s in 0 1 2; do [ -f runs/${n}_s$s/policy_final.npz ] || { echo "MISSING runs/${n}_s$s"; exit 1; }; done; done
echo "$(date +%T) retrain done"
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift ax:mu=4.5 ax:mu=5.5,mass=1.1"
ARGS="lqr_V_val lqr"
for s in 0 1 2; do for n in $NETS; do ARGS="$ARGS ${n}_s${s}_valzs $C/runs/${n}_s$s/policy_final.npz"; done; done
EVAL_FREE_CARS="$CARS" EVAL_FREE_PLAN_SUFFIX=_v08 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=10 OMP_NUM_THREADS=1 $PY -u $EV $ARGS > logs/v08_valzs.log 2>&1
echo "$(date +%T) zero-shot validation done"
while kill -0 529483 2>/dev/null; do sleep 30; done
G="mocap_square2fast_b0_free_v08 mocap_figfast_b0_free_v08"
BASE="--iters 5 --objective task --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --eval-runs 4 --eval-laps 5 --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
declare -A CFG=([S0]="--beta 0.5 --delta 0.05 --cap 0.05"
                [S1]="--beta 0.5 --delta 0.05 --cap 0.05 --fit-realize 0.9"
                [S2]="--beta 1.0 --delta 0.005 --cap 0.1 --fit-realize 0.9"
                [S3]="--beta 1.0 --delta 0.02 --cap 0.1 --fit-realize 0.9")
export OMP_NUM_THREADS=1 F1T_PROCS=3
NPAR=9
for cfg in S0 S2 S1 S3; do for s in 0 1 2; do for n in freeDR5V freeNom5V freePPOV; do
  out=runs/hw/${n}_s${s}_val${cfg}_stage
  [ -f $out/summary.json ] && continue
  while [ $(jobs -r | wc -l) -ge $NPAR ]; do sleep 10; done
  ( rm -rf $out; $L tiny $s $PY -u hw_stage.py --policy runs/${n}_s$s/policy_final.npz --out $out --cars $CARS --goals $G \
      --anchor-plans $G $BASE ${CFG[$cfg]} --gpu $s > logs/hw_${n}_s${s}_val${cfg}_stage.log 2>&1 ) &
  sleep 2
done; done; done
wait
ARGS=""
for s in 0 1 2; do for cfg in S0 S1 S2 S3; do for n in freeDR5V freeNom5V freePPOV; do
  [ -f runs/hw/${n}_s${s}_val${cfg}_stage/summary.json ] && ARGS="$ARGS ${n}_s${s}_val${cfg}+stage $C/runs/hw/${n}_s${s}_val${cfg}_stage"
done; done; done
EVAL_FREE_CARS="$CARS" EVAL_FREE_PLAN_SUFFIX=_v08 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=27 $PY -u $EV $ARGS > logs/v08_valstage.log 2>&1
echo "$(date +%T) CAR_V08_FIX_DONE"
