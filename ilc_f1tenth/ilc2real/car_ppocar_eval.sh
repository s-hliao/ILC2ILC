#!/bin/bash
# car_ppocar_eval.sh (2026-10-30, ube): score the on-car PPO baseline (ppo_real_car, from freePPOV_s*, checkpoints after 0-160
# episodes per car) with eval_free on the 1.0 m envelope, the same cars / _v08 plans / validation seed as car_anchor.sh's stages.
cd "$(dirname "$0")"; source env_tight.sh
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift ax:mu=4.5 ax:mu=5.5,mass=1.1"
ARGS=""
for s in 0 1 2; do for e in 0 10 20 40 80 160; do
  [ -d runs/hw/ppoCarV_s${s}_ep$e ] && ARGS="$ARGS ppoCarV_s${s}_ep${e}_val1m $C/runs/hw/ppoCarV_s${s}_ep$e"
done; done
EVAL_FREE_CARS="$CARS" EVAL_FREE_PLAN_SUFFIX=_v08 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=10 $PY -u $EV $ARGS > logs/car_ppocar_eval.log 2>&1
echo "$(date +%T) CAR_PPOCAR_EVAL_DONE" >> logs/car_ppocar_eval.log
