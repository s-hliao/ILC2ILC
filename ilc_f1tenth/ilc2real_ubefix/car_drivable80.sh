#!/bin/bash
# car_drivable80.sh (2026-10-27): car_drivable.sh with the _mu80 plan instead of _mu60 -- the _mu60 plan (planned for 0.6x grip) steers so hard that every normal-grip car hits the wall within 0.2 laps, even the plan LQR (91 of 144 runs crash); the _mu80 plan is the most drivable single plan (plan LQR crashes 16 of 144, 12 of them the x0.6 car, which no plan shared with the normal cars can serve). Original header:
# ONE drivable plan per track for every car --
# the free plan's _mu80 variant (trajopt_drift --mu-plan 0.8: planned at 0.8 x the nominal friction; the x0.6 car, the
# lowest of the suite, can fly it; at the full-speed plan every method crashes it ~100 %). No ladder, no switching; the
# task objective as everywhere (TASK_SCALE: e_y 0.05 m, pace 0.3 m/s); success = no crash, rms e_y <= 10 cm, pace >= 90 %
# of THAT plan. Every network already trained on these plans (the banks hold the _mu80 variants).
#   zero-shot: LQR, ours nominal 5-track (freeNom5, trained here), ours + DR 5-track (freeDR5), PPO+DR (freePPO, freePPO5)
#   + 10 laps of our stage (5 x one 2-lap trial, worst goal first) on the two training tracks' _mu80 plans: freeNom5,
#     freeDR5, freePPO
# eval_free.py with EVAL_FREE_PLAN_SUFFIX=_mu80 (training tracks + held-out squareH). GPU: gpu_lock shared / tiny.
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
P=""; for t in mocap_square2fast mocap_figfast mocap_square2fastM mocap_figfastM mocap_stadT; do for r in "" _mu80 _mu60; do P="$P ${t}_b0_free$r"; done; done
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --lanes 384 --hidden 256"
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift"
G="mocap_square2fast_b0_free_mu80 mocap_figfast_b0_free_mu80"
ST="--iters 5 --beta 0.5 --delta 0.05 --cap 0.05 --objective task --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --eval-runs 4 --eval-laps 5 --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
stage() { n=$1; pol=$2; g=$3; [ -f runs/hw/${n}_stage/summary.json ] || { rm -rf runs/hw/${n}_stage
  $L tiny $g $PY -u hw_stage.py --policy $pol --out runs/hw/${n}_stage --cars $CARS --goals $G --anchor-plans $G $ST --gpu $g > logs/hw_${n}_stage.log 2>&1; }; }
for s in 0 1 2; do
  ( [ -f runs/freeNom5_s$s/policy_final.npz ] || OMP_NUM_THREADS=2 $L shared $s $PY -u sim_train.py --plans $P --out runs/freeNom5_s$s $V3 \
        --err-scale 0.05 1e9 1e9 1e9 1e9 0.3 --dr 0 --anchor-w 1.0 --seed $s --gpu $s > logs/freeNom5_s$s.log 2>&1
    stage freeNom5_s${s}_d80 runs/freeNom5_s$s/policy_final.npz $s ) &
  stage freeDR5_s${s}_d80 runs/freeDR5_s$s/policy_final.npz $s &
  stage freePPO_s${s}_d80 runs/freePPO_s$s/policy_final.npz $s &
done; wait
echo "$(date +%T) stages done"
ARGS="lqr_d80 lqr"
for s in 0 1 2; do
  for n in freeNom5 freeDR5 freePPO freePPO5; do ARGS="$ARGS ${n}_s${s}_d80zs $C/runs/${n}_s$s/policy_final.npz"; done
  for n in freeNom5 freeDR5 freePPO; do [ -f runs/hw/${n}_s${s}_d80_stage/summary.json ] && ARGS="$ARGS ${n}_s${s}_d80+stage $C/runs/hw/${n}_s${s}_d80_stage"; done
done
EVAL_FREE_PLAN_SUFFIX=_mu80 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=12 $PY -u $EV $ARGS > logs/free_d80.log 2>&1
echo "$(date +%T) CAR_DRIVABLE80_DONE"
