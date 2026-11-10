#!/bin/bash
# car_final_arms.sh (2026-10-27): the rest of the car's no-ladder table, the stage = car_noladder2.sh's nl5V3 (no
# ladder, V scale 3.0, cap 0.05, beta 0.5, 10 laps; best with the fewest knobs: 8 cars train-track e_y 2.6 cm vs PPO+DR
# 6.3, held-out crashes 25 % vs 28 % zero-shot):
#   freeNom5_s<k>         ours nominal on the 5-track bank (freeDR5's recipe with --dr 0), zero-shot + nl5V3 stage
#   freePPO_s<k>_nl5V3    PPO+DR (2-track; the 5-track PPO+DR is worse: held-out crashes 42 %) + our nl5V3 stage
# then eval_free.py (no rungs) on all. GPU: gpu_lock shared (training) / tiny (stages).
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
P=""; for t in mocap_square2fast mocap_figfast mocap_square2fastM mocap_figfastM mocap_stadT; do for r in "" _mu80 _mu60; do P="$P ${t}_b0_free$r"; done; done
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --lanes 384 --hidden 256"
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift"
G="mocap_square2fast_b0_free mocap_figfast_b0_free"
ST="--iters 5 --beta 0.5 --delta 0.05 --cap 0.05 --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --eval-runs 4 --eval-laps 5 --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1 --err-scale 0.05 1e9 1e9 1e9 1e9 3.0"
stage() { n=$1; pol=$2; g=$3; [ -f runs/hw/${n}_stage/summary.json ] || { rm -rf runs/hw/${n}_stage
  $L tiny $g $PY -u hw_stage.py --policy $pol --out runs/hw/${n}_stage --cars $CARS --goals $G --anchor-plans $G $ST --gpu $g > logs/hw_${n}_stage.log 2>&1; }; }
for s in 0 1 2; do
  ( [ -f runs/freeNom5_s$s/policy_final.npz ] || OMP_NUM_THREADS=2 $L shared $s $PY -u sim_train.py --plans $P --out runs/freeNom5_s$s $V3 \
        --err-scale 0.05 1e9 1e9 1e9 1e9 0.3 --dr 0 --anchor-w 1.0 --seed $s --gpu $s > logs/freeNom5_s$s.log 2>&1
    stage freeNom5_s${s}_nl5V3 runs/freeNom5_s$s/policy_final.npz $s ) &
  stage freePPO_s${s}_nl5V3 runs/freePPO_s$s/policy_final.npz $s &
done; wait
ARGS=""
for s in 0 1 2; do ARGS="$ARGS freeNom5_s$s $C/runs/freeNom5_s$s/policy_final.npz"
  for n in freeNom5_s${s}_nl5V3 freePPO_s${s}_nl5V3; do [ -f runs/hw/${n}_stage/summary.json ] && ARGS="$ARGS ${n}+stage $C/runs/hw/${n}_stage"; done; done
EVAL_FREE_NO_RUNGS=1 F1T_PROCS=12 $PY -u $EV $ARGS > logs/free_final_arms.log 2>&1
echo "$(date +%T) CAR_FINAL_ARMS_DONE"
