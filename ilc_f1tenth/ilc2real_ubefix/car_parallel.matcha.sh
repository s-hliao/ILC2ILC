#!/bin/bash
# car_parallel.sh (2026-10-30, matcha; user: run every step of the plan in parallel, deploy this week):
#  (a) PPO ON THE CAR (ppo_real_car.py: privileged critic, true-state reward, per-car actor) from PPO+DR freePPOV_s<k>, the stage's two
#      _v08 tracks, the 14 validation cars; checkpoints 10 / 20 (= our stage's laps per track) and 40 / 80 / 160 episodes per car
#  (b) friction DR widened to x0.5-1.2 (F1T_DR_MU "0.5 1.2"; the default covers x0.8-1.2, the x0.6 / x0.7 cars were outside it), the SAME
#      for ours (freeDR5V's recipe -> freeDRw5V_s<k>) and PPO + DR (freePPO5V's recipe from freeBC5V -> freePPOw5V_s<k>), seeds 0-2
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
PY=~/miniconda3/envs/f1t_ube/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=3 GPU_SLOTS=3
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift ax:mu=4.5 ax:mu=5.5,mass=1.1"
G="mocap_square2fast_b0_free_v08 mocap_figfast_b0_free_v08"
T5="mocap_square2fast mocap_figfast mocap_square2fastM mocap_figfastM mocap_stadT"
P5=""; for t in $T5; do for r in _v08 _v09 ""; do P5="$P5 ${t}_b0_free$r"; done; done
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --hidden 256"
g=1
for s in 0 1 2; do
  [ -f runs/hw/ppoCarV_s${s}_ep160/summary.json ] || ( $L shared $g $PY -u ppo_real_car.py --init runs/freePPOV_s$s/policy_final.npz --arm ppoCarV_s$s \
      --cars $CARS --goals $G --per-iter 10 --checkpoints 10 20 40 80 160 --seed $((4242 + s)) --gpu $g > logs/ppoCarV_s$s.log 2>&1 ) &
  g=$(( g % 3 + 1 )); sleep 30
done
export F1T_DR_MU="0.5 1.2"
for s in 0 1 2; do
  [ -f runs/freeDRw5V_s$s/policy_final.npz ] || ( $L shared $g $PY -u sim_train.py --plans $P5 --out runs/freeDRw5V_s$s $V3 $TASK --lanes 384 \
      --dr 1.0 --anchor-w 1.0 --seed $s --gpu $g > logs/freeDRw5V_s$s.log 2>&1 ) &
  g=$(( g % 3 + 1 )); sleep 30
  [ -f runs/freePPOw5V_s$s/policy_final.npz ] || ( $L shared $g $PY -u ppo_dr.py --plans $P5 --out runs/freePPOw5V_s$s --init runs/freeBC5V_s0/policy_final.npz \
      --mode ppo --dr 1.0 --iters 300 --lanes 1024 --reward task --seed $s --gpu $g > logs/freePPOw5V_s$s.log 2>&1 ) &
  g=$(( g % 3 + 1 )); sleep 30
done
wait
echo "$(date +%T) CAR_PARALLEL_DONE"
