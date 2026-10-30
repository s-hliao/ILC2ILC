#!/bin/bash
# car_grip_probe.sh (2026-10-30; user: the real car runs plastic (mu 0.2) AND rubber tires; both must converge in a few
# laps): the plan-tracking LQR crashes every car with friction >= x2.5 within 0.2 laps (the mu 0.2 plans are power slides
# with ~40 % wheel spin; with grip the car runs to the wheel speed). Probe, seed 0: the DR family with the friction axis
# widened to cover both tires (sim_jax.dr_params, F1T_DR_MU "0.8 5.0": x0.8..x5 log-uniform; every other axis as the DR
# family), the SAME for ours and PPO:
#   freeDRg5_s0   ours + wide DR (freeDR5's recipe)     freePPOg5_s0   PPO + wide DR from freeBC5_s0 (freePPO5's recipe)
# then zero-shot (validation seed 301) on the hardware pair: plastic (real_nom, ax:mu=0.8) and rubber (ax:mu=4.0, ax:mu=5.0).
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs
C=$PWD; PY=~/miniconda3/envs/f1t/bin/python; L=/home/henry/ilc_ws/log/gpu_lock.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=2
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
until [ -f runs/freeBC5_s0/policy_final.npz ]; do sleep 30; done
P5=""; for t in mocap_square2fast mocap_figfast mocap_square2fastM mocap_figfastM mocap_stadT; do for r in "" _mu80 _mu60; do P5="$P5 ${t}_b0_free$r"; done; done
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --hidden 256"
export F1T_DR_MU="0.8 5.0"
[ -f runs/freeDRg5_s0/policy_final.npz ] || $L shared 2 $PY -u sim_train.py --plans $P5 --out runs/freeDRg5_s0 $V3 $TASK --lanes 384 \
    --dr 1.0 --anchor-w 1.0 --seed 0 --gpu 2 > logs/freeDRg5_s0.log 2>&1 &
[ -f runs/freePPOg5_s0/policy_final.npz ] || $L shared 2 $PY -u ppo_dr.py --plans $P5 --out runs/freePPOg5_s0 --init runs/freeBC5_s0/policy_final.npz \
    --mode ppo --dr 1.0 --iters 300 --lanes 1024 --reward task --seed 0 --gpu 2 > logs/freePPOg5_s0.log 2>&1 &
wait
unset F1T_DR_MU
echo "$(date +%T) probe trained"
until [ -f runs/freeDR5_s0/policy_final.npz ] && [ -f runs/freePPO5_s0/policy_final.npz ]; do sleep 30; done
ARGS="lqr_hw lqr"
for n in freeDRg5_s0 freePPOg5_s0 freeNom5_s0 freeDR5_s0 freePPO5_s0 freeBC5_s0; do ARGS="$ARGS ${n}_hwzs $C/runs/$n/policy_final.npz"; done
EVAL_FREE_CARS="real_nom ax:mu=0.8 ax:mu=4.0 ax:mu=5.0" EVAL_FREE_PLAN_SUFFIX=_mu80 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=8 OMP_NUM_THREADS=1 \
  $PY -u $EV $ARGS > logs/grip_probe_zs.log 2>&1
echo "$(date +%T) GRIP_PROBE_DONE"
