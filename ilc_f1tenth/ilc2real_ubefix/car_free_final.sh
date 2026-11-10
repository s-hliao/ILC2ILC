#!/bin/bash
# car_free_final.sh (2026-10-26): the pogo's final recipe on the car -- freeNomNA_s<k> (nominal, no anchor) -> freeDRF_s<k>:
# DR transfer, no anchor, --gain-w 1e-3 (the car's sim stage has no line search), 30 its, v3 recipe otherwise; then
# zero-shot (eval_free.py) and the hardware stage (ladR, hardware anchor 1.0) -> eval on the original free goals.
cd "$(dirname "$0")"; source env_tight.sh; mkdir -p logs runs/hw
PY=~/miniconda3/envs/f1t/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=2
P="mocap_square2fast_b0_free mocap_figfast_b0_free mocap_square2fast_b0_free_mu80 mocap_figfast_b0_free_mu80 mocap_square2fast_b0_free_mu60 mocap_figfast_b0_free_mu60"
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --lanes 192 --hidden 256"
EV=/home/henry/ilc_ws/log/dilc/car_free/eval_free.py
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.9 ax:mu=0.7 ax:mu=0.6 harsh_combo harsh_steer harsh_sense harsh_drift"
G="mocap_square2fast_b0_free mocap_figfast_b0_free"
LADR="--iters 5 --beta 0.5 --delta 0.05 --cap 0.05 --objective task --goal-cycle 1 --goal-order worst --trial-laps 2 --anchor-w 1.0 \
  --fallback-suffix _mu80,_mu60 --fallback-after 1 --fallback-on band --fallback-ey 0.1 --fallback-revert 1 --eval-runs 4 --eval-laps 5 \
  --eval-seed 301 --seed 9001 --launch 1.0 8.0 --save-its 1"
for s in 0 1 2; do
  [ -f runs/freeDRF_s$s/policy_final.npz ] || $PY -u sim_train.py --plans $P --out runs/freeDRF_s$s $V3 $TASK --init runs/freeNomNA_s$s/policy_final.npz \
      --dr 1.0 --anchor-w 0 --gain-w 1e-3 --seed $s --gpu $s > logs/freeDRF_s$s.log 2>&1 &
done; wait
F1T_PROCS=8 OMP_NUM_THREADS=1 $PY -u $EV freeDRF_s0 runs/freeDRF_s0/policy_final.npz freeDRF_s1 runs/freeDRF_s1/policy_final.npz \
    freeDRF_s2 runs/freeDRF_s2/policy_final.npz > logs/free_zs_drf.log 2>&1
for s in 0 1 2; do
  [ -f runs/hw/freeDRF_s${s}_stage/summary.json ] || OMP_NUM_THREADS=1 $PY -u hw_stage.py --policy runs/freeDRF_s$s/policy_final.npz \
      --out runs/hw/freeDRF_s${s}_stage --cars $CARS --goals $G --anchor-plans $G $LADR > logs/hw_freeDRF_s${s}_stage.log 2>&1 &
done; wait
F1T_PROCS=8 OMP_NUM_THREADS=1 $PY -u $EV freeDRF_s0+stage runs/hw/freeDRF_s0_stage freeDRF_s1+stage runs/hw/freeDRF_s1_stage \
    freeDRF_s2+stage runs/hw/freeDRF_s2_stage > logs/free_real_drf.log 2>&1
echo "$(date +%T) CAR_FREE_FINAL_DONE"
