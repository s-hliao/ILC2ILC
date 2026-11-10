#!/bin/bash
# car_redo_rma.sh (2026-10-18): the missing row of car_redo.sh's comparison -- RMA (task reward) + OUR 10-lap ladder hardware
# stage, the identical recipe as v3drBtL_s<k>_ladR_lap10 / ppoL_task_s<k>_ladR_lap10 (LAD below), on the 8 cars, scored on the
# VALIDATION reeval seed 301 (tag _val) as car_redo.sh scored the other arms.
#   hw_stage.py trains the RMA ACTOR pi(o, z_hat) only; the adaptation module phi(history) -> z_hat (runs/rmaL_task_s<k>/
#   rma_adapt.npz) stays frozen and runs in every trial / evaluation, and is copied next to the per-car networks so reeval
#   loads them as RMA. z_hat enters the ILC as a measured, fixed input (K = d pi / d o at the measured z_hat).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PROCS=${PROCS:-3}
G2="--goals mocap_square2fast_b25 mocap_figfast_b25"
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6"
LAD="--iters 5 $G2 --goal-cycle 1 --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1 --fallback-revert 1 --goal-order worst"   # = car_redo.sh's LAD
arm() {  # name policy args gpu
  [ -f runs/hw/$1/summary.json ] || F1T_PROCS=$PROCS $PY -u hw_stage.py --policy $2 --out runs/hw/$1 $3 --cars $CARS --gpu $4 > logs/hw_$1.log 2>&1
}
for s in 0 1 2; do
  arm rmaL_task_s${s}_ladR_lap10 runs/rmaL_task_s$s/policy_final.npz "$LAD" $((s + 1)) &
done
wait
echo "$(date '+%F %T') arms done"
for s in 0 1 2; do
  JAX_PLATFORMS=cpu F1T_PROCS=$((3 * PROCS)) $PY -u reeval.py rmaL_task_s${s}_ladR_lap10 --seed 301 --tag _val >> logs/reeval_redo_rma_val.log 2>&1
done
$PY table_redo.py --tag _val > runs/hw/redo_val_table.txt
echo "$(date '+%F %T') CAR_REDO_RMA_DONE"
