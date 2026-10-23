#!/bin/bash
# run_remaining.sh (flows, car; staged 2026-10-22 for the bigger machine): everything of the car flow analysis that was NOT
# run on the original machine. Idempotent: every step skips what already exists, so it can also just finish a partial copy.
#   1. the frozen hardware-stage recipe (car_redo.sh / car_redo_rma.sh's LAD, unchanged) with the append-only --save-its 1
#      and the stage's own end evaluation on the VALIDATION seed (--eval-seed 301), for the start policies / sim seeds
#      that are missing: ours (v3drBtL), PPO+DR (ppoL_task), RMA (rmaL_task) seeds 1, 2 (+ seeds 0 if not copied over),
#      and the extra nominal-sim start v3nom_s0 (not a frozen arm; it never saw the ladder's rung plans in sim);
#   2. flows_car.py compute on every stage, the phase portraits (s0 of each start: real_mu and ax:mu=0.6), the report.
# Inputs that must exist (absolute paths; copy them over, keeping the layout):
#   /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real/                     the code incl. the uncommitted hw_stage.py (--save-its,
#        RMA support) and ilc_core.py edits; plans_mu02_tight/ (plans + cached _K.npy LQR gains);
#        runs/{v3drBtL,ppoL_task,rmaL_task}_s{0,1,2}/policy_final.npz, runs/rmaL_task_s{0,1,2}/rma_adapt.npz,
#        runs/v3nom_s0/policy_final.npz
#   /home/henry/ilc_ws/src/ilc_f1tenth/lifted_linear_tire_20261009/  tracks_tight/ (incl. mocap_squareH.npz) and mb_sim/
#   /home/henry/ilc_ws/log/dilc/flows/car/                           this directory (flows_car.py; runs/*_ladR_val_its
#        from the first machine, if they are to be reused rather than re-run)
#   python env: ~/miniconda3/envs/f1t (jax CPU, numba, optax, matplotlib)
# Budget knobs: MAXJ stages at once, each 1 main + PROCS workers; FPROCS workers for the flows. Defaults stay inside 8
# processes (MAXJ=2 PROCS=1, FPROCS=4 while stages run). On a big machine e.g. MAXJ=7 PROCS=3 FPROCS=16.
# Runtime measured on the first machine (36 cores at load ~38, single-threaded XLA): one stage ~ see FLOWS.md "Status";
# one flows compute (8 cars x 6 networks x 7 plans) ~ see FLOWS.md.
set -u
OUT=/home/henry/ilc_ws/log/dilc/flows/car
REPO=/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
PY=${PY:-~/miniconda3/envs/f1t/bin/python}
MAXJ=${MAXJ:-2}; PROCS=${PROCS:-1}; FPROCS=${FPROCS:-4}
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
mkdir -p $OUT/runs $OUT/logs $OUT/fields $OUT/figs
G2="--goals mocap_square2fast_b25 mocap_figfast_b25"
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6"
LAD="--iters 5 $G2 --goal-cycle 1 --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1 --fallback-revert 1 --goal-order worst"
stage() {  # name policy
  ( cd $REPO && source env_tight.sh && \
    [ -f $OUT/runs/$1/summary.json ] || F1T_PROCS=$PROCS $PY -u hw_stage.py --policy $2 --out $OUT/runs/$1 $LAD --cars $CARS \
       --eval-seed 301 --save-its 1 > $OUT/logs/hw_$1.log 2>&1 )
}
ALL=""; for s in 0 1 2; do ALL+="v3drBtL_s$s ppoL_task_s$s rmaL_task_s$s "; done; ALL+="v3nom_s0"
# 1. the stages (MAXJ at once)
for n in $ALL; do
  [ -f $OUT/runs/${n}_ladR_val_its/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do wait -n; done
  stage ${n}_ladR_val_its $REPO/runs/$n/policy_final.npz &
  echo "$(date '+%F %T') stage $n launched"
done
wait
echo "$(date '+%F %T') stages done"
# 2. the flows (one stage at a time, FPROCS workers), the portraits, the report
cd $OUT
for n in $ALL; do
  d=runs/${n}_ladR_val_its
  [ -f $d/summary.json ] || { echo "MISSING $d"; continue; }
  [ -f fields/${n}_ladR_val_its.json ] || $PY -u flows_car.py compute --stage $d --procs $FPROCS > logs/flows_$n.log 2>&1 || echo "FAILED flows $n"
  case $n in *_s0)
    for c in real_mu ax:mu=0.6; do
      [ -f "figs/portrait_${n}_ladR_val_its_$c.png" ] || $PY -u flows_car.py portrait --stage $d --car $c --plan mocap_square2fast_b25 \
          --procs 3 >> logs/portrait_$n.log 2>&1 || echo "FAILED portrait $n $c"
    done;;
  esac
done
$PY flows_car.py report > logs/report.log 2>&1 && cat table.md
echo "$(date '+%F %T') RUN_REMAINING_DONE"
