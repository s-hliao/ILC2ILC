#!/bin/bash
# run_stages.sh (flows, 2026-10-22): re-run the car's frozen hardware-stage recipe (car_redo.sh / car_redo_rma.sh's LAD:
# 5 trials x 2 laps, goals = the two b25 plans, ladder _mu80,_mu60, band trigger, revert, worst-first; the 8 cars) for
# every start policy and sim seed, with the append-only --save-its 1 (per-iteration networks) and the stage's own end
# evaluation on the VALIDATION seed (--eval-seed 301). CPU only; at most MAXJ stages at once, each F1T_PROCS workers.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
OUT=/home/henry/ilc_ws/log/dilc/flows/car
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
MAXJ=${MAXJ:-3}; PROCS=${PROCS:-1}
G2="--goals mocap_square2fast_b25 mocap_figfast_b25"
CARS="real_nom real_mass real_mu real_act real_lag ax:mu=0.7 ax:mu=0.9 ax:mu=0.6"
LAD="--iters 5 $G2 --goal-cycle 1 --fallback-suffix _mu80,_mu60 --fallback-on band --fallback-after 1 --fallback-revert 1 --goal-order worst"
arm() {  # name policy
  [ -f $OUT/runs/$1/summary.json ] || F1T_PROCS=$PROCS $PY -u hw_stage.py --policy $2 --out $OUT/runs/$1 $LAD --cars $CARS \
     --eval-seed 301 --save-its 1 > $OUT/logs/hw_$1.log 2>&1
}
JOBS=""
for s in 0 1 2; do JOBS+="v3drBtL_s$s:runs/v3drBtL_s$s/policy_final.npz ppoL_task_s$s:runs/ppoL_task_s$s/policy_final.npz rmaL_task_s$s:runs/rmaL_task_s$s/policy_final.npz "; done
JOBS+="v3nom_s0:runs/v3nom_s0/policy_final.npz"
for j in $JOBS; do
  while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do wait -n; done
  n=${j%%:*}; p=${j#*:}
  arm ${n}_ladR_val_its $p &
  echo "$(date '+%F %T') launched $n"
done
wait
echo "$(date '+%F %T') FLOWS_STAGES_DONE"
