#!/bin/bash
# run_all.sh (2026-10-22): the whole flow analysis (SPEC.md) on the bigger machine, idempotent (every step skips what
# exists). Read ../../../src/paper/HANDOFF_2026-10-22.md §2 first. Paths: absolute /home/henry/ilc_ws.
#   pogo  pogo/run_remaining.sh   (CPU; NJ workers)                       ~3 h on loaded matcha, much less idle
#   car   car/run_remaining.sh    (CPU; MAXJ stages x PROCS, FPROCS flows) the 7 missing stages ~1.3 h each + flows
#   quad  quad/flows_quad.py fly -> compute -> aggregate -> portraits      (container ilc_quad for fly, a GPU for compute)
#   then  merge_flows.py -> fig_flows_all.png (+ the table it prints)
# The three domains are independent: they run in parallel here (DOMAINS="pogo quad" bash run_all.sh for a subset).
# Knobs for a big machine: NJ=16 (pogo), MAXJ=7 PROCS=3 FPROCS=16 (car), QPAR=4 DILC_FLY_JOBS=14 QGPU=1 (quad).
F=/home/henry/ilc_ws/log/dilc/flows
DOMAINS=${DOMAINS:-"pogo car quad"}
mkdir -p $F/logs
for d in $DOMAINS; do
  case $d in
    pogo) ( NJ=${NJ:-8} bash $F/pogo/run_remaining.sh > $F/logs/run_all_pogo.log 2>&1 ) & ;;
    car)  ( MAXJ=${MAXJ:-2} PROCS=${PROCS:-1} FPROCS=${FPROCS:-4} bash $F/car/run_remaining.sh > $F/logs/run_all_car.log 2>&1 ) & ;;
    quad) ( cd $F/quad
            export ILC_QUAD_OVERLAY=/ilc_ws/log/dilc/plane/fixbox/overlay JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_ilc \
                   XLA_PYTHON_CLIENT_PREALLOCATE=false DILC_FLY_JOBS=${DILC_FLY_JOBS:-8}
            PY=~/miniconda3/envs/ilcmjx/bin/python
            $PY -u flows_quad.py fly --par ${QPAR:-3} &&
            $PY -u flows_quad.py compute --gpu ${QGPU:-1} &&
            $PY -u flows_quad.py aggregate &&
            for rb in real_r4 real_r5; do
              $PY -u flows_quad.py portraits --portrait-start "ours+DR" --portrait-robot $rb --portrait-goal 0.6 0.1
              $PY -u flows_quad.py portraits --portrait-start "ours+DR on clean twins" --portrait-robot ${rb}c --portrait-goal 0.6 0.1
            done ) > $F/logs/run_all_quad.log 2>&1 & ;;
  esac
done
wait
~/miniconda3/envs/ilcmjx/bin/python $F/merge_flows.py | tee $F/logs/merge.log
echo "$(date '+%F %T') RUN_ALL_DONE"
