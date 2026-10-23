#!/bin/bash
# run_flows.sh: flows_car.py compute on each re-run stage as soon as it has finished (summary.json), then the phase
# portraits (s0 of each start policy: real_mu = representative, ax:mu=0.6 = outside the DR family) and the report.
cd /home/henry/ilc_ws/log/dilc/flows/car
PY=~/miniconda3/envs/f1t/bin/python
export JAX_PLATFORMS=cpu NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
P=${PROCS:-4}
ST=""; for s in 0 1 2; do ST+="v3drBtL_s$s ppoL_task_s$s rmaL_task_s$s "; done; ST+="v3nom_s0"
for n in $ST; do
  d=runs/${n}_ladR_val_its
  while [ ! -f $d/summary.json ]; do sleep 60; done
  if [ ! -f fields/${n}_ladR_val_its.json ]; then
    $PY -u flows_car.py compute --stage $d --procs $P > logs/flows_$n.log 2>&1 || echo "$(date '+%F %T') FAILED $n"
  fi
  echo "$(date '+%F %T') fields $n"
  case $n in *_s0)
    for c in real_mu ax:mu=0.6; do
      $PY -u flows_car.py portrait --stage $d --car $c --plan mocap_square2fast_b25 --procs 3 >> logs/portrait_$n.log 2>&1 || echo "$(date '+%F %T') PORTRAIT FAILED $n $c"
    done;;
  esac
  $PY flows_car.py report > logs/report.log 2>&1
done
echo "$(date '+%F %T') FLOWS_DONE"
