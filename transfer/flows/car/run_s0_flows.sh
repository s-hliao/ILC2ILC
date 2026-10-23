#!/bin/bash
# run_s0_flows.sh: once the three seed-0 stages have finished, flows_car.py compute on them in parallel (H 20, tube stride
# 120 periods to fit ~1 h here; 2 + 2 + 1 workers = 8 processes), then the report.
cd /home/henry/ilc_ws/log/dilc/flows/car
PY=~/miniconda3/envs/f1t/bin/python
export JAX_PLATFORMS=cpu NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
for n in v3drBtL ppoL_task rmaL_task; do
  while [ ! -f runs/${n}_s0_ladR_val_its/summary.json ]; do sleep 30; done
done
echo "$(date '+%F %T') stages done"
i=0
for n in v3drBtL ppoL_task rmaL_task; do
  p=2; [ $n = rmaL_task ] && p=1
  $PY -u flows_car.py compute --stage runs/${n}_s0_ladR_val_its --H 20 --stride 120 --procs $p > logs/flows_${n}_s0.log 2>&1 &
done
wait
$PY flows_car.py report > logs/report.log 2>&1
echo "$(date '+%F %T') S0_FLOWS_DONE"
