#!/bin/bash
# car_free_drp.sh (2026-10-26; user: the anchor on hardware only, no line search, no gain penalty): ours + DR, simplified
# -- freeNomNA_s<k> (nominal, no anchor) -> freeDRP_s<k>: DR transfer with NO anchor and no gain penalty (the car's sim
# stage has no line search), 30 its, the v3 recipe otherwise; final snapshot.
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=2
P="mocap_square2fast_b0_free mocap_figfast_b0_free mocap_square2fast_b0_free_mu80 mocap_figfast_b0_free_mu80 mocap_square2fast_b0_free_mu60 mocap_figfast_b0_free_mu60"
TASK="--err-scale 0.05 1e9 1e9 1e9 1e9 0.3"
V3="--beta 0.3 --cap 0.04 --max-ey 0.2 --a-off 0.15 --pert 1.5 --explore-frac 0.7 --eval-every 5 --bc-iters 10 --iters 30 --lanes 192 --hidden 256"
for s in 0 1 2; do
  while [ ! -f runs/freeNomNA_s$s/policy_final.npz ]; do sleep 20; done
  [ -f runs/freeDRP_s$s/policy_final.npz ] || $PY -u sim_train.py --plans $P --out runs/freeDRP_s$s $V3 $TASK --init runs/freeNomNA_s$s/policy_final.npz \
      --dr 1.0 --anchor-w 0 --seed $s --gpu $s > logs/freeDRP_s$s.log 2>&1 &
done; wait
echo "$(date +%T) FREE_DRP_DONE"
