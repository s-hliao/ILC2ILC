#!/bin/bash
# eval_its_n6.sh RUN K (2026-10-31; user: "can our stage improve ppo+dr to eventually beat rma? test it"): the plastic evaluation
# (eval_free.py, the 12 plastic cars at mu 0.24-0.4, the _v70 plans, as RMA's ev_N6_rmaN_*_zspl) of the network a long stage
# held after K iterations (runs/hw/RUN/its/it<K>_<car>_policy.npz) -> eval_free/<RUN>_it<K>.json
r=$1 k=$2; cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
src=runs/hw/$r; d=runs/hw/${r}_it$k; mkdir -p $d
for f in $src/its/it${k}_*_policy.npz; do b=$(basename $f); cp $f $d/${b#it${k}_}; done
lab=${r}_it$k
[ -f /home/henry/ilc_ws/log/dilc/car_free/eval_free/$lab.json ] && exit 0
export F1T_MU=0.6 F1T_PLANS=/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu EVAL_FREE_PLAN_SUFFIX=_v70 EVAL_FREE_NO_RUNGS=1 F1T_PROCS=6
export EVAL_FREE_CARS="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
/home/henry/miniconda3/envs/f1t_ube/bin/python -u /home/henry/ilc_ws/log/dilc/car_free/eval_free.py $lab /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real/$d > logs/ev_$lab.log 2>&1
