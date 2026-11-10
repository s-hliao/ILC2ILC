#!/bin/bash
# sb_big_eval2.sh (2026-11-07 ~23:00; user: "make sure that the other held-out tracks actually end up doing well"): after sb_big.sh, the
# SBBIG runs (oval-practised) scored on every OTHER track, its 0-4, 5 laps, seed 701, 12 plastic sim cars: bean CCW, oval CW (direction
# swap), and the old plastic tracks peanut CW _a33, square2fast _a45, fig8R100 _a33. Outputs sb_<plan>.json per run.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=~/miniconda3/envs/f1t/bin/python
until [ -f logs/sbbig/DONE ]; do sleep 60; done
PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export LADDER_ITS="0 1 2 3 4" F1T_PROCS=30 LADDER_CARS="$PC"
for pl in mocap_bean165_b0_free_a26 mocap_oval165CW_b0_free_a26 mocap_peanutM_b0_free_a33 mocap_square2fast_b0_free_a45 mocap_fig8R100_b0_free_a33; do
  LADDER_OUT=sb_$pl.json LADDER_LAPS=5 LADDER_PLANS=$pl taskset -c 0-31 nice -n 5 $PY -u ladder_eval.py runs/screen/SBBIG_plastic_* >> logs/sbbig/eval2.log 2>&1
done
touch logs/sbbig/DONE2
