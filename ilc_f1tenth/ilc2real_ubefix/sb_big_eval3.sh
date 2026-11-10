#!/bin/bash
# sb_big_eval3.sh (2026-11-07 ~23:10; user: "what about fig8"): after sb_big_eval2.sh, the oval-practised SBBIG runs on the room-fitting
# figure-eight mocap_fig8R115 (lobes 1.15 m, 2.3 x 4.7 m) at a_max 2.4 (lap mean 1.72 m/s), its 0-4, 5 laps, seed 701, 12 plastic sim cars.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
until [ -f logs/sbbig/DONE2 ]; do sleep 60; done
export LADDER_ITS="0 1 2 3 4" F1T_PROCS=30 LADDER_CARS="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
LADDER_OUT=sb_mocap_fig8R115_b0_free_a24.json LADDER_LAPS=5 LADDER_PLANS=mocap_fig8R115_b0_free_a24 taskset -c 0-31 nice -n 5 ~/miniconda3/envs/f1t/bin/python -u ladder_eval.py runs/screen/SBBIG_plastic_* >> logs/sbbig/eval3.log 2>&1
touch logs/sbbig/DONE3
