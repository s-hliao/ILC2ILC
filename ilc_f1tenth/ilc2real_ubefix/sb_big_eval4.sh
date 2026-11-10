#!/bin/bash
# sb_big_eval4.sh (2026-11-08 ~00:30): the oval-practised SBBIG runs on the deep-waist peanut (mocap_pnut115w110, lobes 1.15, waist arcs 1.1,
# waist 67 %; a_max 3.0, lap mean 1.90 m/s) in BOTH directions (transfer measured direction-dependent), its 0-4, 5 laps, seed 701, 12 cars.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
until [ -f logs/sbbig/DONE3 ]; do sleep 30; done
export LADDER_ITS="0 1 2 3 4" F1T_PROCS=30 LADDER_CARS="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
for pl in mocap_pnut115w110_b0_free_a30 mocap_pnut115w110CW_b0_free_a30; do
  LADDER_OUT=sb_$pl.json LADDER_LAPS=5 LADDER_PLANS=$pl taskset -c 0-31 nice -n 5 ~/miniconda3/envs/f1t/bin/python -u ladder_eval.py runs/screen/SBBIG_plastic_* >> logs/sbbig/eval4.log 2>&1
done
touch logs/sbbig/DONE4
