#!/bin/bash
# zs_oval_sweep.sh (2026-11-08 ~08:15, OVAL_RULES_1108.md rule 1): zero-shot oval165 speed sweep, 12 plastic cars, both s0 networks.
cd ~/ilc_ws/src/ilc_f1tenth/ilc2real; PY=~/miniconda3/envs/f1t/bin/python
PC='n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000'
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
LADDER_ITS=0 F1T_PROCS=8 LADDER_CARS="$(echo $PC | tr ' ' '|')" LADDER_LAPS=5 LADDER_OUT=zs_oval.json \
  LADDER_PLANS='mocap_oval165_b0_free_a26 mocap_oval165_b0_free_a28 mocap_oval165_b0_free_a30 mocap_oval165_b0_free_a34 mocap_oval165_b0_free_a40' \
  taskset -c 20-35 nice -n 5 $PY -u ladder_eval_launch.py runs/beh2/ZS_plastic_freePPON_s0 runs/beh2/ZS_plastic_rmaN_s0 > logs/zs_oval/sweep.log 2>&1
touch logs/zs_oval/DONE
