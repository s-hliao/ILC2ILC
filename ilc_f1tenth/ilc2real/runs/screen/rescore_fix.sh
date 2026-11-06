#!/bin/bash
# rescore_fix.sh (2026-11-06): the plastic cells re-scored on the FIXED simulator (mb_fiala slip floor U_FLOOR 0.1; a spin = yaw rate
# above 3 mu g / |v| for 0.1 s, F1T_SPIN_RHO=3), same evaluation otherwise (ladder_eval.py: seed 701, 4 starts, its 0 and 3).
# Writes new files next to the old ones (rescore_*.json); the old files are kept.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
PY=$HOME/miniconda3/envs/f1t/bin/python
PC='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000'
export F1T_SPIN_RHO=3 F1T_PROCS=10 LADDER_CARS="$PC"
LADDER_OUT=rescore_peanut.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 nice -n 5 $PY -u ladder_eval.py runs/screen/PLSL* > logs/rescore_fix.log 2>&1
LADDER_OUT=rescore_ethz.json LADDER_LAPS=1 LADDER_PLANS=ethzmobil43_b0_free_a33 nice -n 5 $PY -u ladder_eval.py runs/screen/ETHZ_plastic_* >> logs/rescore_fix.log 2>&1
LADDER_OUT=rescore_peanut.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 nice -n 5 $PY -u ladder_eval.py runs/screen/ETHZ_plastic_* >> logs/rescore_fix.log 2>&1
touch logs/RESCORE_FIX_DONE
