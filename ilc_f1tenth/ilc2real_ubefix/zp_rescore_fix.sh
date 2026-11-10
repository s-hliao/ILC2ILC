#!/bin/bash
# zp_rescore_fix.sh (2026-11-06): the z-path RMA runs (ZPH / ZPE, practised on the old simulator in ../ilc2real) re-scored on the FIXED
# simulator (this copy: slip floor, F1T_SPIN_RHO=3), after the z-path queue's own evaluation; rescore_*.json next to the old files.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix
R=/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real/runs/screen
until grep -q ZP_MAIN_DONE ../ilc2real/logs/zp_main_eval.log 2>/dev/null; do sleep 60; done
PY=~/miniconda3/envs/f1t/bin/python
PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_SPIN_RHO=3
E() { taskset -c 0-29 env "$@"; }
E LADDER_OUT=rescore_peanut.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py $R/ZPH_plastic_* $R/ZPE_plastic_* > logs/zp_rescore_fix.log 2>&1
E LADDER_OUT=rescore_peanut.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py $R/ZPH_rubber_* $R/ZPE_rubber_* >> logs/zp_rescore_fix.log 2>&1
E LADDER_OUT=rescore_ethz.json LADDER_LAPS=1 F1T_PROCS=24 LADDER_PLANS=ethzmobil43_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py $R/ZPE_plastic_* >> logs/zp_rescore_fix.log 2>&1
E LADDER_OUT=rescore_ethz.json LADDER_LAPS=1 F1T_PROCS=24 LADDER_PLANS=ethzmobil43_b0_free $PY -u ladder_eval.py $R/ZPE_rubber_* >> logs/zp_rescore_fix.log 2>&1
echo ZP_RESCORE_DONE >> logs/zp_rescore_fix.log
