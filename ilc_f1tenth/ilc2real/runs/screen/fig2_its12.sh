#!/bin/bash
# fig2_its12.sh (2026-11-06, figure 2): the main runs' iterations 1 and 2 (2 / 4 real laps) on the fixed simulator, as rescore_fix.sh
# scored 0 and 3 -> rescore12_peanut.json
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; PY=$HOME/miniconda3/envs/f1t/bin/python
PC='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000'
export F1T_SPIN_RHO=3 F1T_PROCS=3 LADDER_ITS="1 2"
LADDER_OUT=rescore12_peanut.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" nice -n 5 $PY -u ladder_eval.py runs/screen/PLSL* > logs/fig2_its12.log 2>&1
LADDER_OUT=rescore12_peanut.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 nice -n 5 $PY -u ladder_eval.py runs/screen/SL50x120* >> logs/fig2_its12.log 2>&1
touch logs/FIG2_ITS12_DONE
