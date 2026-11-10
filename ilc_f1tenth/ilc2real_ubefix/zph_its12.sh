#!/bin/bash
# zph_its12.sh (2026-11-06, figure 2): the RMA z-path main runs' iterations 1 and 2 on the fixed simulator -> rescore12_peanut.json
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; PY=~/miniconda3/envs/f1t/bin/python; R=/home/henry/ilc_ws/src/ilc_f1tenth/ilc2real/runs/screen
PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_SPIN_RHO=3 LADDER_ITS="1 2"
E() { taskset -c 0-29 env "$@"; }
E LADDER_OUT=rescore12_peanut.json LADDER_LAPS=5 F1T_PROCS=8 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" nice -n 5 $PY -u ladder_eval.py $R/ZPH_plastic_* > logs/zph_its12.log 2>&1
E LADDER_OUT=rescore12_peanut.json LADDER_LAPS=5 F1T_PROCS=8 LADDER_PLANS=mocap_peanutM_b0_free_x120 nice -n 5 $PY -u ladder_eval.py $R/ZPH_rubber_* >> logs/zph_its12.log 2>&1
echo ZPH_ITS12_DONE >> logs/zph_its12.log
