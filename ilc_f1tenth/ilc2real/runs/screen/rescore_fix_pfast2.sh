#!/bin/bash
# rescore_fix_pfast2.sh (2026-11-06): the PFAST2 runs (pfast2_eval.sh's evaluation) on the fixed simulator -> rescore_pfast2.json
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; PY=$HOME/miniconda3/envs/f1t/bin/python
PC='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000'
export F1T_SPIN_RHO=3
ev() { LADDER_OUT=rescore_pfast2.json LADDER_LAPS=5 F1T_PROCS=10 LADDER_PLANS=$2 LADDER_CARS="$3" nice -n 5 $PY -u ladder_eval.py runs/screen/$1 >> logs/rescore_fix_pfast2.log 2>&1; }
ev "PFAST2_rubber_peax150_*" mocap_peanutM_b0_free_x150 'ax:mu=1.667|ax:mu=1.833,mass=1.1'
ev "PFAST2_rubber_f8x140_*" mocap_fig8R100_b0_free_x140 'ax:mu=1.667|ax:mu=1.833,mass=1.1'
ev "PFAST2_plastic_pea120_*" mocap_peanutM_b0_free_a33_x120 "$PC"
ev "PFAST2_plastic_sq120_*" mocap_square2fast_b0_free_a45_x120 "$PC"
touch logs/RESCORE_FIX_PFAST2_DONE
