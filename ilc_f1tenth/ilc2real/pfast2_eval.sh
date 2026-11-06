#!/bin/bash
# pfast2_eval.sh (2026-11-06, ube): the PFAST2 runs' evaluation (PFAST_RULES.md): ladder_eval.py on each run's PRACTISED plan, 5 laps,
# it 0 vs 3, that tire's cars -> pfast2_eval.json
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; PY=~/miniconda3/envs/f1t/bin/python
PC='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000'
ev() {  # glob plan cars
  LADDER_OUT=pfast2_eval.json LADDER_LAPS=5 F1T_PROCS=32 LADDER_PLANS=$2 LADDER_CARS="$3" $PY -u ladder_eval.py runs/screen/$1 >> logs/pfast2_eval.log 2>&1
}
ev "PFAST2_plastic_sq120_*" mocap_square2fast_b0_free_a45_x120 "$PC"
ev "PFAST2_plastic_pea120_*" mocap_peanutM_b0_free_a33_x120 "$PC"
ev "PFAST2_rubber_f8x140_*" mocap_fig8R100_b0_free_x140 'ax:mu=1.667|ax:mu=1.833,mass=1.1'
ev "PFAST2_rubber_peax150_*" mocap_peanutM_b0_free_x150 'ax:mu=1.667|ax:mu=1.833,mass=1.1'
echo PFAST2_EVAL_DONE >> logs/pfast2_eval.log
