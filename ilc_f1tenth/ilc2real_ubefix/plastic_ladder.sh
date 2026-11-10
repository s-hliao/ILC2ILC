#!/bin/bash
# plastic_ladder.sh (2026-11-05, runs/screen/SPEED_LADDER.md, plastic section): the plastic peanut ladder on matcha CPUs (5 workers: matcha load <= 34).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
LADDER_OUT=ladder_peanut_plastic.json LADDER_LAPS=5 F1T_PROCS=5 \
  LADDER_PLANS='mocap_peanutM_b0_free_a33 mocap_peanutM_b0_free_a33_x105 mocap_peanutM_b0_free_a33_x110 mocap_peanutM_b0_free_a33_x115 mocap_peanutM_b0_free_a33_x120' \
  LADDER_CARS='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000' \
  ~/miniconda3/envs/f1t/bin/python -u ladder_eval.py runs/screen/PLSL* > logs/ladder_plastic.log 2>&1
echo PLASTIC_DONE >> logs/ladder_plastic.log
