#!/bin/bash
# car_zp_queue.sh (ube, CAR_SIGNIF_1106.md B): RMA with --z-path 1, as A's RMA runs (iters 8, seeds 0-2, ss9001, both tires), sharing
# A's cap of 6 stage runs at once; then the held-out peanut evaluation at iterations 0 3 5 8 -> long_eval.json (as A)
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
for tire in plastic rubber; do for s in 0 1 2; do
  while [ $(pgrep -fc "taskset -c 0-30 .*hw_stage.py|hw_stage.py --policy runs/.*--out runs/screen/(LONG|ZP)_") -ge 6 ]; do sleep 30; done
  bash car_long.sh $tire rmaN $s 9001 8 ZP --z-path 1 & sleep 20
done; done
wait
PY=~/miniconda3/envs/f1t/bin/python
PC='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000'
LADDER_ITS="0 3 5 8" LADDER_OUT=long_eval.json LADDER_LAPS=5 F1T_PROCS=12 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" taskset -c 0-30 $PY -u ladder_eval.py runs/screen/ZP_plastic_* > logs/zp_eval_plastic.log 2>&1
LADDER_ITS="0 3 5 8" LADDER_OUT=long_eval.json LADDER_LAPS=5 F1T_PROCS=12 LADDER_PLANS=mocap_peanutM_b0_free_x120 taskset -c 0-30 $PY -u ladder_eval.py runs/screen/ZP_rubber_* > logs/zp_eval_rubber.log 2>&1
echo ZP_DONE >> logs/zp_eval_rubber.log
