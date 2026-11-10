#!/bin/bash
# car_long_queue.sh (ube, CAR_SIGNIF_1106.md A): 12 runs (plastic, rubber x PPO+DR, RMA x seeds 0-2), at most 6 at once, then the held-out
# peanut evaluation at iterations 0 3 5 8 (ladder_eval.py, as the main tables) -> long_eval.json
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
J=(); for tire in plastic rubber; do for n in freePPON rmaN; do for s in 0 1 2; do J+=("$tire $n $s"); done; done; done
for j in "${J[@]}"; do
  while [ $(pgrep -fc "hw_stage.py --policy runs/.*_s[0-9]/policy_final.npz --out runs/screen/LONG_") -ge 6 ]; do sleep 30; done
  set -- $j; bash car_long.sh $1 $2 $3 9001 8 LONG & sleep 20
done
wait
PY=~/miniconda3/envs/f1t/bin/python
PC='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000'
LADDER_ITS="0 3 5 8" LADDER_OUT=long_eval.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" taskset -c 0-30 $PY -u ladder_eval.py runs/screen/LONG_plastic_* > logs/long_eval_plastic.log 2>&1
LADDER_ITS="0 3 5 8" LADDER_OUT=long_eval.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_x120 taskset -c 0-30 $PY -u ladder_eval.py runs/screen/LONG_rubber_* > logs/long_eval_rubber.log 2>&1
echo LONG_DONE >> logs/long_eval_rubber.log
