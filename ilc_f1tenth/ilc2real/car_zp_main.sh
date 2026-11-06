#!/bin/bash
# car_zp_main.sh (ube, CAR_SIGNIF_1106.md C): the main RMA cells rerun with --z-path 1, at most 6 stage runs at once (JAX CPU, cores 0-29),
# then their evaluations as the main tables.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
J=()
for s in 0 1 2; do for ss in 9001 9002 9003; do J+=("H plastic $s $ss"); done; done
for s in 0 1 2; do for ss in 9001 9002 9003 9004 9005 9006; do J+=("H rubber $s $ss"); done; done
for t in plastic rubber; do for s in 0 1 2; do for ss in 9001 9002 9003; do J+=("E $t $s $ss"); done; done; done
for j in "${J[@]}"; do
  while [ $(pgrep -fc "hw_stage.py --policy runs/rmaN_s[0-9]/policy_final.npz --out runs/screen/ZP[HE]_") -ge 6 ]; do sleep 30; done
  set -- $j
  if [ $1 = H ]; then bash car_long.sh $2 rmaN $3 $4 3 ZPH --z-path 1 &
  else bash ethz_zp.sh $2 rmaN $3 $4 & fi
  sleep 15
done
wait
PY=~/miniconda3/envs/f1t/bin/python
PC='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000'
E() { taskset -c 0-29 env "$@"; }
E LADDER_OUT=ladder_peanut_plastic.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/screen/ZPH_plastic_* > logs/zp_main_eval.log 2>&1
E LADDER_OUT=ladder_peanut.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/screen/ZPH_rubber_* >> logs/zp_main_eval.log 2>&1
E LADDER_OUT=ethz_eval.json LADDER_LAPS=1 F1T_PROCS=24 LADDER_PLANS=ethzmobil43_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/screen/ZPE_plastic_* >> logs/zp_main_eval.log 2>&1
E LADDER_OUT=ethz_eval.json LADDER_LAPS=1 F1T_PROCS=24 LADDER_PLANS=ethzmobil43_b0_free $PY -u ladder_eval.py runs/screen/ZPE_rubber_* >> logs/zp_main_eval.log 2>&1
E LADDER_OUT=peanut_eval.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/screen/ZPE_plastic_* >> logs/zp_main_eval.log 2>&1
E LADDER_OUT=peanut_eval.json LADDER_LAPS=5 F1T_PROCS=24 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/screen/ZPE_rubber_* >> logs/zp_main_eval.log 2>&1
echo ZP_MAIN_DONE >> logs/zp_main_eval.log
