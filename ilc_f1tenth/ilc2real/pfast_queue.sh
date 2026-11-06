#!/bin/bash
# pfast_queue.sh (2026-11-05, runs/screen/PFAST_RULES.md): 12 fast-peanut practise runs on ube (one job per GPU, 10 workers each), then the
# evaluation on the practised plan (5 laps, it 0 vs 3) on the CPUs.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real
J=(); for tire in plastic rubber; do for n in freePPON rmaN; do for s in 0 1 2; do J+=("$tire $n $s"); done; done; done
i=0; while [ $i -lt ${#J[@]} ]; do
  for g in 0 1 2; do
    [ $i -lt ${#J[@]} ] || break
    bash pfast_stage.sh ${J[$i]} 9001 $g & i=$((i + 1))
  done
  wait
done
PY=~/miniconda3/envs/f1t/bin/python
LADDER_OUT=pfast_eval.json LADDER_LAPS=5 F1T_PROCS=32 LADDER_PLANS=mocap_peanutM_b0_free_a33_x110 \
  LADDER_CARS='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000' \
  $PY -u ladder_eval.py runs/screen/PFAST_plastic_* > logs/pfast_eval_plastic.log 2>&1
LADDER_OUT=pfast_eval.json LADDER_LAPS=5 F1T_PROCS=32 LADDER_PLANS=mocap_peanutM_b0_free_x130 $PY -u ladder_eval.py runs/screen/PFAST_rubber_* > logs/pfast_eval_rubber.log 2>&1
echo ALL_DONE
