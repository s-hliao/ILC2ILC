#!/bin/bash
# rf_eval.sh (2026-11-06, runs/screen/REFIT_SCREEN_1106.md): every 10 min, score the FINISHED refit-screen runs (summary.json) on the
# held-out peanut _a33, 12 plastic cars, its 0-4 (as sb_eval.sh's plastic), until RF_STAGE_DONE and nothing left.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; PY=~/miniconda3/envs/f1t/bin/python
PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_SPIN_RHO=3 LADDER_ITS="0 1 2 3 4" F1T_PROCS=${F1T_PROCS:-8}
while true; do
  stop=0; [ -f logs/rf/RF_STAGE_DONE ] && stop=1
  L=$(for d in runs/screen/RF_*; do [ -f $d/summary.json ] && [ ! -f $d/sb_heldout.json ] && echo $d; done)
  [ -n "$L" ] && env LADDER_OUT=sb_heldout.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" \
      taskset -c 0-29 nice -n 5 $PY -u ladder_eval.py $L >> logs/rf/eval.log 2>&1
  [ $stop = 1 ] && [ -z "$L" ] && break
  sleep 600
done
echo RF_EVAL_DONE >> logs/rf/eval.log
