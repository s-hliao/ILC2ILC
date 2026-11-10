#!/bin/bash
# sb_eval.sh (2026-11-06, runs/screen/SB_RERUN_1106.md): every 10 min, score the FINISHED SB runs (summary.json present) as the main tables
# (ladder_eval.py, fixed simulator, F1T_SPIN_RHO=3, its 0-4); until SB_STAGE_DONE and nothing left. Writes sb_<plan-tag>.json per run.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; PY=~/miniconda3/envs/f1t/bin/python
PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
TC=$($PY -c "import mb_car as m; print('|'.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
export F1T_SPIN_RHO=3 LADDER_ITS="0 1 2 3 4" F1T_PROCS=${F1T_PROCS:-8}
fin() { for d in runs/screen/SB_$1_*; do [ -f $d/summary.json ] && [ ! -f $d/$2 ] && echo $d; done; }
ev() {  # protocol out plan laps cars
  local L=$(fin $1 $2); [ -z "$L" ] && return
  E="LADDER_OUT=$2 LADDER_LAPS=$4 LADDER_PLANS=$3"; [ -n "$5" ] && E="$E LADDER_CARS=$5"
  env $E taskset -c 0-29 nice -n 5 $PY -u ladder_eval.py $L >> logs/sb/eval.log 2>&1
}
while true; do
  stop=0; [ -f logs/sb/SB_STAGE_DONE ] && stop=1
  ev plastic sb_heldout.json mocap_peanutM_b0_free_a33 5 "$PC"
  ev rubber sb_heldout.json mocap_peanutM_b0_free_x120 5
  ev tier sb_heldout.json mocap_peanutM_b0_free 5 "$TC"
  ev ethzp sb_practised.json ethzmobil43_b0_free_a33 1 "$PC"; ev ethzp sb_heldout.json mocap_peanutM_b0_free_a33 5 "$PC"
  ev ethzr sb_practised.json ethzmobil43_b0_free 1;           ev ethzr sb_heldout.json mocap_peanutM_b0_free_x120 5
  ev ethzr110 sb_practised.json ethzmobil43_b0_free_x110 1;   ev ethzr110 sb_heldout.json mocap_peanutM_b0_free_x120 5
  ev peax150 sb_practised.json mocap_peanutM_b0_free_x150 5
  [ $stop = 1 ] && break
  sleep 600
done
echo SB_EVAL_DONE >> logs/sb/eval.log
