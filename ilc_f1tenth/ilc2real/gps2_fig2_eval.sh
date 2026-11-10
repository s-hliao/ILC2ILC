#!/bin/bash
# gps2_fig2_eval.sh N (2026-11-07): gps_fig2_eval.sh for the faithful rerun GPS2_* (its 0 1 2 3 6 12 24, the panel's held-out plan,
# 5 laps); stops when N of this machine's GPS2_* runs are scored.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; PY=~/miniconda3/envs/f1t/bin/python; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_SPIN_RHO=3 LADDER_ITS="0 1 2 3 6 12 24" F1T_PROCS=${F1T_PROCS:-8} JAX_PLATFORMS=cpu NUMBA_NUM_THREADS=1
PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
TC=$($PY -c "import mb_car as m; print('|'.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
ev() { L=$(for d in runs/screen/GPS2_$1_*; do [ -f $d/summary.json ] && [ ! -f $d/gps_heldout.json ] && echo $d; done); [ -z "$L" ] && return
  E="LADDER_OUT=gps_heldout.json LADDER_LAPS=5 LADDER_PLANS=$2"; [ -n "$3" ] && E="$E LADDER_CARS=$3"
  env $E $PY -u ladder_eval.py $L >> logs/gps2/eval.log 2>&1; }
while true; do
  ev plastic mocap_peanutM_b0_free_a33 "$PC"; ev rubber mocap_peanutM_b0_free_x120; ev tier mocap_peanutM_b0_free "$TC"
  n=$(ls runs/screen/GPS2_*/gps_heldout.json 2>/dev/null | wc -l); [ $n -ge $1 ] && break
  sleep 300
done
echo GPS2_EVAL_DONE >> logs/gps2/eval.log
