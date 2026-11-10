#!/bin/bash
# yaw_test_lambda.sh (moved from matcha 02:20 for load; Lambda cores 26-45) = yaw_test.sh (2026-11-08 ~02:40, from matcha; user: "you should still test the yaw rate term", rubber fig8 stays _x120 at 3.1 m/s):
# hw_stage_yaw.py --yaw-path 1.0 (cost row 4 = r - V kappa(s), scale 1 rad/s fixed beforehand) vs the PRACB twins (same seeds, --skip-below
# 0.5, no yaw term). Cells: rubber fig8_x120, rubber square_x120, plastic square nominal, plastic fig8 nominal (added 02:20: user "get it working no matter what"; pass there = better score than its twin) x freePPON_s0 / rmaN_s0, stage seed 9001.
# PRE-REGISTERED PASS (all three): rubber fig8 fewer crashed cars or better score than its twin; rubber square <= 3 points worse; plastic
# square <= 3 points worse AND rear-out share (signed beta < -3 deg) at it3 within 10 points of the twin's. Out runs/screen/YAW_*.
cd ~/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=~/miniconda3/envs/f1t/bin/python; mkdir -p logs/yaw
PC='n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000'
RC='ax:mu=1.667 ax:mu=1.833,mass=1.1'
one() {  # prot track net procs
  local CARS G A
  case $1/$2 in
    rubber/fig8)    CARS=$RC; G=mocap_fig8R100_b0_free_x120;    A='mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120';;
    rubber/square)  CARS=$RC; G=mocap_square2fast_b0_free_x120; A='mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120';;
    plastic/square) CARS=$PC; G=mocap_square2fast_b0_free;      A='mocap_square2fast_b0_free mocap_figfast_b0_free';;
    plastic/fig8)   CARS=$PC; G=mocap_fig8R100_b0_free;         A='mocap_fig8R100_b0_free mocap_square2fast_b0_free';;
  esac
  local D=runs/screen/YAW_$1_$2_$3_s0_ss9001 X=''; [ $3 = rmaN ] && X='--z-path 1'
  F1T_PROCS=$4 taskset -c 26-45 nice -n 5 $PY -u hw_stage_yaw.py --policy runs/$3_s0/policy_final.npz --out $D --cars $CARS --goals $G --anchor-plans $A \
    --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
    --iters 4 --beta 0.5 --delta 0.05 --cap 0.4 --fit-stall 0 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 --yaw-path 1.0 \
    --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
    --eval-seed 701 --seed 9001 --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-below 0.5 --save-its 1 $X > logs/yaw/$(basename $D).log 2>&1
  [ -f $D/summary.json ] || return
  local CB="$(echo $CARS | tr ' ' '|')"
  LADDER_ITS='0 1 2 3 4' LADDER_OUT=ladder_launch.json LADDER_LAPS=5 LADDER_PLANS=$G LADDER_CARS="$CB" F1T_PROCS=$4 taskset -c 26-45 nice -n 5 $PY -u ladder_eval_launch.py $D >> logs/yaw/$(basename $D).log 2>&1
  LADDER_ITS='0 1 2 3 4' LADDER_OUT=sb_practised.json LADDER_LAPS=5 LADDER_PLANS=$G LADDER_CARS="$CB" F1T_PROCS=$4 taskset -c 26-45 nice -n 5 $PY -u ladder_eval.py $D >> logs/yaw/$(basename $D).log 2>&1
}
for n in freePPON rmaN; do one rubber fig8 $n 2 & one rubber square $n 2 & one plastic square $n 5 & one plastic fig8 $n 5 & done; wait
touch logs/yaw/DONE
