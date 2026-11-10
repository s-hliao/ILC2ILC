#!/bin/bash
# sb_big_drift.sh (2026-11-08 ~01:00; user: "with plastic the car would drift given the corrections"): sb_big.sh with the practice oval at a_max 4.0 (lap mean 2.86 m/s, corners ~2.57 = past plastic grip: drift by speed, track-only reference), anchors the a40 oval + square2fast_a45; held out = the deep peanut a34 both directions.
# room-sized tracks (make_big_tracks.py): practise the oval CCW mocap_oval165_b0_free_a26 (a_max 2.6: corners ~2.0 m/s, lap mean 2.29),
# anchors the oval + square2fast_a45 (as before: the practised plan + one training-track plan), 12 plastic sim cars, SB flags unchanged,
# 4 iterations. Then ladder_eval (its 0-4, 5 laps, seed 701): held-out = the clockwise bean mocap_pnut115w110CW_b0_free_a34, practised = the oval.
# freePPON_s0 / rmaN_s0 (the hardware networks) x stage seeds 9001 9002, 8 workers each (32 cores).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=8 JAX_PLATFORMS=cpu
PY=~/miniconda3/envs/f1t/bin/python
PC="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
G=mocap_oval165_b0_free_a40; ANC="mocap_oval165_b0_free_a40 mocap_square2fast_b0_free_a45"; mkdir -p logs/sbdrift
for n in freePPON rmaN; do for ss in 9001 9002; do
  D=runs/screen/SBDRIFT_plastic_${n}_s0_ss$ss; [ -f $D/summary.json ] && continue
  X=""; [ $n = rmaN ] && X="--z-path 1"
  taskset -c 0-31 $PY -u hw_stage.py --policy runs/${n}_s0/policy_final.npz --out $D --cars $PC --goals $G --anchor-plans $ANC \
    --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
    --iters 4 --beta 0.5 --delta 0.05 --cap 0.4 --fit-stall 0 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
    --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
    --eval-seed 701 --seed $ss --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 $X \
    > logs/sbdrift/$(basename $D).log 2>&1 &
done; done; wait
touch logs/sbdrift/STAGE_DONE
PCB=$(echo $PC | tr " " "|"); export LADDER_ITS="0 1 2 3 4" F1T_PROCS=30 LADDER_CARS="$PCB"
LADDER_OUT=sb_heldout.json LADDER_LAPS=5 LADDER_PLANS=mocap_pnut115w110CW_b0_free_a34 taskset -c 0-31 nice -n 5 $PY -u ladder_eval.py runs/screen/SBDRIFT_plastic_* > logs/sbdrift/eval.log 2>&1
LADDER_OUT=sb_practised.json LADDER_LAPS=5 LADDER_PLANS=mocap_oval165_b0_free_a40 taskset -c 0-31 nice -n 5 $PY -u ladder_eval.py runs/screen/SBDRIFT_plastic_* >> logs/sbdrift/eval.log 2>&1
LADDER_OUT=sb_pnut_ccw.json LADDER_LAPS=5 LADDER_PLANS=mocap_pnut115w110_b0_free_a34 taskset -c 0-31 nice -n 5 $PY -u ladder_eval.py runs/screen/SBDRIFT_plastic_* >> logs/sbdrift/eval.log 2>&1
touch logs/sbdrift/DONE
