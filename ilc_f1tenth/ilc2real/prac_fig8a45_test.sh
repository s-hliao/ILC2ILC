#!/bin/bash
# prac_fig8a45_test.sh (2026-11-08 ~01:55, from matcha): ONE SB stage test, rmaN_s0, plastic, practise mocap_fig8R100_b0_free_a45 (fig8 nominal
# does not converge in sim; a33-a45 drive in grip zero-shot) -- does practice at the grip limit induce the slide? prac_queue.sh flags exactly,
# anchors = the practised plan + square2fast nominal (the plastic fig8 pattern). Scored with ladder_eval_launch.py (launched starts) its 0-4.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu F1T_PROCS=4
PY=~/miniconda3/envs/f1t/bin/python
PC='n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000'
D=runs/screen/PRACT_plastic_fig8a45_rmaN_s0_ss9001; G=mocap_fig8R100_b0_free_a45
nice -n 5 $PY -u hw_stage.py --policy runs/rmaN_s0/policy_final.npz --out $D --cars $PC --goals $G --anchor-plans $G mocap_square2fast_b0_free \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
  --iters 4 --beta 0.5 --delta 0.05 --cap 0.4 --fit-stall 0 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
  --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed 9001 --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 --z-path 1 > logs/prac/$(basename $D).log 2>&1
LADDER_CARS="$(echo $PC | tr ' ' '|')" LADDER_LAPS=5 LADDER_ITS='0 1 2 3 4' LADDER_PLANS=$G LADDER_OUT=ladder_launch.json nice -n 5 $PY -u ladder_eval_launch.py $D >> logs/prac/$(basename $D).log 2>&1
touch logs/prac/DONE_fig8a45_test
