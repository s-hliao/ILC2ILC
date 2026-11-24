#!/bin/bash
# screen_zs_1119.sh TAG CAR PLAN (2026-11-19, user: "the car experiment ... needs much larger initial deviation"): ZERO-SHOT only
# (1 trial of 2 laps, no update) of freePPON on a candidate "real" car, same hw_stage args as screen_ladder_1116.sh.
t=$1 car=$2 G=$3; cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=2 CUDA_VISIBLE_DEVICES=${GPU:-0}
case $G in *peanut*) A="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120";; *) A="$G mocap_fig8R115_b0_free";; esac
D=runs/screen/ZS1119_$t
[ -f $D/summary.json ] && exit 0
~/miniconda3/envs/f1t/bin/python -u hw_stage.py --policy runs/freePPON_s0/policy_final.npz --out $D --cars $car --goals $G --anchor-plans $A \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.8 --fit-realize 0.9 --fit-max-steps 8000 --fit-stall 0 --crash-rollback 1 --slope-w 1.0 \
  --iters 1 --beta 0.5 --delta 0.05 --cap 0.8 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 1.0 \
  --goal-cycle 1 --trial-laps 2 --anchor-w 10 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed 9001 --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 0 > logs/zs1119_$t.log 2>&1
echo "done $t rc=$?"
