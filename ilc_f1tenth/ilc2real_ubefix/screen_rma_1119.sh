#!/bin/bash
# screen_rma_1119.sh PLAN CAR POLICY TAG SS (2026-11-19: the RMA arm on the new car tracks, --z-path 1 as stage_track_1116.sh does for rma):
G=$1 CAR=$2 POL=$3 n=$4 ss=$5; c=${G#mocap_}; cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4 CUDA_VISIBLE_DEVICES=${GPU:-0}
case $G in *peanut*) A="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120";; *) A="$G mocap_fig8R115_b0_free";; esac
D=runs/screen/RM1119_${c}_${n}_s0_ss$ss
[ -f $D/summary.json ] && exit 0
PYTHONFAULTHANDLER=1 ~/miniconda3/envs/f1t/bin/python -u hw_stage.py --policy $POL --out $D --cars $CAR \
  --goals $G --anchor-plans $A \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.8 --fit-realize 0.9 --fit-max-steps 8000 --fit-stall 0 --crash-rollback 1 --slope-w 1.0 \
  --iters 4 --beta 0.5 --delta 0.05 --cap 0.8 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 1.0 \
  --goal-cycle 1 --trial-laps 2 --anchor-w 10 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed $ss --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 --z-path 1 \
  > logs/rm1119_${c}_${n}_ss$ss.log 2>&1
echo "done $c $ss rc=$?"
