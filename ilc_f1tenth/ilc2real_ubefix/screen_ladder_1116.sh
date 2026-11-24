#!/bin/bash
# screen_ladder_1116.sh CASE NET SS (2026-11-16 speed ladder, user: "do the car experiments we talked about"): identical stage
# settings to screen_hwplan_1116b.sh (BIGSTEP), only the goal plan's speed scale changes.
#   rubber stand-in mu x1.667: peanut x125 / x130 / x135      plastic stand-in mu x0.4: fig8R115 x095
c=$1 n=$2 ss=$3; cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4 CUDA_VISIBLE_DEVICES=${GPU:-0}
case $c in
  rubberpeanutx1[23]?) CAR=ax:mu=1.667; G=mocap_peanutM_b0_free_${c#rubberpeanut}; A="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120";;
  plasticfig8x095) CAR=ax:mu=0.4; G=mocap_fig8R115_b0_free_x095; A="mocap_fig8R115_b0_free_x095 mocap_fig8R115_b0_free";;
  *) echo "usage: CASE rubberpeanutx125|x130|x135|plasticfig8x095 NET SS"; exit 1;;
esac
D=runs/screen/HW1116L_${c}_${n}_s0_ss$ss
[ -f $D/summary.json ] && exit 0
PYTHONFAULTHANDLER=1 ~/miniconda3/envs/f1t/bin/python -u hw_stage.py --policy runs/${n}_s0/policy_final.npz --out $D --cars $CAR \
  --goals $G --anchor-plans $A \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.8 --fit-realize 0.9 --fit-max-steps 8000 --fit-stall 0 --crash-rollback 1 --slope-w 1.0 \
  --iters 4 --beta 0.5 --delta 0.05 --cap 0.8 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 1.0 \
  --goal-cycle 1 --trial-laps 2 --anchor-w 10 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed $ss --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 \
  > logs/hw1116L_${c}_${n}_ss$ss.log 2>&1
echo "done $c $ss rc=$?"
