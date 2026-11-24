#!/bin/bash
# screen_hwplan_1116b.sh CASE NET SS (2026-11-16, user: "see this one happen on rubber in hardware" + "continuous drift on plastic
# ... figure 8"): the BIGSTEP sim settings of runs/screen/BIG_SB_peax150_freePPON_s0_ss9001_r1667 (the uniform-convergence
# peanut flow; args.json there) on the two hardware candidates:
#   rubberpeanut  peanut x1.2 (x1.5 is infeasible on the real car: 5.55 m/s, 13.5 m/s^2), rubber stand-in mu x1.667
#   plasticfig8   fig8R115 full speed (b0_free: 2.3-3.4 m/s, 6.8 m/s^2 > plastic grip), plastic stand-in mu x0.4
c=$1 n=$2 ss=$3; cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_ubefix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4 CUDA_VISIBLE_DEVICES=${GPU:-0}
case $c in
  rubberpeanut) CAR=ax:mu=1.667; G=mocap_peanutM_b0_free_x120; A="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120";;
  plasticfig8)  CAR=ax:mu=0.4;   G=mocap_fig8R115_b0_free;     A="mocap_fig8R115_b0_free mocap_square2fast_b0_free";;
  plasticfig8x090) CAR=ax:mu=0.4; G=mocap_fig8R115_b0_free_x090; A="mocap_fig8R115_b0_free_x090 mocap_fig8R115_b0_free";;
  *) echo "usage: CASE rubberpeanut|plasticfig8 NET SS"; exit 1;;
esac
EXTRA=""; [ "$n" = rmaN ] && EXTRA="--z-path 1"
D=runs/screen/HW1116_${c}_${n}_s0_ss$ss
[ -f $D/summary.json ] && exit 0
PYTHONFAULTHANDLER=1 ~/miniconda3/envs/f1t/bin/python -u hw_stage.py --policy runs/${n}_s0/policy_final.npz --out $D --cars $CAR \
  --goals $G --anchor-plans $A \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.8 --fit-realize 0.9 --fit-max-steps 8000 --fit-stall 0 --crash-rollback 1 --slope-w 1.0 \
  --iters 4 --beta 0.5 --delta 0.05 --cap 0.8 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 1.0 \
  --goal-cycle 1 --trial-laps 2 --anchor-w 10 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed $ss --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 $EXTRA \
  > logs/hw1116_${c}_${n}_ss$ss.log 2>&1
