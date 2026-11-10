#!/bin/bash
# ethz_x110.sh NET SEED SS GPU (2026-11-06, ETHZ_RULES.md): the rubber ETHZMobil stage at x1.1, the fixed simulator + the yaw-rate spin rule
n=$1 s=$2 ss=$3 g=$4; cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4 CUDA_VISIBLE_DEVICES=$g
EXTRA=""; [ "$n" = rmaN ] && EXTRA="--z-path 1"
D=runs/screen/ETHZX110_rubber_${n}_s${s}_ss$ss
[ -f $D/summary.json ] && exit 0
PYTHONFAULTHANDLER=1 ~/miniconda3/envs/f1t/bin/python -u hw_stage.py --policy runs/${n}_s${s}/policy_final.npz --out $D --cars ax:mu=1.667 ax:mu=1.833,mass=1.1 \
  --goals ethzmobil43_b0_free_x110 --anchor-plans mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120 \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
  --iters 3 --beta 0.5 --delta 0.05 --cap 0.1 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
  --goal-cycle 1 --trial-laps 1 --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed $ss --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 $EXTRA \
  > logs/ethzx110_${n}_s${s}_ss$ss.log 2>&1
