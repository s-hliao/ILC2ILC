#!/bin/bash
# pfast_stage.sh TIRE NET SEED STAGESEED GPU (2026-11-05; user: "peanut magnitude ... incredibly small. make it larger, might have to run faster"): practise the PEANUT itself at a faster plan where zero-shot tracks badly but does not spin out (plastic _a33_x110: e_y 10.1 cm; rubber _x130: 13.9 cm; runs/screen/SPEED_LADDER.md), the existing stage settings otherwise (2-lap trials). Rules: runs/screen/PFAST_RULES.md.
# BayesRace ETHZMobil track at x4.3 (ethzmobil43, 55 m, sim only) with the existing stage's exact settings (plastic: PLSL_*, rubber:
# SL50x120*, args.json), except the practised plan and ONE-lap trials (half of the stage's two). plastic: ethzmobil43_b0_free_a33, 12
# plastic cars; rubber: ethzmobil43_b0_free (uncapped; x1.2 spins out 38 % zero-shot), 2 rubber cars. F1T_DEV_TIME=1. Rules:
# runs/screen/ETHZ_RULES.md.
tire=$1 n=$2 s=$3 ss=$4 g=$5; cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 \
       F1T_PROCS=${F1T_PROCS:-10}
if [ "$tire" = plastic ]; then
  CARS="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
  GOAL=mocap_peanutM_b0_free_a33_x110; ANC="mocap_square2fast_b0_free_a45 mocap_figfast_b0_free_a45"
else
  CARS="ax:mu=1.667 ax:mu=1.833,mass=1.1"
  GOAL=mocap_peanutM_b0_free_x130; ANC="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120"
fi
D=runs/screen/PFAST_${tire}_${n}_s${s}_ss$ss
[ -f $D/summary.json ] && exit 0
PYTHONFAULTHANDLER=1 ~/miniconda3/envs/f1t/bin/python -u hw_stage.py --policy runs/${n}_s${s}/policy_final.npz --out $D --cars $CARS \
  --goals $GOAL --anchor-plans $ANC \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
  --iters 3 --beta 0.5 --delta 0.05 --cap 0.1 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
  --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed $ss --gpu $g --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 \
  > logs/pfast_${tire}_${n}_s${s}_ss$ss.log 2>&1
