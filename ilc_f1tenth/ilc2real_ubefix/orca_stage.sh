#!/bin/bash
# orca_stage.sh NET SEED STAGESEED GPU (2026-11-05; user: "run training on harder track like ethzurich orca, also don't stop the trial at
# 0.5 m deviation, just mark it as deviation time"): the plastic stage (runs/screen_final/plastic/PLSL_*/args.json: 12 plastic cars,
# 3 iterations x 2-lap trials, trust region, launch) practising the scaled ORCA track orca43 (make_orca_track.py) instead of
# square2fast; F1T_DEV_TIME=1 (leaving the 0.5 m envelope is timed, not stopped; spin-outs / stalls still end a run). Sim only.
n=$1 s=$2 ss=$3 g=$4; cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 \
       F1T_PROCS=${F1T_PROCS:-4}
CARS="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
D=runs/screen/ORCA_${n}_s${s}_ss$ss
[ -f $D/summary.json ] && exit 0
~/miniconda3/envs/f1t/bin/python -u hw_stage.py --policy runs/${n}_s${s}/policy_final.npz --out $D --cars $CARS \
  --goals orca43_b0_free_a33 --anchor-plans mocap_square2fast_b0_free_a45 mocap_figfast_b0_free_a45 \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
  --iters 3 --beta 0.5 --delta 0.05 --cap 0.1 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
  --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed $ss --gpu $g --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 \
  > logs/orca_${n}_s${s}_ss$ss.log 2>&1
