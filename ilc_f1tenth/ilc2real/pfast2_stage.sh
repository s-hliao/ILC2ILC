#!/bin/bash
# pfast2_stage.sh TIRE TAG PLAN NET SEED STAGESEED GPU (2026-11-06, runs/screen/PFAST_RULES.md): practise a car hardware track at its rule
# rung (plastic square2fast_a45_x120, peanut a33_x120; rubber fig8R100_x140, peanut x150) with the existing stage settings (2-lap trials).
tire=$1 tag=$2 plan=$3 n=$4 s=$5 ss=$6 g=$7; cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 \
       F1T_PROCS=${F1T_PROCS:-4}
if [ "$tire" = plastic ]; then
  CARS="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
  ANC="mocap_square2fast_b0_free_a45 mocap_figfast_b0_free_a45"
else
  CARS="ax:mu=1.667 ax:mu=1.833,mass=1.1"; ANC="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120"
fi
D=runs/screen/PFAST2_${tire}_${tag}_${n}_s${s}_ss$ss
[ -f $D/summary.json ] && exit 0
PYTHONFAULTHANDLER=1 ~/miniconda3/envs/f1t/bin/python -u hw_stage.py --policy runs/${n}_s${s}/policy_final.npz --out $D --cars $CARS \
  --goals $plan --anchor-plans $ANC \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
  --iters 3 --beta 0.5 --delta 0.05 --cap 0.1 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
  --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed $ss --gpu $g --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 \
  > logs/pfast2_${tire}_${tag}_${n}_s${s}_ss$ss.log 2>&1
