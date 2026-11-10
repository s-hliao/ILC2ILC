#!/bin/bash
# stp_run.sh ARM CELL SEED STAGESEED DEV (2026-11-06, runs/screen/STEP_SCREEN_1106.md on matcha): one step-size screen run on the
# FIXED simulator (this copy). ARM 0 | A | B; CELL ethz (rubber, GPU DEV) | plastic (CPU; DEV ignored).
arm=$1 cell=$2 s=$3 ss=$4 g=$5
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 XLA_PYTHON_CLIENT_PREALLOCATE=false NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1
case $arm in 0) X="--cap 0.1";; A) X="--cap 0.4";; B) X="--cap 0.4 --fit-stall 0";; esac
n=freePPON
if [ $cell = ethz ]; then
  export CUDA_VISIBLE_DEVICES=$g F1T_PROCS=4
  CARS="ax:mu=1.667 ax:mu=1.833,mass=1.1"; G=ethzmobil43_b0_free; ANC="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120"; TL=1
  D=runs/screen/STP${arm}_ethz_rubber_${n}_s${s}_ss$ss
else
  export JAX_PLATFORMS=cpu F1T_PROCS=4
  CARS="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
  G=mocap_square2fast_b0_free_a45; ANC="mocap_square2fast_b0_free_a45 mocap_figfast_b0_free_a45"; TL=2
  D=runs/screen/STP${arm}_plastic_${n}_s${s}_ss$ss
fi
[ -f $D/summary.json ] && exit 0
PYTHONFAULTHANDLER=1 taskset -c 0-29 ~/miniconda3/envs/f1t/bin/python -u hw_stage.py --policy runs/${n}_s${s}/policy_final.npz --out $D --cars $CARS \
  --goals $G --anchor-plans $ANC \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
  --iters 3 --beta 0.5 --delta 0.05 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
  --goal-cycle 1 --trial-laps $TL --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed $ss --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 $X \
  > logs/$(basename $D).log 2>&1
