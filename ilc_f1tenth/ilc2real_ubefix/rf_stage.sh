#!/bin/bash
# rf_stage.sh ARM SEED STAGESEED (2026-11-06, runs/screen/REFIT_SCREEN_1106.md): sb_stage.sh plastic freePPON with ONE refit knob changed
# the protocol's original flags otherwise, the fixed simulator (this copy), 4 iterations, CPU (JAX_PLATFORMS=cpu), cores 0-29.
arm=$1 p=plastic n=freePPON s=$2 ss=$3
case $arm in AW0) XR="--anchor-w 0";; AW5) XR="--anchor-w 5";; AW80) XR="--anchor-w 80";; AD0) XR="--age-decay 0";; AD25) XR="--age-decay 0.25";; AD1) XR="--age-decay 1";; FIT) XR="--steps 3000 --lr 3e-4";; *) echo "unknown arm $arm"; exit 1;; esac
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4 JAX_PLATFORMS=cpu
PC="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
RC="ax:mu=1.667 ax:mu=1.833,mass=1.1"
PA="mocap_square2fast_b0_free_a45 mocap_figfast_b0_free_a45"; RA="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120"
TL=2
case $p in
  plastic)  CARS=$PC; G=mocap_square2fast_b0_free_a45; ANC=$PA;;
  rubber)   CARS=$RC; G=mocap_fig8R100_b0_free_x120; ANC=$RA;;
  ethzp)    CARS=$PC; G=ethzmobil43_b0_free_a33; ANC=$PA; TL=1;;
  ethzr)    CARS=$RC; G=ethzmobil43_b0_free; ANC=$RA; TL=1;;
  ethzr110) CARS=$RC; G=ethzmobil43_b0_free_x110; ANC=$RA; TL=1;;
  peax150)  CARS=$RC; G=mocap_peanutM_b0_free_x150; ANC=$RA;;
  tier)     CARS=$(~/miniconda3/envs/f1t/bin/python -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
            G=mocap_square2fast_b0_free; ANC="mocap_square2fast_b0_free mocap_figfast_b0_free";;
  *) echo "unknown protocol $p"; exit 1;;
esac
case $n in freePPON|rmaN) POL=runs/${n}_s$s/policy_final.npz;; *) POL=runs/fig1/${n}_s$s/policy_final.npz;; esac
X=""; [ $n = rmaN ] && X="--z-path 1"
D=runs/screen/RF_${arm}_${p}_${n}_s${s}_ss$ss; mkdir -p logs/rf
[ -f $D/summary.json ] && exit 0
PYTHONFAULTHANDLER=1 taskset -c 0-29 ~/miniconda3/envs/f1t/bin/python -u hw_stage.py --policy $POL --out $D --cars $CARS \
  --goals $G --anchor-plans $ANC \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
  --iters 4 --beta 0.5 --delta 0.05 --cap 0.4 --fit-stall 0 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
  --goal-cycle 1 --trial-laps $TL --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed $ss --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 $X $XR \
  > logs/rf/$(basename $D).log 2>&1
