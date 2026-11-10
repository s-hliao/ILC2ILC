#!/bin/bash
# gps2_fig2.sh PANEL SEED [GPU] (2026-11-07, runs/screen/GPS_1106.md "FAITHFUL RERUN"): figure 2's GPS (MDGPS) arm, the released-code-faithful
# version: hw_stage_gps4.py (gps_core4: local TVLG sampling, PolicyPriorGMM policy linearization, code eta/del rules, per-condition fixed
# starts, policy variance re-estimate, code S-step) with the dynamics prior at the code's strength (--gps-n0 1). Otherwise gps_fig2.sh exactly.
# GPU given: JAX on that GPU (matcha); none: CPU JAX (Lambda).
p=$1 s=$2 g=$3
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=${F1T_PROCS:-2} XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -n "$g" ]; then export CUDA_VISIBLE_DEVICES=$g; else export JAX_PLATFORMS=cpu; fi
PC="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
RC="ax:mu=1.667 ax:mu=1.833,mass=1.1"
case $p in
  plastic) CARS=$PC; G=mocap_square2fast_b0_free_a45;;
  rubber)  CARS=$RC; G=mocap_fig8R100_b0_free_x120;;
  tier)    CARS=$(~/miniconda3/envs/f1t/bin/python -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))"); G=mocap_square2fast_b0_free;;
esac
D=runs/screen/GPS2_${p}_freePPON_s${s}; mkdir -p logs/gps2
[ -f $D/summary.json ] && exit 0
~/miniconda3/envs/f1t/bin/python -u hw_stage_gps4.py --gps 1 --policy runs/freePPON_s$s/policy_final.npz --out $D --cars $CARS \
  --goals $G --anchor-plans $G --iters 24 --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 --goal-cycle 1 --trial-laps 2 \
  --eval-runs 0 --eval-seed 701 --seed $((9001 + s)) --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 --gps-n0 1 \
  > logs/gps2/$(basename $D).log 2>&1
echo "exit $?" >> logs/gps2/$(basename $D).log
