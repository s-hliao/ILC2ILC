#!/bin/bash
# fig2_stage.sh NET PANEL SEED STAGESEED (2026-11-06, paper/FIGURE_PLAN_1106.md figure 2): our stage on the figure's panels, the fixed
# simulator (this copy, F1T_SPIN_RHO=3), 4 iterations x 2-lap trials (0-8 real laps; user: few-shot, at most 8), --save-its 1.
#   NET   freePPON (runs/freePPON_s<seed>) | rmaN (runs/rmaN_s<seed>, --z-path 1) | ours / oursdr (runs/fig1/<net>_s<seed>, figure 1's)
#   PANEL tier    the 36 pre-registered coverage-tier cars (mb_car.py A / B / C, nominal tires): practise square2fast at the plan
#                 speed (mocap_square2fast_b0_free; anchors square2fast + figfast free), the plastic protocol's flags
#         plastic the main plastic protocol (car_long.sh plastic: 12 cars, square2fast_a45)
#         rubber  the main rubber protocol (car_long.sh rubber: 2 cars, fig8R100_x120)
n=$1 panel=$2 s=$3 ss=$4
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real_fix; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=4 JAX_PLATFORMS=cpu
case $n in freePPON|rmaN) POL=runs/${n}_s$s/policy_final.npz;; *) POL=runs/fig1/${n}_s$s/policy_final.npz;; esac
X=""; [ $n = rmaN ] && X="--z-path 1"
case $panel in
  tier)    CARS=$($HOME/miniconda3/envs/f1t/bin/python -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
           G=mocap_square2fast_b0_free; ANC="mocap_square2fast_b0_free mocap_figfast_b0_free";;
  plastic) CARS="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
           G=mocap_square2fast_b0_free_a45; ANC="mocap_square2fast_b0_free_a45 mocap_figfast_b0_free_a45";;
  rubber)  CARS="ax:mu=1.667 ax:mu=1.833,mass=1.1"; G=mocap_fig8R100_b0_free_x120; ANC="mocap_square2fast_b0_free_x120 mocap_fig8R100_b0_free_x120";;
esac
D=runs/fig2/${panel}_${n}_s${s}_ss$ss; mkdir -p runs/fig2 logs/fig2
[ -f $D/summary.json ] && exit 0
PYTHONFAULTHANDLER=1 taskset -c 0-29 ~/miniconda3/envs/f1t/bin/python -u hw_stage.py --policy $POL --out $D --cars $CARS \
  --goals $G --anchor-plans $ANC \
  --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
  --iters 4 --beta 0.5 --delta 0.05 --cap 0.1 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
  --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
  --eval-seed 701 --seed $ss --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 $X \
  > logs/fig2/$(basename $D).log 2>&1
