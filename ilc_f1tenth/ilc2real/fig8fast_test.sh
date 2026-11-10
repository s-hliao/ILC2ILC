#!/bin/bash
# fig8fast_test.sh (2026-11-08 ~04:15, from matcha; user: "might it be counterintuitively easier to run the plastic at higher speeds because it begins drifting earlier, which if controlled, allows for sustained patterns"): plastic fig8R115 at NOMINAL (2.62 m/s) and x110 vs the running x090 cell. PASS (before results): it3 change < 0 for both networks, no added crashes, rear-out share at it3 >= it0. Was fig8x_test.sh (user: plastic fig8 must work "no matter what"): SB stage tests of plastic fig8 at 0.85 / 0.90
# x nominal speed (scale_plan.py; zero-shot sweep: the slide grows with speed, x085-x090 = rear out 23-40 % of corner time at e_y 0.12-0.17
# m, no crashes; nominal diverges, a45 grips). prac_queue3.sh flags exactly (--skip-below 0.5, no yaw term), anchors = the practised plan +
# square2fast nominal, freePPON_s0 / rmaN_s0, stage seed 9001, 4 workers each pinned 20-35 (matcha load). Scored ladder_eval_launch.py +
# ladder_eval.py, its 0-4. PASS (written before results): mean per-car change it3 < 0 for both networks with no added crashes, and the
# rear-out share at it3 >= its it0.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu F1T_PROCS=4
PY=~/miniconda3/envs/f1t/bin/python; mkdir -p logs/fig8fast
PC='n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000'
one() {  # k net
  local G=mocap_fig8R115_b0_free$1 D=runs/screen/PRACB_plastic_fig8fast$1_$2_s0_ss9001 X=''; [ $2 = rmaN ] && X='--z-path 1'
  taskset -c 20-35 nice -n 5 $PY -u hw_stage.py --policy runs/$2_s0/policy_final.npz --out $D --cars $PC --goals $G --anchor-plans $(printf "%s\n" $G mocap_fig8R115_b0_free | sort -u | tr "\n" " ") \
    --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
    --iters 4 --beta 0.5 --delta 0.05 --cap 0.4 --fit-stall 0 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
    --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
    --eval-seed 701 --seed 9001 --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-below 0.5 --save-its 1 $X > logs/fig8fast/$(basename $D).log 2>&1
  [ -f $D/summary.json ] || return
  local CB="$(echo $PC | tr ' ' '|')"
  LADDER_ITS='0 1 2 3 4' LADDER_OUT=ladder_launch.json LADDER_LAPS=5 LADDER_PLANS=$G LADDER_CARS="$CB" taskset -c 20-35 nice -n 5 $PY -u ladder_eval_launch.py $D >> logs/fig8fast/$(basename $D).log 2>&1
  LADDER_ITS='0 1 2 3 4' LADDER_OUT=sb_practised.json LADDER_LAPS=5 LADDER_PLANS=$G LADDER_CARS="$CB" taskset -c 20-35 nice -n 5 $PY -u ladder_eval.py $D >> logs/fig8fast/$(basename $D).log 2>&1
}
for k in "" _x110; do for n in freePPON rmaN; do one "$k" $n & done; done; wait
touch logs/fig8fast/DONE
