#!/bin/bash
# sb_big2.sh (2026-11-07 ~23:05, matcha; user: "does training on both help?"): sb_big.sh (ube) with the stage practising BOTH room-sized
# tracks -- the oval CCW mocap_oval165_b0_free_a26 and the figure-eight mocap_fig8R115_b0_free_a24, alternated (--goal-cycle 1) -- same
# anchors, SB flags, 4 iterations, 12 plastic sim cars, freePPON_s0 / rmaN_s0 x stage seed 9001 (8 workers each). Simulator = matcha's
# ilc2real (differs from ube's ilc2real_fix only by opt-in additions unused here: checked 23:05). Then the same held-out scoring.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 F1T_PROCS=8 JAX_PLATFORMS=cpu
PY=~/miniconda3/envs/f1t/bin/python
PC="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
G="mocap_oval165_b0_free_a26 mocap_fig8R115_b0_free_a24"; ANC="mocap_oval165_b0_free_a26 mocap_square2fast_b0_free_a45"; mkdir -p logs/sbbig2
for n in freePPON rmaN; do
  D=runs/screen/SBBIG2_plastic_${n}_s0_ss9001; [ -f $D/summary.json ] && continue
  X=""; [ $n = rmaN ] && X="--z-path 1"
  taskset -c 0-15 $PY -u hw_stage.py --policy runs/${n}_s0/policy_final.npz --out $D --cars $PC --goals $G --anchor-plans $ANC \
    --rollback 0.7 --tr 1 --tr-lo 0.25 --tr-hi 0.75 --tr-max 0.4 --fit-realize 0.9 --fit-max-steps 8000 --crash-rollback 1 --slope-w 1.0 \
    --iters 4 --beta 0.5 --delta 0.05 --cap 0.4 --fit-stall 0 --jac closed --objective task --err-scale 0.1 1e9 1e9 1e9 1e9 0.24 \
    --goal-cycle 1 --trial-laps 2 --anchor-w 20 --anchor-n0 6 --age-decay 0.5 --steps 800 --lr 1e-4 --eval-runs 0 \
    --eval-seed 701 --seed 9001 --gpu 0 --backend sim --launch 1.0 8.0 --sim-launch 1 --skip-launch 0.5 --save-its 1 $X \
    > logs/sbbig2/$(basename $D).log 2>&1 &
done; wait
touch logs/sbbig2/STAGE_DONE
export LADDER_ITS="0 1 2 3 4" F1T_PROCS=16 LADDER_CARS="$(echo $PC | tr ' ' '|')"
for pl in mocap_bean165CW_b0_free_a26 mocap_oval165_b0_free_a26 mocap_fig8R115_b0_free_a24 mocap_bean165_b0_free_a26 mocap_oval165CW_b0_free_a26 mocap_peanutM_b0_free_a33; do
  LADDER_OUT=sb_$pl.json LADDER_LAPS=5 LADDER_PLANS=$pl taskset -c 0-15 nice -n 5 $PY -u ladder_eval.py runs/screen/SBBIG2_plastic_* >> logs/sbbig2/eval.log 2>&1
done
touch logs/sbbig2/DONE
