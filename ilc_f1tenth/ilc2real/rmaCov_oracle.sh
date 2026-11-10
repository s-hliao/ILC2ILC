#!/bin/bash
# rmaCov_oracle.sh (2026-11-07 ~02:35, user: "train a separate rma that directly covers deployment dynamics instead of the original dr
# set"): figure 1/2's RMA oracle-latent reference, v2. The RMA teacher = ppo_dr.py --mode rma with rmaN_s<s>'s exact arguments except
# the DR family: F1T_DR_BOX = the union box of all 50 deployment cars (tiers A-C, plastic, rubber) + ~5 % margin (checked: every car's z
# inside the sampled box), phase 2 skipped (--adapt-iters 0: the oracle reads the true z). Then rma_oracle_car.py (true z, same ZNOM /
# ZSC) and figure 2's zero-shot scoring (ladder_eval, held-out peanut, 5 laps, seed 701) -> runs/fig2base/rmaCovOracle_<panel>_s<s>.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 XLA_PYTHON_CLIENT_PREALLOCATE=false
export F1T_DR_BOX='{"m": [0.6, 1.4], "mu": [0.38, 1.93], "ks": [0.68, 1.32], "kt": [0.68, 1.32], "tau_s": [0.015, 0.155], "d_off": [-0.105, 0.105]}'
PY=$HOME/miniconda3/envs/f1t/bin/python
mkdir -p logs/rmaCov
for s in 0 1 2; do
  [ -f runs/rmaCov_s$s/policy_final.npz ] && continue
  CUDA_VISIBLE_DEVICES=${GPU:-1} taskset -c $((14 + 2 * s))-$((15 + 2 * s)) $PY -u ppo_dr.py --mode rma --dr 1.0 --init runs/freeBCN_s$s/policy_final.npz \
    --plans mocap_square2fast_b0_free mocap_square2fast_b0_free_v85 mocap_square2fast_b0_free_v70 mocap_figfast_b0_free mocap_figfast_b0_free_v85 mocap_figfast_b0_free_v70 \
    --iters 300 --lanes 1024 --T 160 --epochs 4 --mb 16384 --lr 1e-4 --log-std -2.5 --gamma 0.98 --lam 0.95 --clip 0.2 --hist 40 \
    --adapt-iters 0 --reward task --seed $s --out runs/rmaCov_s$s > logs/rmaCov/train_s$s.log 2>&1 &
done
wait
unset F1T_DR_BOX
export F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu F1T_PROCS=8 LADDER_ITS=0
TIER=$($PY -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
PC="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
RC="ax:mu=1.667 ax:mu=1.833,mass=1.1"
for s in 0 1 2; do
  $PY rma_oracle_car.py runs/fig2base/rmaCovOracle_rubber_s$s runs/rmaCov_s$s $RC
  $PY rma_oracle_car.py runs/fig2base/rmaCovOracle_plastic_s$s runs/rmaCov_s$s $PC
  $PY rma_oracle_car.py runs/fig2base/rmaCovOracle_tier_s$s runs/rmaCov_s$s $TIER
done >> logs/rmaCov/build.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2base/rmaCovOracle_rubber_s? > logs/rmaCov/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$(echo $PC | tr ' ' '|')" $PY -u ladder_eval.py runs/fig2base/rmaCovOracle_plastic_s? >> logs/rmaCov/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free LADDER_CARS="$(echo $TIER | tr ' ' '|')" $PY -u ladder_eval.py runs/fig2base/rmaCovOracle_tier_s? >> logs/rmaCov/eval.log 2>&1
touch logs/rmaCov/DONE
