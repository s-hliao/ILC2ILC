#!/bin/bash
# rma_oracle_fig2.sh (2026-11-07, paper/FIGURE_PLAN_1106.md): figure 2's RMA-with-oracle-latent reference: rmaN_s<s>'s actor + each car's true
# latent (rma_oracle_car.py), scored exactly as figure 2's zero-shot points (ladder_eval, the panel's held-out plan, seed 701, 5 laps).
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu F1T_PROCS=8 LADDER_ITS=0
PY="taskset -c 26-35 $HOME/miniconda3/envs/f1t/bin/python -u"
TIER=$($HOME/miniconda3/envs/f1t/bin/python -c "import mb_car as m; print(' '.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
PC="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
RC="ax:mu=1.667 ax:mu=1.833,mass=1.1"
mkdir -p logs/rma_oracle
for s in 0 1 2; do
  $PY rma_oracle_car.py runs/fig2base/rmaOracle_rubber_s$s runs/rmaN_s$s $RC
  $PY rma_oracle_car.py runs/fig2base/rmaOracle_plastic_s$s runs/rmaN_s$s $PC
  $PY rma_oracle_car.py runs/fig2base/rmaOracle_tier_s$s runs/rmaN_s$s $TIER
done >> logs/rma_oracle/build.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY ladder_eval.py runs/fig2base/rmaOracle_rubber_s? > logs/rma_oracle/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$(echo $PC | tr ' ' '|')" $PY ladder_eval.py runs/fig2base/rmaOracle_plastic_s? >> logs/rma_oracle/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free LADDER_CARS="$(echo $TIER | tr ' ' '|')" $PY ladder_eval.py runs/fig2base/rmaOracle_tier_s? >> logs/rma_oracle/eval.log 2>&1
touch logs/rma_oracle/DONE
