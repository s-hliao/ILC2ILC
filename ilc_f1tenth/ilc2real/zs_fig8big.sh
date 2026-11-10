#!/bin/bash
# zs_fig8big.sh (2026-11-08 ~17:30; user: "fix fig8 ... maybe take longer loops since it's only in sim now"): zero-shot sweep on
# mocap_fig8R165 (lobes R 1.65 = the oval's curvature 0.61; fig8R115's crossing angle; 3.3 x 6.7 m, sim only), both tires, both s0
# networks, launched scorer, 5 laps. Rule (OVAL_RULES_1108.md rule 1): plastic = the fastest plan with no crash and e_y <= 0.17 for both;
# rubber = the fastest plan where neither network crashes.
cd ~/ilc_ws/src/ilc_f1tenth/ilc2real; PY=~/miniconda3/envs/f1t/bin/python; mkdir -p logs/zs_fig8big
PC='n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000'
RC='ax:mu=1.667|ax:mu=1.833,mass=1.1'
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu LADDER_ITS=0 LADDER_LAPS=5 F1T_PROCS=6
LADDER_CARS="$PC" LADDER_OUT=zs_fig8big.json LADDER_PLANS='mocap_fig8R165_b0_free_a30 mocap_fig8R165_b0_free_a34 mocap_fig8R165_b0_free' \
  taskset -c 12-35 nice -n 5 $PY -u ladder_eval_launch.py runs/beh2/ZS_plastic_freePPON_s0 runs/beh2/ZS_plastic_rmaN_s0 > logs/zs_fig8big/plastic.log 2>&1
for n in freePPON rmaN; do D=runs/beh2/ZS_rubber_${n}_s0; mkdir -p $D/its; for c in ax:mu=1.667 ax:mu=1.833,mass=1.1; do ln -sf $PWD/runs/${n}_s0/policy_final.npz "$D/its/it0_${c}_policy.npz"; done
  [ -f runs/${n}_s0/rma_adapt.npz ] && ln -sf $PWD/runs/${n}_s0/rma_adapt.npz $D/its/rma_adapt.npz; done
LADDER_CARS="$RC" LADDER_OUT=zs_fig8big.json LADDER_PLANS='mocap_fig8R165_b0_free_a30 mocap_fig8R165_b0_free_a34 mocap_fig8R165_b0_free' \
  taskset -c 12-35 nice -n 5 $PY -u ladder_eval_launch.py runs/beh2/ZS_rubber_freePPON_s0 runs/beh2/ZS_rubber_rmaN_s0 > logs/zs_fig8big/rubber.log 2>&1
touch logs/zs_fig8big/DONE
