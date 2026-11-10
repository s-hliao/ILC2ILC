#!/bin/bash
# rmaD_oracle.sh (2026-11-07 ~02:50, Lambda; paper/FIGURE_PLAN_1106.md "RMA oracle-latent v3"): the deployment-covering RMA teacher
# rmaD_s<s> (rmaD_train.py: the multi-body deployment simulator, every axis the deployment cars vary, 14-d latent; init the planar covering
# teacher rmaCov_s<s>, pushed here from matcha), then the oracle = rmaD_s<s> + each car's TRUE 14-d z (mode rma_oracle), scored as figure 2's
# zero-shot points (ladder_eval, the panel's held-out peanut, 5 laps, seed 701) -> runs/fig2base/rmaDOracle_<panel>_s<s>.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/miniconda3/envs/f1t/bin/python
mkdir -p logs/rmaD
train() { s=$1
  until [ -f runs/rmaCov_s$s/policy_final.npz ] && [ -f runs/rmaCov_s$s/agent_final.npz ]; do sleep 60; done
  [ -f runs/rmaD_s$s/policy_final.npz ] || F1T_PROCS=${PROCS:-8} $PY -u rmaD_train.py --init runs/rmaCov_s$s/policy_final.npz --out runs/rmaD_s$s \
      --seed $s > logs/rmaD/train_s$s.log 2>&1; }
for s in 0 1 2; do train $s & done; wait
$PY - <<'PYEOF' >> logs/rmaD/build.log 2>&1
import os, numpy as np, mb_car as mb, rmaD_latent as L
tier = sorted(k for k in mb.REAL_CARS if k[:5] in ('tiera', 'tierb', 'tierc'))
panels = dict(tier=tier, plastic=list(mb.PLASTIC_N04), rubber=list(mb.RUBBER_N06))
for s in range(3):
    d = np.load(f'runs/rmaD_s{s}/policy_final.npz'); W = {k: d[k] for k in d.files if k[0] in 'Wb' and k[1:].isdigit()}
    for p, cars in panels.items():
        o = f'runs/fig2base/rmaDOracle_{p}_s{s}/its'; os.makedirs(o, exist_ok=True)
        for c in cars:
            np.savez(os.path.join(o, f'it0_{c}_policy.npz'), **W, mode='rma_oracle', z=L.z_of(c), z_raw=L.raw(c))
    print('built', s)
PYEOF
TIER=$($PY -c "import mb_car as m; print('|'.join(sorted(k for k in m.REAL_CARS if k[:5] in ('tiera','tierb','tierc'))))")
PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_PROCS=12 LADDER_ITS=0
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2base/rmaDOracle_rubber_s? > logs/rmaD/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/fig2base/rmaDOracle_plastic_s? >> logs/rmaD/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free LADDER_CARS="$TIER" $PY -u ladder_eval.py runs/fig2base/rmaDOracle_tier_s? >> logs/rmaD/eval.log 2>&1
touch logs/rmaD/DONE
