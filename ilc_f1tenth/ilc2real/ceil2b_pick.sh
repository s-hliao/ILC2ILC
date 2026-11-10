#!/bin/bash
# ceil2b_pick.sh (2026-11-07 ~12:30): ceiling v2's FINAL iterate degraded on rubber (ax:mu=1.833,mass=1.1: train reward 0.945 at it 40 ->
# 0.43 at it 200). Rule (all 14 cars): the saved iterate (every 25) with the highest training reward (mean over its +-2 iterations) -> scored
# exactly as v2 -> runs/fig2base/ceil2bPPO_<panel>_s0.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=$HOME/miniconda3/envs/f1t/bin/python
$PY - <<'PYEOF' > logs/ceil2/pick.log 2>&1
import os, json, numpy as np, mb_car as mb, rmaD_latent as L
for p, cars in (('plastic', list(mb.PLASTIC_N04)), ('rubber', list(mb.RUBBER_N06))):
    o = f'runs/fig2base/ceil2bPPO_{p}_s0/its'; os.makedirs(o, exist_ok=True)
    for c in cars:
        t = c.replace(':', '_').replace('=', '_').replace(',', '_')
        rec = [json.loads(l) for l in open(f'logs/ceil2/{t}.log') if l.startswith('{"it"')]
        rew = {r['it']: r['reward'] for r in rec}
        cand = [k for k in range(0, 201, 25) if os.path.exists(f'runs/ceil2/{t}/policy_it{k}.npz')]
        sc = {k: np.mean([rew[j] for j in range(k - 2, k + 3) if j in rew]) for k in cand}
        k = max(sc, key=sc.get)
        d = np.load(f'runs/ceil2/{t}/policy_it{k}.npz'); W = {q: d[q] for q in d.files if q[0] in 'Wb' and q[1:].isdigit()}
        np.savez(os.path.join(o, f'it0_{c}_policy.npz'), **W, mode='rma_oracle', z=L.z_of(c), z_raw=L.raw(c))
        print(p, c, 'picked it', k, 'train reward %.3f' % sc[k], {kk: round(v, 3) for kk, v in sc.items()}, flush=True)
PYEOF
PC="n04_real_nom|n04_real_mass|n04_real_mu|n04_real_act|n04_real_lag|n04_harsh_combo|n04_harsh_steer|n04_harsh_sense|n04_harsh_drift|ax:mu=0.6000|ax:mu=0.4667|ax:mu=0.4000"
export F1T_PROCS=12 LADDER_ITS=0
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2base/ceil2bPPO_rubber_s0 > logs/ceil2/eval_b.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PC" $PY -u ladder_eval.py runs/fig2base/ceil2bPPO_plastic_s0 >> logs/ceil2/eval_b.log 2>&1
touch logs/ceil2/DONE_b
