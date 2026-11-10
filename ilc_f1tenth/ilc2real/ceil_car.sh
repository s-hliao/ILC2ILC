#!/bin/bash
# ceil_car.sh (2026-11-07 ~04:40, user: "Per-car PPO ceiling"): figure 2's CEILING reference = PPO on each test car's own dynamics in the
# deployment simulator (rmaD_train.py --car: every episode on that car, its true z), init rmaD_s0 (the deployment-sim teacher), rmaD's PPO
# settings, 100 iterations x 64 episodes of 5.05 laps on rmaN's 6 training plans; seed 0; plastic (12) + rubber (2) cars. Then each car's
# final policy + its true z scored as figure 2's zero-shot points (ladder_eval, held-out peanut, 5 laps, seed 701) -> runs/fig2base/ceilPPO_<panel>_s0.
cd /home/henry/ilc_ws/src/ilc_f1tenth/ilc2real; source env_tight.sh >/dev/null 2>&1
export F1T_MU=0.6 F1T_PLANS=$PWD/plans_nom06 F1T_DEV_TIME=1 F1T_SPIN_RHO=3 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/miniconda3/envs/f1t/bin/python
PC="n04_real_nom n04_real_mass n04_real_mu n04_real_act n04_real_lag n04_harsh_combo n04_harsh_steer n04_harsh_sense n04_harsh_drift ax:mu=0.6000 ax:mu=0.4667 ax:mu=0.4000"
RC="ax:mu=1.667 ax:mu=1.833,mass=1.1"
mkdir -p logs/ceil
for c in $RC $PC; do
  t=$(echo $c | tr ':=,' '___')
  [ -f runs/ceil/$t/policy_final.npz ] || F1T_PROCS=${PROCS:-2} $PY -u rmaD_train.py --init runs/rmaD_s0/policy_final.npz --out runs/ceil/$t \
      --car "$c" --iters 100 --episodes 64 --seed 0 > logs/ceil/$t.log 2>&1 &
done
wait
$PY - <<'PYEOF' >> logs/ceil/build.log 2>&1
import os, numpy as np, mb_car as mb, rmaD_latent as L
for p, cars in (('plastic', list(mb.PLASTIC_N04)), ('rubber', list(mb.RUBBER_N06))):
    o = f'runs/fig2base/ceilPPO_{p}_s0/its'; os.makedirs(o, exist_ok=True)
    for c in cars:
        t = c.replace(':', '_').replace('=', '_').replace(',', '_')
        d = np.load(f'runs/ceil/{t}/policy_final.npz'); W = {k: d[k] for k in d.files if k[0] in 'Wb' and k[1:].isdigit()}
        np.savez(os.path.join(o, f'it0_{c}_policy.npz'), **W, mode='rma_oracle', z=L.z_of(c), z_raw=L.raw(c))
    print('built', p)
PYEOF
PCB="$(echo $PC | tr ' ' '|')"
export F1T_PROCS=16 LADDER_ITS=0
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_x120 $PY -u ladder_eval.py runs/fig2base/ceilPPO_rubber_s0 > logs/ceil/eval.log 2>&1
LADDER_OUT=fig2_eval.json LADDER_LAPS=5 LADDER_PLANS=mocap_peanutM_b0_free_a33 LADDER_CARS="$PCB" $PY -u ladder_eval.py runs/fig2base/ceilPPO_plastic_s0 >> logs/ceil/eval.log 2>&1
touch logs/ceil/DONE
