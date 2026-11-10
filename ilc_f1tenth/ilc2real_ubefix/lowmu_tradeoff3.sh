#!/bin/bash
# low-friction round 3 (2026-10-11), on the STOPPING-RULE network (runs/v3drBt_stop, stop_pipeline.sh) at the 10-lap budget.
# Round 2 (on the fixed-30 network): bigger GN steps help real_mu (cap 0.15 beta 0.8: 29.2 % strict / 95.8 % complete vs 12.5 / 97.9).
# (a) real_mu only: cap 0.15 / 0.20 / 0.25 at beta 0.8 (path-weighted objective, pace target 0.90);
# (b) the best of (a) on ALL five cars (the setting must not cost the other cars), scored like the headline (reeval, 12 runs).
cd "$(dirname "$0")"; source env_tight.sh
PY=~/miniconda3/envs/f1t/bin/python
until grep -q STOP_ARMS_DONE logs/stop_pipeline.out 2>/dev/null; do sleep 60; done
P=runs/v3drBt_stop/policy_final.npz
G2="--goals mocap_square2fast_b25 mocap_figfast_b25"
OBJ="--err-scale 0.025 1e9 1e9 1e9 1e9 0.60 --pace-target 0.90 --beta 0.8"
run() { [ -f runs/hw/$1/summary.json ] || F1T_PROCS=4 $PY -u hw_stage.py --policy $P --out runs/hw/$1 --iters 5 $G2 --goal-cycle 1 $2 --gpu $3 > logs/hw_$1.log 2>&1; }
run lmS_c15_mu "$OBJ --cap 0.15 --cars real_mu" 0 & run lmS_c20_mu "$OBJ --cap 0.20 --cars real_mu" 2 & wait
run lmS_c25_mu "$OBJ --cap 0.25 --cars real_mu" 0 & wait
F1T_PROCS=8 $PY -u reeval.py lmS_c15_mu lmS_c20_mu lmS_c25_mu --runs 6 > logs/reeval_lmS.log 2>&1
BEST=$($PY - <<'PYEOF'
import json, numpy as np
s = {}
for c in (15, 20, 25):
    v = list(json.load(open(f'runs/hw/lmS_c{c}_mu/eval_big.json'))['real_mu'].values())
    s[c] = np.mean([x['success'] for x in v]) + 1e-3 * np.mean([1 - x['fail'] for x in v])
print(max(s, key=s.get))
PYEOF
)
echo "$(date +%T) best cap 0.$BEST on real_mu"
F1T_PROCS=8 $PY -u hw_stage.py --policy $P --out runs/hw/lmS_c${BEST}_all_lap10 --iters 5 $G2 --goal-cycle 1 $OBJ --cap 0.$BEST --gpu 0 \
  > logs/hw_lmS_c${BEST}_all_lap10.log 2>&1
F1T_PROCS=8 $PY -u reeval.py lmS_c${BEST}_all_lap10 > logs/reeval_lmS_all.log 2>&1
echo "$(date +%T) LMS_DONE"
