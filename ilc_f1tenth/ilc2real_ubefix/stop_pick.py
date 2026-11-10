#!/usr/bin/env python3
"""stop_pick.py RUN SIMVAL.json OUT_DIR: apply the sim-stage stopping rule (stop_analysis.py: the snapshot with the best mean
held-out success over the 5 non-nominal sim conditions; ties -> the earliest) to RUN's snapshots and copy the picked one to
OUT_DIR/policy_final.npz (+ pick.json). Sim-only: no real data."""
import json
import os
import shutil
import sys

import numpy as np

run, simval, out = sys.argv[1:4]
COND = ['nominal', 'pstart', 'delay', 'dist', 'dr', 'drwide']
S = json.load(open(simval))[os.path.basename(run.rstrip('/'))]
its = sorted(int(k) for k in S)
score = np.array([100 * np.mean([S[str(i)][c]['heldout']['success'] for c in COND[1:]]) for i in its])
k = int(np.flatnonzero(score >= score.max() - 1e-9)[0])
it = its[k]
src = os.path.join(run, 'policy_bc.npz' if it == 0 else f'policy_it{it:03d}.npz')
os.makedirs(out, exist_ok=True)
shutil.copy(src, os.path.join(out, 'policy_final.npz'))
json.dump(dict(run=run, rule='argmax mean non-nominal/heldout/success', picked_it=it, src=src,
               scores={str(i): float(s) for i, s in zip(its, score)}), open(os.path.join(out, 'pick.json'), 'w'), indent=1)
print(f'{run}: picked iteration {it} (score {score[k]:.1f}); it30 {score[its.index(30)]:.1f}' if 30 in its else f'picked {it}')
