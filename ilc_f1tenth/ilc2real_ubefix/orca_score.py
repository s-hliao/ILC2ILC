"""orca_score.py GLOB [GLOB ...] (2026-11-05): the ORCA study's table (runs/screen/ORCA_RULES.md) from orca_eval.json files. Per
network and plan: % change of the score (stage it 3 vs the run's own zero-shot it 0), bootstrap 95 % CI over RUNS (each run's
cars averaged), runs better; and the parts, zero-shot -> stage: rms e_y, pace, time outside the 0.5 m tube, failures.
    python orca_score.py 'runs/screen/PLSL*'            (held-out ORCA: practised square2fast)
    python orca_score.py 'runs/screen/ORCA_*_ss*'       (practised ORCA)"""
import glob
import json
import sys

import numpy as np

rng = np.random.default_rng(0)
files = sorted(f for g in sys.argv[1:] for f in glob.glob(g + '/orca_eval.json'))
E = {f.split('/')[-2]: json.load(open(f)) for f in files}
print(f'{len(E)} runs')
for net, lab in (('freePPON', 'PPO+DR + stage'), ('rmaN', 'RMA + stage')):
    runs = {k: v for k, v in E.items() if f'_{net}_' in k}
    if not runs:
        continue
    for pn in sorted({p for e in runs.values() for p in e}):
        R, parts = [], {q: ([], []) for q in ('rms_ey', 'pace', 'dev_time', 'fail')}
        for e in runs.values():
            if pn not in e or '3' not in e[pn] or '0' not in e[pn]:
                continue
            cars = [c for c in e[pn]['0'] if c in e[pn]['3']]
            R.append(np.mean([100 * (e[pn]['3'][c]['score'] / e[pn]['0'][c]['score'] - 1) for c in cars]))
            for q in parts:
                parts[q][0].append(np.nanmean([e[pn]['0'][c][q] for c in cars]))
                parts[q][1].append(np.nanmean([e[pn]['3'][c][q] for c in cars]))
        if not R:
            continue
        R = np.array(R)
        bs = R[rng.integers(0, len(R), (4000, len(R)))].mean(1)
        print(f'{lab:15s} {pn.replace("_b0_free", ""):20s} score {R.mean():+6.1f} % [{np.percentile(bs, 2.5):+.1f}, '
              f'{np.percentile(bs, 97.5):+.1f}]  runs better {int((R < 0).sum())}/{len(R)}')
        z, s = parts['rms_ey']
        print(f'{"":36s} rms e_y {100 * np.mean(z):.1f} -> {100 * np.mean(s):.1f} cm, pace {np.mean(parts["pace"][0]):.3f} -> '
              f'{np.mean(parts["pace"][1]):.3f}, outside tube {np.mean(parts["dev_time"][0]):.2f} -> {np.mean(parts["dev_time"][1]):.2f} s, '
              f'failures {np.mean(parts["fail"][0]):.2f} -> {np.mean(parts["fail"][1]):.2f}')
