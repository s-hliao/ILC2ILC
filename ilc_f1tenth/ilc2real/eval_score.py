"""eval_score.py FILE 'GLOB' ['GLOB' ...] (2026-11-05): one table for any per-run evaluation file of ladder_eval.py / orca_eval.py
(runs/screen/SPEED_LADDER.md, ETHZ_RULES.md). Per network and plan: score change of the stage (it 3) vs the run's own zero-shot (it 0),
mean of each run's cars, bootstrap 95 % CI over RUNS, runs better; and the parts zero-shot -> stage: rms e_y, pace, time outside the
0.5 m tube, spin-outs. A failed run scores 5."""
import glob
import json
import sys

import numpy as np

rng = np.random.default_rng(0)
fname = sys.argv[1]
E = {f.split('/')[-2]: json.load(open(f)) for g in sys.argv[2:] for f in sorted(glob.glob(f'{g}/{fname}'))}
print(f'{len(E)} runs, {fname}')
for net, lab in (('freePPON', 'PPO+DR + stage'), ('rmaN', 'RMA + stage')):
    runs = [e for k, e in E.items() if f'_{net}_' in k]
    if not runs:
        continue
    for pn in sorted({p for e in runs for p in e}, key=lambda p: (len(p), p)):
        R, Q = [], {q: ([], []) for q in ('score', 'rms_ey', 'pace', 'dev_time', 'fail')}
        for e in runs:
            if pn not in e or '0' not in e[pn] or '3' not in e[pn]:
                continue
            cars = [c for c in e[pn]['0'] if c in e[pn]['3']]
            R.append(np.mean([100 * (e[pn]['3'][c]['score'] / e[pn]['0'][c]['score'] - 1) for c in cars]))
            for q in Q:
                Q[q][0].append(np.nanmean([e[pn]['0'][c][q] for c in cars]))
                Q[q][1].append(np.nanmean([e[pn]['3'][c][q] for c in cars]))
        if not R:
            continue
        R = np.array(R)
        bs = R[rng.integers(0, len(R), (4000, len(R)))].mean(1)
        m = lambda q, i: np.nanmean(Q[q][i])
        print(f'{lab:15s} {pn.replace("_b0_free", ""):26s} score {m("score", 0):.2f} -> {m("score", 1):.2f}  {R.mean():+6.1f} % '
              f'[{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]  runs better {int((R < 0).sum())}/{len(R)} | '
              f'e_y {100 * m("rms_ey", 0):.1f} -> {100 * m("rms_ey", 1):.1f} cm, pace {m("pace", 0):.3f} -> {m("pace", 1):.3f}, '
              f'outside tube {m("dev_time", 0):.2f} -> {m("dev_time", 1):.2f} s, spin-outs {100 * m("fail", 0):.0f} -> {100 * m("fail", 1):.0f} %')
