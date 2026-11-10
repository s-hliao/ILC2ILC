#!/usr/bin/env python3
"""table_redo.py [--tag _val]: the car comparison of car_redo.sh, 3 seeds per arm: b25 success, crash, all-plans success
(mean +- sd over seeds) and per car b25 success, from reeval.py's eval_big<tag>.json (ladder arms: the rung-adaptive
eval_big_adaptive<tag>.json; the baselines on our rungs: eval_big_adaptive<tag>_rungs.json)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else '_val'
CARS = ['real_nom', 'real_mass', 'real_mu', 'real_act', 'real_lag', 'ax:mu=0.9', 'ax:mu=0.7', 'ax:mu=0.6']
B25 = ('mocap_square2fast_b25', 'mocap_figfast_b25')
ARMS = [('ours zero-shot', 'v3drBtL_s{s}_zs8', f'eval_big{TAG}.json'),
        ('ours + 10-lap ladder', 'v3drBtL_s{s}_ladR_lap10', f'eval_big_adaptive{TAG}.json'),
        ('PPO+DR (task) zero-shot', 'ppoL_task_s{s}_zs8', f'eval_big{TAG}.json'),
        ('PPO+DR (task) on our rungs', 'ppoL_task_s{s}_zs8', f'eval_big_adaptive{TAG}_rungs.json'),
        ('PPO+DR (task) + our ladder', 'ppoL_task_s{s}_ladR_lap10', f'eval_big_adaptive{TAG}.json'),
        ('RMA (task) zero-shot', 'rmaL_task_s{s}_zs8', f'eval_big{TAG}.json'),
        ('RMA (task) on our rungs', 'rmaL_task_s{s}_zs8', f'eval_big_adaptive{TAG}_rungs.json'),
        ('RMA (task) + our ladder', 'rmaL_task_s{s}_ladR_lap10', f'eval_big_adaptive{TAG}.json')]


def stats(ev):
    b = [ev[c][p]['success'] for c in CARS if c in ev for p in B25 if p in ev[c]]
    f = [ev[c][p]['fail'] for c in CARS if c in ev for p in B25 if p in ev[c]]
    al = [v['success'] for c in CARS if c in ev for v in ev[c].values()]
    per = {c: np.mean([ev[c][p]['success'] for p in B25 if p in ev[c]]) for c in CARS if c in ev}
    return np.mean(b), np.mean(f), np.mean(al), per


print(f'reeval tag {TAG}; b25 = both 25 deg drift plans; mean +- sd over seeds 0-2')
print(f'{"arm":<28}| n | b25 succ      | crash        | all plans    | ' + ' '.join(f'{c.replace("real_", "")[:6]:>6}' for c in CARS))
for name, pat, f in ARMS:
    rows = []
    for s in (0, 1, 2):
        p = os.path.join(HERE, 'runs/hw', pat.format(s=s), f)
        if os.path.exists(p):
            rows.append(stats(json.load(open(p))))
    if not rows:
        continue
    m = lambda i: (100 * np.mean([r[i] for r in rows]), 100 * np.std([r[i] for r in rows]))
    per = ' '.join(f'{100 * np.mean([r[3].get(c, np.nan) for r in rows]):6.0f}' for c in CARS)
    (b, bs), (cr, cs), (al, as_) = m(0), m(1), m(2)
    print(f'{name:<28}| {len(rows)} | {b:5.1f} +- {bs:4.1f} | {cr:5.1f} +- {cs:4.1f} | {al:5.1f} +- {as_:4.1f} | {per}')

# paired differences (as the pogo's ci.paired_ci): the unit of resampling is the (car, b25 plan) cluster, the 3 seeds
# pooled inside it (12 runs x 3 seeds); both arms resampled on the same clusters; 95 % percentile bootstrap, B 10000
PAIRS = [('ours + 10-lap ladder', 'RMA (task) + our ladder'), ('ours + 10-lap ladder', 'PPO+DR (task) + our ladder'),
         ('PPO+DR (task) + our ladder', 'RMA (task) + our ladder'), ('RMA (task) + our ladder', 'RMA (task) on our rungs'),
         ('RMA (task) + our ladder', 'RMA (task) zero-shot')]


def clusters(name):
    pat, f = next((p, f) for n, p, f in ARMS if n == name)
    evs = [json.load(open(p)) for p in (os.path.join(HERE, 'runs/hw', pat.format(s=s), f) for s in (0, 1, 2)) if os.path.exists(p)]
    if not evs:
        return 0, {}
    return len(evs), {(c, p): np.mean([ev[c][p]['success'] for ev in evs]) for c in CARS for p in B25 if all(c in ev and p in ev[c] for ev in evs)}


print('\npaired b25 differences, percentage points (cluster = car x b25 plan, seeds pooled; 95 % bootstrap CI)')
for x, y in PAIRS:
    (nx, cx), (ny, cy) = clusters(x), clusters(y)
    if nx < 3 or ny < 3:
        continue
    keys = sorted(set(cx) & set(cy))
    d = np.array([cx[k] - cy[k] for k in keys])
    bs = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
    print(f'{x:<28} - {y:<28}: {100 * d.mean():+6.1f} [{100 * np.quantile(bs, 0.025):+6.1f}, {100 * np.quantile(bs, 0.975):+6.1f}]  '
          f'({len(d)} clusters; {np.sum(d > 0)} better / {np.sum(d < 0)} worse)')
