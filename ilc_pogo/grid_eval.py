#!/usr/bin/env python3
"""grid_eval.py --arm DIR [--zs POLICY]: success / apex / speed error of an arm's per-robot networks over a grid of
goals (the goal box, 7 x 5), from 2 perturbed drops x 10 s each; with --zs also the starting network -> DIR/grid.json and
a goal-plane figure DIR/grid.png (the improvement map: does correcting some goals correct the surrounding ones?)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib                                  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                    # noqa: E402
import numpy as np                                 # noqa: E402

import common as C                                 # noqa: E402
import policies                                    # noqa: E402
import real_pogo as rp                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--arm', required=True)
ap.add_argument('--zs', required=True)
ap.add_argument('--robots', nargs='+', default=rp.DEFAULT_ROBOTS)
a = ap.parse_args()
vs, hs = np.linspace(0.0, 1.5, 7), np.linspace(0.55, 0.75, 5)
grid = np.array([[v, h] for h in hs for v in vs])
s = json.load(open(os.path.join(a.arm, 'summary.json')))
trained = np.array(s['goals_train'])
out = {}
for tag, pol_of in (('zs', lambda r: policies.load_any(a.zs)), ('adapted', lambda r: policies.load_any(os.path.join(a.arm, f'{r}_policy.npz')))):
    res = rp.evaluate({r: pol_of(r) for r in a.robots}, grid, a.robots, runs=2, seed=4242)
    err = np.zeros((len(a.robots), len(grid)))
    suc = np.zeros_like(err)
    for i, r in enumerate(a.robots):
        for j in range(len(grid)):
            ms = res[r][2 * j:2 * j + 2]
            suc[i, j] = np.mean([m['success'] for m in ms])
            ok = [m for m in ms if m['rms_apex'] is not None]
            err[i, j] = np.mean([m['rms_apex'] / 0.02 + abs(m['v_err']) / 0.1 for m in ok]) if ok else np.nan
    out[tag] = dict(success=suc.tolist(), err=err.tolist())
out.update(grid=grid.tolist(), trained=trained.tolist(), robots=a.robots)
json.dump(out, open(os.path.join(a.arm, 'grid.json'), 'w'))
zs, ad = np.array(out['zs']['err']), np.array(out['adapted']['err'])
ratio = np.nanmean(ad, 0) / np.nanmean(zs, 0)
d = np.min(np.linalg.norm((grid[:, None] - trained[None]) * np.array([1 / 1.5, 1 / 0.2]), axis=2), 1)
for dd, rr, s0, s1 in sorted(zip(d, ratio, np.mean(out['zs']['success'], 0), np.mean(out['adapted']['success'], 0))):
    print(f'dist {dd:.2f}  error ratio adapted/zero-shot {rr:.2f}  success {100 * s0:5.1f} -> {100 * s1:5.1f}')
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, (lab, val, cm, lim) in zip(axs, (('normalized error, adapted / zero-shot (mean of 6 robots)', ratio, 'RdBu_r', (0, 2)),
                                         ('success after, % (6 robots)', 100 * np.mean(out['adapted']['success'], 0), 'Blues', (0, 100)))):
    sc = ax.scatter(grid[:, 0], grid[:, 1], c=val, cmap=cm, vmin=lim[0], vmax=lim[1], s=160, edgecolor='0.3', linewidth=0.5)
    ax.scatter(trained[:, 0], trained[:, 1], marker='*', s=220, color='k', label='trained goal(s)')
    ax.set_xlabel('v* (m/s)'); ax.set_title(lab, loc='left', fontsize=8)
    fig.colorbar(sc, ax=ax, shrink=0.85)
axs[0].set_ylabel('h* (m)'); axs[0].legend(fontsize=7, loc='lower left', bbox_to_anchor=(0, 1.08), frameon=False)
fig.savefig(os.path.join(a.arm, 'grid.png'), dpi=150, bbox_inches='tight')
