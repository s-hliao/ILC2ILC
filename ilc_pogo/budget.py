#!/usr/bin/env python3
"""
budget.py: the minimum hardware budget for the maximal result, from the budget-curve runs (scheduler_fix 'budget':
runs/v3/<arm>_s<seed>_curve/summary_it<k>.json and summary.json, validation set).
Per arm, seeds pooled: success and its cluster-bootstrap 95 % CI at every budget (trials per robot = 6 x iterations).
The plateau = the budget >= 48 trials with the highest success. B* = the smallest budget that is not worse than the
plateau: the paired cluster-bootstrap CI of (success(B) - success(plateau)) contains 0 and the mean gap is <= TOL.
Writes runs/v3/budget.json and prints the table.
    python budget.py [--kind curve|g3|g2|T150] [--file summary_test.json]   (the frozen test pass reads summary_test / summary_it<k>_test)
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ('nom', 'drLK20', 'ppo_dr')
TRIAL_S = 3.0
TOL = 0.02          # B*: within 2 points of the plateau AND not significantly worse (the CI alone is too loose at n = 3 seeds)


def curves(root='runs/v3', suffix='', kind='curve'):
    out = {}
    for arm in ARMS:
        by_b = {}
        for d in sorted(glob.glob(os.path.join(HERE, root, f'{arm}_s[0-9]_{kind}'))):
            files = glob.glob(os.path.join(d, f'summary_it*{suffix}.json')) + [os.path.join(d, f'summary{suffix}.json')]
            for f in files:
                if not os.path.exists(f):
                    continue
                s = json.load(open(f))
                if 'eval' not in s:
                    continue
                b = int(np.mean(list(s['real_trials'].values())))
                by_b.setdefault(b, []).append(s)
        if by_b:
            out[arm] = dict(sorted(by_b.items()))
    return out


def analyse(cv):
    res = {}
    for arm, by_b in cv.items():
        rows = {}
        for b, ss in by_b.items():
            m, lo, hi = ci.cluster_ci(ss)
            seeds = [float(np.mean([mm['success'] for r in s['runs'] for mm in s['runs'][r]])) for s in ss]
            rows[b] = dict(success=m, lo=lo, hi=hi, seeds=seeds, n_seeds=len(ss))
        big = [b for b in rows if b >= 48] or list(rows)
        plat = max(big, key=lambda b: rows[b]['success'])
        bstar = None
        for b in sorted(rows):
            if len(by_b[b]) != len(by_b[plat]):
                continue
            d, lo, hi = ci.paired_ci(by_b[b], by_b[plat])
            rows[b]['vs_plateau'] = (d, lo, hi)
            if bstar is None and hi >= 0 and d >= -TOL:
                bstar = b
        res[arm] = dict(rows=rows, plateau=plat, bstar=bstar)
    return res


def main():
    suffix = ''
    kind = sys.argv[sys.argv.index('--kind') + 1] if '--kind' in sys.argv else 'curve'
    trial_s = 1.5 if kind == 'T150' else TRIAL_S
    if '--file' in sys.argv:
        suffix = sys.argv[sys.argv.index('--file') + 1].replace('summary', '').replace('.json', '')
    res = analyse(curves(suffix=suffix, kind=kind))
    for arm, r in res.items():
        print(f'== {arm} [{kind}]: plateau {r["plateau"]} trials, B* = {r["bstar"]} trials ({(r["bstar"] or 0) * trial_s:.0f} s)')
        for b, q in r['rows'].items():
            vp = q.get('vs_plateau')
            print(f'  {b:4d} trials  {100 * q["success"]:5.1f} % [{100 * q["lo"]:5.1f}, {100 * q["hi"]:5.1f}]  seeds '
                  + ' / '.join(f'{100 * x:.0f}' for x in q['seeds'])
                  + (f'   vs plateau {100 * vp[0]:+5.1f} [{100 * vp[1]:+5.1f}, {100 * vp[2]:+5.1f}]' if vp else ''))
    json.dump({k: dict(v, rows={str(b): q for b, q in v['rows'].items()}) for k, v in res.items()},
              open(os.path.join(HERE, 'runs/v3', f'budget_{kind}{suffix}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
