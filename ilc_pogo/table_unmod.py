#!/usr/bin/env python3
"""
table_unmod.py: the unmodelled-dynamics suite (scheduler_unmod.py, runs/v5) -- success per robot (seeds pooled), the
95 % cluster-bootstrap CI (ci.py: (robot, goal) clusters) over the 5 unmodelled robots, and the same arm on real_nom
(the existing runs; same validation goals and drops) for reference. Paired differences vs ours nominal + 36 trials.
    python table_unmod.py [--file summary.json | summary_it2.json | summary_it4.json]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ci                                       # noqa: E402
import real_pogo as rp                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
UNM = rp.UNMODELLED_ROBOTS
ARMS = [  # label, suite run (per seed), real_nom reference run (per seed)
    ('ours nominal, zero-shot', 'v5/nomzs_s{s}', 'v3/nom_s{s}_zs'),
    ('ours nominal + 36 trials', 'v5/nomhw36_s{s}', 'v4/fix10_nom_s{s}'),
    ('PPO+DR, zero-shot', 'v5/ppo_unm_s{s}_ep0', 'v3/ppo_dr_s{s}_zs'),
    ('PPO+DR + ours 36 trials', 'v5/ppodrhw36_s{s}', 'v4/fix10_ppo_dr_s{s}'),
    ('RMA (fixed, rmaR1)', 'v5/rmaR1_s{s}_zs', 'v3/rmaR1_s{s}_zs'),
    ('FADA (fixed, F2), zero-shot', 'v5/fadaF2_s{s}_zs', 'v3/fadaF2_s{s}_zs'),
    ('FADA (fixed, F2) + 24 trials', 'v5/fadaF2_s{s}_hw24', 'v3/fadaF2_s{s}_hw24'),
    ('PPO on the robots, 48 trials', 'v5/ppo_unm_s{s}_ep48', 'v3/ppo_s{s}_ep48'),
]


def load(pat, robots, fname):
    out = []
    for s in (0, 1, 2):
        d = os.path.join(HERE, 'runs', pat.format(s=s))
        f = os.path.join(d, fname if os.path.basename(pat.format(s=s)).startswith(('nomhw', 'ppodrhw', 'fix10')) else 'summary.json')
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        runs = j['runs'] if isinstance(j.get('runs'), dict) else j['per_run']
        out.append({'runs': {r: runs[r] for r in robots if r in runs}})
    return out


def succ(ss, r):
    v = [m['success'] for s in ss for m in s['runs'].get(r, [])]
    return 100 * np.mean(v) if v else np.nan


def main():
    fname = sys.argv[sys.argv.index('--file') + 1] if '--file' in sys.argv else 'summary.json'
    hdr = f'{"arm":<30}| ' + ' | '.join(f'{r[5:]:>6}' for r in UNM) + ' | 5 unmodelled [95% CI]  | real_nom [95% CI]      | seeds'
    print(f'success % (validation goals, 8 goals x 4 drops per robot, 3 seeds pooled); hw arms read {fname}')
    print(hdr)
    print('-' * len(hdr))
    data = {}
    for lab, pat, ref in ARMS:
        ss = load(pat, UNM, fname)
        rs = load(ref, ['real_nom'], 'summary.json')
        data[lab] = ss
        if not ss:
            print(f'{lab:<30}| (missing)')
            continue
        m, lo, hi = ci.cluster_ci(ss)
        mr, lor, hir = ci.cluster_ci(rs) if rs else (np.nan,) * 3
        print(f'{lab:<30}| ' + ' | '.join(f'{succ(ss, r):6.1f}' for r in UNM) +
              f' | {100 * m:5.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}] | {100 * mr:5.1f} [{100 * lor:5.1f}, {100 * hir:5.1f}] | {len(ss)}')
    print('\npaired difference (row - ours nominal + 36), (robot, goal) clusters over the 5 unmodelled robots:')
    base = data.get('ours nominal + 36 trials')
    for lab, ss in data.items():
        if ss and base and lab != 'ours nominal + 36 trials':
            d, lo, hi = ci.paired_ci(ss, base)
            per = ' '.join(f'{r[5:]} {succ(ss, r) - succ(base, r):+.0f}' for r in UNM)
            print(f'  {lab:<30} {100 * d:+6.1f} [{100 * lo:+6.1f}, {100 * hi:+6.1f}]   per robot: {per}')


if __name__ == '__main__':
    main()
