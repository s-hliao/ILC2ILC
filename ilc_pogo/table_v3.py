#!/usr/bin/env python3
"""
table_v3.py [--root runs/v3] [--file summary.json] [ARM_GROUP ...]: the results table with seeds and confidence
intervals. Arms runs/<root>/<name>_s<k><rest>/ are grouped by <name><rest> (seeds pooled).
Columns: seeds; success mean over seeds +- seed std; 95 % cluster-bootstrap CI over (robot, goal) clusters, seeds
pooled (ci.py); success inside / outside the DR family; RMS apex error (cm); |speed error| (m/s); real trials per
robot. --file summary_test.json reads the frozen test pass (reeval.py).
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
IN_FAM, OUT_FAM = ('real_nom', 'real_mass', 'real_spring'), ('real_act', 'real_offset', 'real_prog')


def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def groups(root, fname):
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, root, '*', fname)) + glob.glob(os.path.join(HERE, root, '*', 'it*', fname))):
        dd = os.path.dirname(f)
        arm = os.path.basename(dd)
        if arm.startswith('it') and arm[2:].isdigit():                  # a saved checkpoint of a curve run
            arm = f'{os.path.basename(os.path.dirname(dd))}_{arm}'
        g = re.sub(r'_s\d+', '', arm, count=1)
        s = json.load(open(f))
        if 'eval' not in s:
            continue
        if not isinstance(s.get('runs'), dict) and isinstance(s.get('per_run'), dict):
            s['runs'] = s['per_run']
        out.setdefault(g, []).append((arm, s))
    return out


def row(ss):
    succ = [np.mean([m['success'] for r in s['runs'] for m in s['runs'][r]]) for _, s in ss]
    m, lo, hi = ci.cluster_ci([s for _, s in ss])
    allm = [m_ for _, s in ss for r in s['runs'] for m_ in s['runs'][r]]
    ok = [q for q in allm if not q['failed']]
    fam = lambda grp: 100 * np.mean([q['success'] for _, s in ss for r in grp if r in s['runs'] for q in s['runs'][r]])
    tr = [np.mean(list(s.get('real_trials', {}).values()) or [0]) for _, s in ss]
    return dict(n=len(ss), mean=100 * np.mean(succ), sd=100 * np.std(succ), lo=100 * lo, hi=100 * hi, infam=fam(IN_FAM),
                outfam=fam(OUT_FAM), apex=100 * np.mean([q['rms_apex'] for q in ok if q['rms_apex'] is not None]) if ok else np.nan,
                verr=np.mean([abs(q['v_err']) for q in ok if q['v_err'] is not None]) if ok else np.nan,
                trials=int(np.mean(tr)), seeds=[round(100 * x) for x in succ])


def main():
    root, fname = arg('--root', 'runs/v3'), arg('--file', 'summary.json')
    want = [a for a in sys.argv[1:] if not a.startswith('--') and a not in (root, fname)]
    gs = groups(root, fname)
    print(f'{root} / {fname}')
    hdr = f'{"arm":<28}| seeds | succ% mean+-sd | 95% CI       | in-DR% | out-DR% | apex cm | |dv| m/s | trials | per seed'
    print(hdr)
    print('-' * len(hdr))
    for g in (want or sorted(gs)):
        if g not in gs:
            continue
        q = row(gs[g])
        print(f'{g:<28}| {q["n"]:5d} | {q["mean"]:5.1f} +- {q["sd"]:4.1f} | [{q["lo"]:4.1f}, {q["hi"]:5.1f}] | {q["infam"]:6.1f} | '
              f'{q["outfam"]:7.1f} | {q["apex"]:7.2f} | {q["verr"]:8.3f} | {q["trials"]:6d} | {q["seeds"]}')


if __name__ == '__main__':
    main()
