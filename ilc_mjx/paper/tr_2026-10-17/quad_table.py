#!/usr/bin/env python3
"""
quad_table.py (2026-10-18): the quad hardware-stage arms of tr_arms.sh / noise_arms.sh / landing_arms.sh on the
validation goals. Score = 0.01 x landing cost + 20 per fall (summary.json planeval_final), LOWER is better. The 95 % CI
is a bootstrap over the (seed, robot) cells; paired differences are against the recipe on the same cells. "cos" is the
median cosine of the sensitivity's predicted vs the measured landing change (summary.json pred): the landing model's
direction accuracy, which the landing_arms.sh fixes target.
    python quad_table.py [--dir log/dilc/plane/tr] [arm ...]      (default: every arm found)
"""
import glob
import json
import os
import sys

import numpy as np

D = sys.argv[sys.argv.index('--dir') + 1] if '--dir' in sys.argv else '/home/henry/ilc_ws/log/dilc/plane/tr'
ROBOTS = ('real_r1', 'real_s1', 'real_r4', 'real_r5')
ORDER = ['recipe', 'tr', 'tr_r20', 'recipe_r2', 'tr_r2', 'jac_real', 'sens_goal', 'sens_shared', 'sens_vres', 'nbr05',
         'ctl_random']


def cells(arm):
    """{(seed, robot): summary} for one arm."""
    out = {}
    for d in glob.glob(os.path.join(D, f'{arm}_sd*')):
        if not os.path.isdir(d):
            continue
        sd = d.rsplit('_sd', 1)[1]
        for rb in ROBOTS:
            f = os.path.join(d, rb, 'summary.json')
            if os.path.exists(f):
                out[(sd, rb)] = json.load(open(f))
    return out


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    return x.mean(), *np.percentile(x[r].mean(1), [2.5, 97.5])


def main():
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith('--') and sys.argv[i - 1] != '--dir']
    found = sorted({os.path.basename(d).rsplit('_sd', 1)[0] for d in glob.glob(os.path.join(D, '*_sd*')) if os.path.isdir(d)})
    arms = args or [a for a in ORDER if a in found] + [a for a in found if a not in ORDER]
    data = {a: cells(a) for a in arms}
    zs = [s['planeval_start'] for c in data.values() for s in c.values() if s.get('planeval_start') is not None]
    print(f'quad, validation goals; score lower is better; dir {D}')
    if zs:
        print(f'zero-shot (start) {np.mean(zs):.2f} over {len(zs)} cells')
    hdr = f'{"arm":<12} | {"final":>6} [95% CI]       | ' + ' | '.join(f'{r[5:]:>5}' for r in ROBOTS) + \
        ' | perturbed | falls | cos med | cells | paired vs recipe [95% CI]'
    print(hdr + '\n' + '-' * len(hdr))
    base = data.get('recipe', {})
    for a in arms:
        c = data[a]
        if not c:
            print(f'{a:<12} | (missing)')
            continue
        m, lo, hi = boot([s['planeval_final'] for s in c.values()])
        per = ' | '.join(f'{np.mean([s["planeval_final"] for (sd, r), s in c.items() if r == rb]):5.1f}'
                         if any(r == rb for _, r in c) else '    -' for rb in ROBOTS)
        pert = np.mean([s['rob_final'] for s in c.values()])
        falls = sum(s['planeval_final_falls'] for s in c.values())
        cos = [p['cos'] for s in c.values() for p in (s.get('pred') or []) if 'cos' in p]
        common = sorted(set(c) & set(base))
        pd = ''
        if a != 'recipe' and common:
            d, dlo, dhi = boot([c[k]['planeval_final'] - base[k]['planeval_final'] for k in common])
            pd = f'{d:+6.2f} [{dlo:+6.2f}, {dhi:+6.2f}] ({len(common)})'
        print(f'{a:<12} | {m:6.2f} [{lo:6.2f}, {hi:6.2f}] | {per} | {pert:9.2f} | {falls:5d} | '
              f'{np.median(cos) if cos else np.nan:7.2f} | {len(c):5d} | {pd}')


if __name__ == '__main__':
    main()
