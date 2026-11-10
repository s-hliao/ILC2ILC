"""long_score.py (2026-11-06, runs/screen/CAR_SIGNIF_1106.md): the longer-practice (LONG_*) and RMA-fix (ZP_*) runs' held-out peanut
evaluation (long_eval.json) per tire and arm: score change vs the run's own zero-shot at iterations 3, 5, 8 (mean of each run's cars;
a failed lap scores 5), bootstrap 95 % CI over runs, runs better, e_y and pace; and B's rule: ZP RMA vs LONG RMA at it 3, paired by seed."""
import glob, json, os
import numpy as np
rng = np.random.default_rng(0)
def load(tag, tire, net):
    out = {}
    for d in sorted(glob.glob(f'runs/screen/{tag}_{tire}_{net}_s*_ss9001')):
        f = f'{d}/long_eval.json'
        if os.path.exists(f):
            j = json.load(open(f)); pl = list(j)[0]; out[d.split('_s')[-2][-1] if False else d.split(f'{net}_')[1]] = j[pl]
    return out
def mean_of(cell, q):
    return float(np.mean([v[q] for v in cell.values()]))
def ci(x):
    x = np.asarray(x); b = [np.mean(rng.choice(x, len(x))) for _ in range(4000)]
    return np.percentile(b, [2.5, 97.5])
for tire in ('plastic', 'rubber'):
    for tag, net, name in (('LONG', 'freePPON', 'PPO+DR'), ('LONG', 'rmaN', 'RMA'), ('ZP', 'rmaN', 'RMA + z-path fix')):
        R = load(tag, tire, net)
        if not R: continue
        z0 = {s: mean_of(r['0'], 'score') for s, r in R.items()}
        line = f'{tire:7s} {name:17s} n={len(R)} zero-shot {np.mean(list(z0.values())):.2f}'
        for it in ('3', '5', '8'):
            ch = [100 * (mean_of(r[it], 'score') / z0[s] - 1) for s, r in R.items() if it in r]
            if not ch: continue
            lo, hi = ci(ch)
            ey0 = np.mean([mean_of(r['0'], 'rms_ey') for r in R.values()]); ey = np.mean([mean_of(r[it], 'rms_ey') for r in R.values()])
            pc0 = np.mean([mean_of(r['0'], 'pace') for r in R.values()]); pc = np.mean([mean_of(r[it], 'pace') for r in R.values()])
            line += f' | it{it} {np.mean(ch):+.1f} % [{lo:+.1f}, {hi:+.1f}] better {sum(c < 0 for c in ch)}/{len(ch)} e_y {100*ey0:.1f}->{100*ey:.1f} pace {pc0:.3f}->{pc:.3f}'
        print(line)
    A, B = load('LONG', tire, 'rmaN'), load('ZP', tire, 'rmaN')
    common = sorted(set(A) & set(B))
    if common:
        d = [mean_of(B[s]['3'], 'score') / mean_of(B[s]['0'], 'score') - mean_of(A[s]['3'], 'score') / mean_of(A[s]['0'], 'score') for s in common]
        print(f'{tire:7s} rule B: ZP - LONG RMA it3 change, paired over {len(common)} seeds: {100*np.mean(d):+.1f} pts (negative = the fix is better)')
