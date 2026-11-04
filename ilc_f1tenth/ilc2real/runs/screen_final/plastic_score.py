"""plastic_score.py ROOT: the plastic replicates (PLSL = stage seed 9001, PLSLr2 9002, PLSLr3 9003; network seeds 0-2; 12 plastic cars), eval seed 701
(screen_eval_701.json): % nerr change of it 3 (6 real laps) vs the run's own it 0 (zero-shot) per (network seed, stage seed, car) cell;
mean [bootstrap 95 % CI over cells], cells better, crash rates. Same protocol as the rubber table (screen_score.py)."""
import sys, os, json, glob, numpy as np
root = sys.argv[1]; rng = np.random.default_rng(0)
for net in ('freePPON', 'rmaN'):
    fs = sorted(glob.glob(f'{root}/PLSL*_{net}_s*/screen_eval_701.json'))
    if not fs: continue
    tags = [f.split('/')[-2] for f in fs]
    E = [json.load(open(f)) for f in fs]
    print(f'{net}: {len(fs)} runs {tags}')
    for pn in sorted(set.intersection(*[set(e) for e in E])):
        d, z, s, cz, cs = [], [], [], [], []
        for e in E:
            if '3' not in e[pn]: continue
            for c in e[pn]['0']:
                a, b = e[pn]['0'][c], e[pn]['3'][c]
                d.append(100 * (b / a - 1)); z.append(a); s.append(b); cz.append(a >= 5); cs.append(b >= 5)
        d = np.array(d); bs = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
        print(f'   {pn.replace("mocap_", "").replace("_b0_free", ""):18s} zero-shot {np.mean(z):.2f} -> stage {np.mean(s):.2f}: {d.mean():+5.1f} % [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]'
              f'  better {int((d < 0).sum())}/{len(d)}  crash {100 * np.mean(cz):.0f} -> {100 * np.mean(cs):.0f} %')
