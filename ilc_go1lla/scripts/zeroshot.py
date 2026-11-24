"""Zero-shot walks of a walker under the LLA-MPPI-style perturbations: falls, path deviation, final error, time.

Walks: start at the origin facing +x, goals at (L, y) with L 2.5-3.5 m, |y| <= 0.5 m (a lab-sized version of their 7 m walk).
    ~/venvs/pg0/bin/python scripts/zeroshot.py --params runs/walker_nom_s0/params.pkl --out results/zs_nom_s0.json
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ilc_go1lla import walkenv as WE, srb as S

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeroshot_cases import CASES

ap = argparse.ArgumentParser()
ap.add_argument('--params', required=True)
ap.add_argument('--cases', nargs='*', default=list(CASES))
ap.add_argument('--walks', type=int, default=8)
ap.add_argument('--steps', type=int, default=300)
ap.add_argument('--out', required=True)
a = ap.parse_args()
rng = np.random.default_rng(100)
goals = np.stack([rng.uniform(2.5, 3.5, a.walks), rng.uniform(-0.6, 0.6, a.walks)], 1)
res = {}
for c in a.cases:
    t0 = time.time(); w = WE.Walker(a.params, **CASES[c]); rows = []
    for g in goals:
        out = w.rollout(g, n=a.steps)
        e = S.measured_rows(out, g, W=1)                     # per step: cross, along (/5 cm), heading (/0.1 rad)
        cross, along = 5 * e[0::S.NR], 5 * e[1::S.NR]
        dist = np.linalg.norm(out['p'][:, :2] - g, axis=1)
        reach = np.where(dist < 0.15)[0]
        rows.append(dict(goal=g.round(3).tolist(), fell=WE.fell(out), cross_rms_cm=float(np.sqrt(np.mean(cross ** 2))),
                         along_rms_cm=float(np.sqrt(np.mean(along ** 2))), track_rms_cm=float(np.sqrt(np.mean(cross ** 2 + along ** 2))),
                         final_cm=float(100 * dist[-1]), t_reach=float(reach[0] * 0.02) if len(reach) else None,
                         z_mean=float(out['p'][50:, 2].mean())))
    ok = [r for r in rows if not r['fell']]
    s = dict(case=c, falls=sum(r['fell'] for r in rows), n=len(rows),
             track_rms_cm=float(np.mean([r['track_rms_cm'] for r in ok])) if ok else None,
             cross_rms_cm=float(np.mean([r['cross_rms_cm'] for r in ok])) if ok else None,
             along_rms_cm=float(np.mean([r['along_rms_cm'] for r in ok])) if ok else None,
             final_cm=float(np.median([r['final_cm'] for r in ok])) if ok else None,
             reached=sum(r['t_reach'] is not None for r in ok),
             t_reach=float(np.median([r['t_reach'] for r in ok if r['t_reach'] is not None])) if any(r['t_reach'] for r in ok) else None,
             z_mean=float(np.mean([r['z_mean'] for r in ok])) if ok else None, wall_s=time.time() - t0)
    res[c] = dict(summary=s, rows=rows)
    print(json.dumps(s), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(res, open(a.out, 'w'), indent=1)
