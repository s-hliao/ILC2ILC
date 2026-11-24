"""Zero-shot fall rate and tracking error vs reference speed, per perturbation (where does the walker start falling?).
    ~/venvs/pg0/bin/python scripts/speed_boundary.py --params runs/walker_nom_s0/params.pkl --cases payload4R lockFR --out results/speed_boundary.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_go1lla import walkenv as WE, srb as S
from zeroshot_cases import CASES
ap = argparse.ArgumentParser()
ap.add_argument('--params', required=True)
ap.add_argument('--cases', nargs='+', required=True)
ap.add_argument('--speeds', type=float, nargs='+', default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
ap.add_argument('--walks', type=int, default=6)
ap.add_argument('--dist', type=float, default=2.5)
ap.add_argument('--out', required=True)
a = ap.parse_args()
rng = np.random.default_rng(100)
ang = np.radians(rng.uniform(-10, 10, a.walks))
res = {}
for c in a.cases:
    w = WE.Walker(a.params, **CASES[c])
    for v in a.speeds:
        n = int((0.5 + a.dist / v + v + 1.0) / 0.02)
        n = int(np.ceil(n / 10) * 10)
        rows = []
        for th in ang:
            g = np.array([a.dist * np.cos(th), a.dist * np.sin(th), v])
            out = w.rollout(g, n=n)
            e = S.measured_rows(out, g, W=1)
            rows.append((WE.fell(out), 5 * np.sqrt(np.mean(e[0::S.NR] ** 2 + e[1::S.NR] ** 2))))
        falls = sum(f for f, _ in rows); ok = [t for f, t in rows if not f]
        r = dict(case=c, v=v, falls=falls, n=len(rows), track_cm=float(np.mean(ok)) if ok else None)
        res.setdefault(c, []).append(r)
        print(json.dumps(r), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(res, open(a.out, 'w'), indent=1)
