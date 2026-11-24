"""Step-0 positivity of the SRB-linearized GN step for the joint-target Go1 walker under a perturbation.

Per walk: fly the perturbed ("real") robot; G = d rows / d du from the nominal SRB along the measured walk (ilc_go1lla/srb.py:
measured-saturation motors, rows cross/along/heading/roll/pitch); du = -beta (G'G + lam I)^-1 G'e, lam = delta tr(G'G)/n,
rms-capped; re-fly with du added to the actions; compare the predicted row change G du with the measured one.
A walk that falls is used up to FALL_MARGIN steps before the fall (rows and du only there); its fall time is compared.
A repeat-walk noise floor is also reported (the same walk with a 1e-4 action kick).

    ~/venvs/pg0/bin/python scripts/positivity.py --params runs/walker_nom_s0/params.pkl --case lockRL --speed 0.3 --max-lag 25 --out results/pos_lockRL.json
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ilc_go1lla import walkenv as WE, srb as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zeroshot_cases import CASES

FALL_MARGIN = 10

ap = argparse.ArgumentParser()
ap.add_argument('--params', required=True)
ap.add_argument('--case', required=True)
ap.add_argument('--speed', type=float, default=0.8)
ap.add_argument('--walks', type=int, default=6)
ap.add_argument('--steps', type=int, default=300)
ap.add_argument('--beta', type=float, default=0.5)
ap.add_argument('--delta', type=float, default=0.05)
ap.add_argument('--cap', type=float, default=0.15, help='rms cap of du (action units; 1 = 0.5 rad of joint target)')
ap.add_argument('--window', type=int, default=10)
ap.add_argument('--max-lag', type=int, default=0, help='> 0: drop G entries whose input precedes the row window end by more than this many steps')
ap.add_argument('--no-sat', action='store_true', help='ignore the measured motor saturation (the model assumes every motor works)')
ap.add_argument('--out', required=True)
a = ap.parse_args()
w = WE.Walker(a.params, **CASES[a.case])
rng = np.random.default_rng(100)
goals = np.stack([rng.uniform(2.5, 3.5, a.walks), rng.uniform(-0.6, 0.6, a.walks), np.full(a.walks, a.speed)], 1)
NR, W = S.NR, a.window
res = []
for i, g in enumerate(goals):
    t0 = time.time()
    out = w.rollout(g, n=a.steps)
    kf = S.fall_index(out)
    n_use = a.steps if kf is None else ((kf - FALL_MARGIN - 1) // W) * W + 1
    if n_use < 2 * W + 1:
        res.append(dict(walk=i, fell=True, usable=False, t_fall=kf * 0.02)); print(json.dumps(res[-1]), flush=True); continue
    cut = {k: v[:n_use] for k, v in out.items()}
    e_meas = S.measured_rows(cut, g, W=W)
    rows, G = S.make_G(w, cut, g, W=W, measured_sat=not a.no_sat)
    t_g = time.time() - t0
    if a.max_lag > 0:
        nr = G.shape[0] // NR
        Gr = np.array(G).reshape(nr, NR, n_use, 12)
        for j in range(nr):
            Gr[j, :, :max(0, (j + 1) * W - a.max_lag)] = 0.0
        G = Gr.reshape(G.shape)
    N = G.T @ G
    lam = a.delta * np.trace(N) / N.shape[0]
    du = -a.beta * np.linalg.solve(N + lam * np.eye(N.shape[0]), G.T @ rows)
    rms = np.sqrt(np.mean(du ** 2))
    if rms > a.cap:
        du *= a.cap / rms
    pred = G @ du
    du_full = np.zeros((a.steps, 12)); du_full[:n_use] = du.reshape(n_use, 12)
    out2 = w.rollout(g, n=a.steps, du=du_full)
    kick = np.zeros((a.steps, 12)); kick[5] = 1e-4
    out3 = w.rollout(g, n=a.steps, du=kick)
    kf2 = S.fall_index(out2)
    m2 = min(n_use, kf2 + 1) if kf2 is not None else n_use     # compare only where the new walk is still up
    m2 = ((m2 - 1) // W) * W + 1
    if m2 < W + 1:
        r = dict(walk=i, fell=kf is not None, t_fall=None if kf is None else kf * 0.02, fell_early_after=True,
                 t_fall_after=kf2 * 0.02)
        res.append(r); print(json.dumps(r), flush=True); continue
    e2 = S.measured_rows({k: v[:m2] for k, v in out2.items()}, g, W=W)
    e1 = e_meas[:len(e2)]
    e3 = S.measured_rows({k: v[:m2] for k, v in out3.items()}, g, W=W)
    pr = pred[:len(e2)]
    meas = e2 - e1
    cos = float(pr @ meas / (np.linalg.norm(pr) * np.linalg.norm(meas) + 1e-12))
    track = lambda e: float(5.0 * np.sqrt(np.mean(e[0::NR] ** 2 + e[1::NR] ** 2)))     # cm (cross + along)
    att = lambda e: float(np.degrees(0.1) * np.sqrt(np.mean(e[3::NR] ** 2 + e[4::NR] ** 2)))   # deg (roll + pitch)
    sat = S.saturation(cut, w.env)
    r = dict(walk=i, goal=g.round(3).tolist(), fell=kf is not None, t_fall=None if kf is None else kf * 0.02,
             fell_after=kf2 is not None, t_fall_after=None if kf2 is None else kf2 * 0.02, n_use=n_use,
             rows_match=float(np.abs(rows - e_meas).max()), cos=cos, gain=float(meas @ pr / (pr @ pr + 1e-12)),
             du_rms=float(np.sqrt(np.mean(du ** 2))), track_before=track(e1), track_after=track(e2),
             att_before=att(e1), att_after=att(e2), noise_floor=float(np.linalg.norm(e3 - e1)),
             meas_norm=float(np.linalg.norm(meas)), pred_norm=float(np.linalg.norm(pr)),
             sat_frac_per_leg=np.round(sat.reshape(-1, 4, 3).mean((0, 2)), 2).tolist(), G_s=t_g)
    res.append(r); print(json.dumps(r), flush=True)
ok = [r for r in res if 'cos' in r]
summ = dict(case=a.case, speed=a.speed, n=len(res), n_scored=len(ok),
            falls_before=sum(bool(r.get('fell')) for r in res), falls_after=sum(bool(r.get('fell_after') or r.get('fell_early_after')) for r in res),
            cos_median=float(np.median([r['cos'] for r in ok])) if ok else None,
            positive=float(np.mean([r['cos'] > 0 for r in ok])) if ok else None,
            track_before=float(np.mean([r['track_before'] for r in ok])) if ok else None,
            track_after=float(np.mean([r['track_after'] for r in ok])) if ok else None,
            att_before=float(np.mean([r['att_before'] for r in ok])) if ok else None,
            att_after=float(np.mean([r['att_after'] for r in ok])) if ok else None)
print('SUMMARY', json.dumps(summ), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(dict(summary=summ, rows=res, args=vars(a)), open(a.out, 'w'), indent=1)
