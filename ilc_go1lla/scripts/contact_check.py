"""(2026-11-24) Sim tests of the contact-variable model (srbc.make_G_contact) vs the fixed-schedule model (srb.make_G), goal walker.

Per test robot (--cases) and goal (--bearings, --dist), one walk of the base network; the walk is cut 10 steps before a fall (as the
stage does). For each model: rows (time rows vs the nominal walk), G, the damped GN step (0.5 s lag truncation, cap); then
  positivity   du applied open-loop on the same robot: cos(G du, actual row change), J before / predicted / after, fall before / after
  calibration  the same step computed from a walk whose mocap is miscalibrated (trunk tilted --tilt-deg about x, shifted
               --dz / --dx): cos(du_true, du_miscal) and the relative change of du
    python scripts/contact_check.py runs/goalwalk/L_vel2_s0/params.pkl --cases payload3R foamE:0.1,0.05,2.0
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_go1lla import goalstage as GS, srb as S, srbc as SC   # noqa: E402
from zeroshot_cases import parse_case                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('params'); ap.add_argument('--variant', default='vel2')
ap.add_argument('--cases', nargs='+', default=['payload3R', 'foamE:0.1,0.05,2.0'])
ap.add_argument('--bearings', type=float, nargs='+', default=[0, 90, 180, -135])
ap.add_argument('--dist', type=float, default=1.0)
ap.add_argument('--rows', default='time')
ap.add_argument('--beta', type=float, default=0.5); ap.add_argument('--delta', type=float, default=0.05)
ap.add_argument('--cap', type=float, default=0.03); ap.add_argument('--max-lag', type=int, default=25); ap.add_argument('--W', type=int, default=10)
ap.add_argument('--tilt-deg', type=float, default=3.0); ap.add_argument('--dz', type=float, default=0.02); ap.add_argument('--dx', type=float, default=0.01)
ap.add_argument('--out', default='')
a = ap.parse_args()
W = a.W
nom = GS.GoalSimWalker(a.params, a.variant)


def gn_step(rows, G, n_use, nr):
    G = np.array(G)
    nw = G.shape[0] // nr
    Gr = G.reshape(nw, nr, n_use, 12)
    for j in range(nw):
        Gr[j, :, :max(0, (j + 1) * W - a.max_lag)] = 0.0
    G = Gr.reshape(G.shape)
    N = G.T @ G; lam = a.delta * np.trace(N) / N.shape[0]
    du = -a.beta * np.linalg.solve(N + lam * np.eye(N.shape[0]), G.T @ rows)
    r = np.sqrt(np.mean(du ** 2))
    return du * min(1.0, a.cap / max(r, 1e-12)), G


def miscal(out):
    """The walk as a miscalibrated mocap would report it: trunk pose tilted / shifted by a fixed rigid offset."""
    th = np.radians(a.tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    o = dict(out)
    R = np.stack([np.asarray(S.quat_R(q)) for q in out['quat']]) @ Rx
    o['quat'] = np.stack([np.r_[1.0, 0, 0, 0] if False else _R2q(r) for r in R]).astype(out['quat'].dtype)
    o['p'] = out['p'] + np.array([a.dx, 0.0, a.dz], out['p'].dtype)
    return o


def _R2q(R):
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1); q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]); q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = 0.25 * s; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q); return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


res = []
for case in a.cases:
    real = GS.GoalSimWalker(a.params, a.variant, **parse_case(case))
    kin = S.LegKin(real.env.mj_model)
    for b in a.bearings:
        g = np.array([a.dist * np.cos(np.radians(b)), a.dist * np.sin(np.radians(b))], np.float32)
        n = GS.walk_steps(g, W)
        ref = GS.reference(nom.rollout(g, n))
        rf = GS.row_fn_for(a.rows, g, ref); nr = GS.NRS[a.rows]
        out = real.rollout(g, n)
        kf = S.fall_index(out)
        n_use = n if kf is None else min(n, ((kf - 10 - 1) // W) * W + 1)
        if n_use < 2 * W + 1:
            print(json.dumps(dict(case=case, bearing=b, skipped=f'fell at {kf}')), flush=True); continue
        cut = {k: v[:n_use] for k, v in out.items()}
        e0 = S.measured_rows(cut, g, W=W, row_fn=rf, nr=nr)
        for model in ['srb', 'srbc']:
            mk = (lambda o: S.make_G(real, o, g, W=W, cmd_fn=GS.cmd_fn_for(g), row_fn=rf, nr=nr)) if model == 'srb' else \
                 (lambda o: SC.make_G_contact(real, o, g, W=W, cmd_fn=GS.cmd_fn_for(g), row_fn=rf, nr=nr, kin=kin))
            rows, G = mk(cut)
            du, Gt = gn_step(rows, G, n_use, nr)
            dfull = np.zeros((n, 12), np.float32); dfull[:n_use] = du.reshape(n_use, 12)
            out2 = real.rollout(g, n, du=dfull)
            kf2 = S.fall_index(out2)
            m2 = n_use if kf2 is None else min(n_use, max(2 * W + 1, ((kf2 - 1) // W) * W + 1))
            e1 = S.measured_rows({k: v[:m2] for k, v in out2.items()}, g, W=W, row_fn=rf, nr=nr)
            e0m = e0[:len(e1)]
            pred, act = (Gt @ du)[:len(e1)], e1 - e0m
            cos = float(pred @ act / (np.linalg.norm(pred) * np.linalg.norm(act) + 1e-12))
            rows_m, G_m = mk(miscal(cut))
            du_m, _ = gn_step(rows_m, G_m, n_use, nr)
            cal_cos = float(du @ du_m / (np.linalg.norm(du) * np.linalg.norm(du_m) + 1e-12))
            cal_rel = float(np.linalg.norm(du_m - du) / (np.linalg.norm(du) + 1e-12))
            r = dict(case=case, bearing=b, model=model, n_use=n_use, fell_at=kf, fell_after=kf2, rows_match=float(np.abs(rows - e0).max()),
                     cos=round(cos, 3), J0=round(float(np.mean(e0m ** 2)), 3), J_pred=round(float(np.mean((e0m + pred) ** 2)), 3),
                     J1=round(float(np.mean(e1 ** 2)), 3), calib_cos=round(cal_cos, 3), calib_rel=round(cal_rel, 3))
            res.append(r); print(json.dumps(r), flush=True)
if a.out:
    json.dump(res, open(a.out, 'w'), indent=1)
