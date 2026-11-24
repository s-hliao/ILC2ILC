"""(2026-11-24) Positivity check for the goal walker stage, per row type: one damped GN step from the SRB G on a perturbed walk,
applied open-loop (du added to the actions) on the same perturbed sim robot; cos(predicted row change, actual), and J before / after.
    python scripts/goalstage_check.py PARAMS --variant vel2 --case payload3R --bearings 0 90 180 -90 --dist 1.5
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilc_go1lla import goalstage as GS, srb as S   # noqa: E402
from zeroshot_cases import parse_case               # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('params'); ap.add_argument('--variant', default='vel2'); ap.add_argument('--case', default='payload3R')
ap.add_argument('--bearings', type=float, nargs='+', default=[0, 90, 180, -90]); ap.add_argument('--dist', type=float, default=1.5)
ap.add_argument('--kinds', nargs='+', default=['time', 'progress', 'goal'])
ap.add_argument('--beta', type=float, default=0.5); ap.add_argument('--delta', type=float, default=0.05)
ap.add_argument('--cap', type=float, default=0.03); ap.add_argument('--max-lag', type=int, default=25); ap.add_argument('--W', type=int, default=10)
a = ap.parse_args()
real = GS.GoalSimWalker(a.params, a.variant, **parse_case(a.case))
nom = GS.GoalSimWalker(a.params, a.variant)
KIN = S.LegKin(real.env.mj_model)
for b in a.bearings:
    g = np.array([a.dist * np.cos(np.radians(b)), a.dist * np.sin(np.radians(b))], np.float32)
    n = GS.walk_steps(g, a.W)
    ref = GS.reference(nom.rollout(g, n))
    out = real.rollout(g, n)
    kf = S.fall_index_feet(out, KIN)
    for kind in a.kinds:
        rf = GS.row_fn_for(kind, g, ref); nr = GS.NRS[kind]
        e = S.measured_rows(out, g, W=a.W, row_fn=rf, nr=nr)
        rows, G = S.make_G(real, out, g, W=a.W, cmd_fn=GS.cmd_fn_for(g), row_fn=rf, nr=nr)
        G = np.array(G)
        nw = G.shape[0] // nr
        Gr = G.reshape(nw, nr, n, 12)
        for j in range(nw):
            Gr[j, :, :max(0, (j + 1) * a.W - a.max_lag)] = 0.0
        G = Gr.reshape(G.shape)
        N = G.T @ G; lam = a.delta * np.trace(N) / N.shape[0]
        du = -a.beta * np.linalg.solve(N + lam * np.eye(N.shape[0]), G.T @ rows)
        r = np.sqrt(np.mean(du ** 2)); du = du * min(1.0, a.cap / max(r, 1e-12))
        out2 = real.rollout(g, n, du=du.reshape(n, 12))
        e2 = S.measured_rows(out2, g, W=a.W, row_fn=rf, nr=nr)
        pred, act = G @ du, e2 - e
        cos = float(pred @ act / (np.linalg.norm(pred) * np.linalg.norm(act) + 1e-12))
        print(json.dumps(dict(bearing=b, kind=kind, fell=kf is not None, rows_match=float(np.abs(rows - e).max()), cos=round(cos, 3),
                              J0=round(float(np.mean(e ** 2)), 3), J1=round(float(np.mean(e2 ** 2)), 3),
                              J_pred=round(float(np.mean((e + pred) ** 2)), 3), du_rms=round(float(np.sqrt(np.mean(du ** 2))), 4),
                              final_cm=round(100 * float(np.linalg.norm(out['p'][-1, :2] - g)), 1),
                              final_cm_after=round(100 * float(np.linalg.norm(out2['p'][-1, :2] - g)), 1))), flush=True)
