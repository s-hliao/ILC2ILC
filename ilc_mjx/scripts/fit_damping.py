#!/usr/bin/env python3
"""fit_damping.py BASE.npz PERTURBED.npz [...] --ref REFERENCE.npz (2026-11-01): the Go1's joint damping from flights: fly the plan's forces once, then with known force changes, and
take the damping (N m s / rad, thigh and calf) whose full-body linearization along the base trial best predicts the measured change of (vx, vz) through stance. No model of the
response is needed beyond FullBodyLin; on the robot it takes one base jump and two or three perturbed ones. The controller's stance gains are --kp / --kd (default 0, 1).
Sim check (known truth): why/g A1_tt_k0d1 (2.0), why/h Q_0.5_k0d1 (0.5), Q_0.1_k0d1 (0.1)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("ILC_OVERLAY_HOST", "/home/henry/ilc_ws/log/dilc/plane/fixbox/overlay"))
os.environ["ILC_PAPER"] = "1"
from ilc_quad.sim_quad_model import QuadModel
import ilc_quad.ilc_gen as gen

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("perturbed", nargs="+")
ap.add_argument("--ref", required=True); ap.add_argument("--kp", type=float, default=0.0); ap.add_argument("--kd", type=float, default=1.0)
ap.add_argument("--phases", type=int, nargs=3, default=[30, 30, 30])
ap.add_argument("--grid", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
ap.add_argument("--menagerie", default=os.environ.get("MENAGERIE", "/home/henry/mujoco_menagerie"))
a = ap.parse_args()
Nd, Ns, Nf = a.phases; Nc, N = Nd + Ns, Nd + Ns + Nf
qm = QuadModel("go1", a.menagerie); fb = gen.PlanarQuadModel(qm)
ref = np.load(a.ref, allow_pickle=True); b = np.load(a.base, allow_pickle=True)
Xb, Ub = np.asarray(b["log_X"], float), np.asarray(b["U_flown"], float)
A_flight = np.eye(6); A_flight[0, 3] = A_flight[1, 4] = A_flight[2, 5] = 0.01
Z = [np.load(p, allow_pickle=True) for p in a.perturbed]
ks = np.arange(Nd + 2, Nc - 2, 2)
out = []
for d in a.grid:
    fb.damping = 2.0 * np.array([d] * 4)
    L = gen.FullBodyLin(fb, Nd, Nc, N, 0.01, a.kp, a.kd)
    G = L.lifted(Xb, Ub, np.asarray(ref["u_ref"], float), np.asarray(b["log_q"], float), np.asarray(b["log_qdot"], float), np.asarray(ref["info_tau"], float), np.asarray(ref["info_q_ref"], float), np.asarray(ref["info_qd_ref"], float), A_flight)
    sse, tot = 0.0, 0.0
    for z in Z:
        du = (np.asarray(z["U_flown"], float) - Ub).ravel(); dX = np.asarray(z["log_X"], float) - Xb
        p = (G @ du).reshape(N, 6)
        for k in ks:
            e = dX[k, 3:5] - p[k - 1, 3:5]; sse += float(e @ e); tot += float(dX[k, 3:5] @ dX[k, 3:5])
    out.append((d, np.sqrt(sse / max(tot, 1e-12))))
    print(f"damping {d:5.2f}: relative prediction error {out[-1][1]:.3f}", flush=True)
b_ = min(out, key=lambda r: r[1]); print(f"best damping {b_[0]:.2f} N m s / rad (relative error {b_[1]:.3f})")
