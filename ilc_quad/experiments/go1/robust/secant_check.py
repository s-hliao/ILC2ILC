"""secant_check.py RUN PLAN: for the 60 cm flat jump, landing-state change trial to trial:
actual, SRB-predicted, and predicted with the Broyden correction learned so far (the
stage3_secant option, offline)."""
import sys, numpy as np
from ilc_quad.sim_quad_model import QuadModel
from ilc_quad.ilc_gen import JumpILC, build_lifted_G
run, ref = sys.argv[1], sys.argv[2]
ilc = JumpILC(QuadModel("go1"), jump=(0.6, 0.0), box=None, phases=(30, 30, 30), margin=0.85,
              reference=ref, _check_reference=False)
N, Nc, nx, nu = ilc.N, ilc.Nc, ilc.nx, ilc.nu
T = [np.load(f"{run}/trial_{k:03d}.npz", allow_pickle=True) for k in range(1, 21)]
C = np.zeros((nx, Nc * nu)); e_srb = []; e_cor = []
for k in range(19):
    a, b = T[k], T[k + 1]
    du = (b["U_flown"] - a["U_flown"]).reshape(-1)
    if np.abs(du).max() < 1e-3: continue
    U_full = np.vstack([a["U_flown"], np.zeros((N - Nc, nu))])
    A, B = ilc.nominal.linearize_along_trial(a["log_X"], U_full, ilc.R1, ilc.R2, ilc.dt)
    GN = build_lifted_G(A, B, N, Nc, nx, nu, flatten=True)[-nx:]
    dx = b["log_X"][-1] - a["log_X"][-1]
    p0, p1 = GN @ du, (GN + C) @ du
    print(f"{k+1:2d}->{k+2:2d} th act {np.degrees(dx[2]):+6.2f} srb {np.degrees(p0[2]):+6.2f} corrected {np.degrees(p1[2]):+6.2f} | "
          f"x act {100*dx[0]:+5.2f} srb {100*p0[0]:+5.2f} corr {100*p1[0]:+5.2f} cm")
    C += np.outer(dx - p1, du) / (du @ du)
