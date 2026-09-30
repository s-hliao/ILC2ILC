"""sens_check.py RUN PLAN DX DZ [XF H]: predicted (SRB lifted G) vs actual change of the
landing state between consecutive trials of a run (flown on PLAN, phases 30/30/30, margin 0.85)."""
import sys, numpy as np
from ilc_quad.sim_quad_model import QuadModel
from ilc_quad.ilc_gen import JumpILC, build_lifted_G

run, ref, jump = sys.argv[1], sys.argv[2], (float(sys.argv[3]), float(sys.argv[4]))
box = None if len(sys.argv) < 7 else dict(x_front=float(sys.argv[5]), height=float(sys.argv[6]))
ilc = JumpILC(QuadModel("go1"), jump=jump, box=box, phases=(30, 30, 30), margin=0.85,
              reference=ref, _check_reference=False)
N, Nc, nx, nu = ilc.N, ilc.Nc, ilc.nx, ilc.nu
for k in range(1, 20):
    try:
        a = np.load(f"{run}/trial_{k:03d}.npz", allow_pickle=True)
        b = np.load(f"{run}/trial_{k + 1:03d}.npz", allow_pickle=True)
    except FileNotFoundError:
        break
    Ua, Ub, X = a["U_flown"], b["U_flown"], a["log_X"]
    du = Ub - Ua
    if np.abs(du).max() < 1e-9:
        continue
    U_full = np.vstack([Ua, np.zeros((N - Nc, nu))])
    A, B = ilc.nominal.linearize_along_trial(X, U_full, ilc.R1, ilc.R2, ilc.dt)
    G = build_lifted_G(A, B, N, Nc, nx, nu, flatten=True)
    pred = (G @ du.reshape(-1)).reshape(N, nx)[-1]
    act = b["log_X"][-1] - X[-1]
    print(f"{k:2d}->{k + 1:2d} stage {int(b['stage'])} |du| {np.abs(du).max():6.1f} N  "
          f"dx pred {100 * pred[0]:+6.2f} act {100 * act[0]:+6.2f} cm | "
          f"dz pred {100 * pred[1]:+6.2f} act {100 * act[1]:+6.2f} cm | "
          f"dth pred {np.degrees(pred[2]):+6.2f} act {np.degrees(act[2]):+6.2f} deg")
