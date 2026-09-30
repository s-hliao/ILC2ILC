"""ratio_check.py RUN [RUN...]: per run, least-squares gain actual/predicted of the landing
change (theta, x, z) over all trial-to-trial steps: how well the SRB model's sensitivities
match the robot (1 = exact; ~0 = no relation)."""
import sys, json, numpy as np
from ilc_quad.sim_quad_model import QuadModel
from ilc_quad.ilc_gen import JumpILC, build_lifted_G
from ilc_quad.trial_log import TrialRecorder
qm = QuadModel("go1")
for run in sys.argv[1:]:
    T = [np.load(f, allow_pickle=True) for f in sorted(__import__("glob").glob(f"{run}/trial_*.npz"))]
    cfg = json.loads(str(T[0]["config"]))
    ilc = JumpILC(qm, jump=cfg["jump"], box=cfg["box"], phases=cfg["phases"], margin=cfg["margin"],
                  reference=TrialRecorder.load(run).reference, _check_reference=False)
    N, Nc, nx, nu = ilc.N, ilc.Nc, ilc.nx, ilc.nu
    out = []
    for k in range(len(T) - 1):
        a, b = T[k], T[k + 1]
        du = (b["U_flown"] - a["U_flown"]).reshape(-1)
        if np.abs(du).max() < 0.05: continue
        U_full = np.vstack([a["U_flown"], np.zeros((N - Nc, nu))])
        A, B = ilc.nominal.linearize_along_trial(a["log_X"], U_full, ilc.R1, ilc.R2, ilc.dt)
        GN = build_lifted_G(A, B, N, Nc, nx, nu, flatten=True)[-nx:]
        p = GN @ du; d = b["log_X"][-1] - a["log_X"][-1]
        out.append((k + 1, np.degrees(p[2]), np.degrees(d[2]), 100 * p[0], 100 * d[0], 100 * p[1], 100 * d[1]))
    o = np.array(out)
    r = lambda i, j: float(o[:, i] @ o[:, j] / (o[:, i] @ o[:, i]))
    print(f"{run.split('/')[-1]:28s} steps {len(o):2d}  gain act/pred: theta {r(1,2):+.2f}  x {r(3,4):+.2f}  z {r(5,6):+.2f}  | "
          + " ".join(f"{a:+.1f}/{b:+.1f}" for _, a, b, *_ in out[:10]))
