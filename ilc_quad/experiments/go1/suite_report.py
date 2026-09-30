"""suite_report.py OUT (TASKS=file, default tasks.txt here): per task, the paper's metrics {ex, ey, eθ} of the REAL landing state
(recorded whole-body CoM and trunk pitch at the plan's end, t = N dt), falls, and the
landing on the last trial. Pass (the paper's Table I standard for its ILC at trial 20):
no falls, |ex| <= 1 cm, |ey| <= 2 cm, |eθ| <= 2 deg on the last trial."""
import json, math, os, re, subprocess, sys, numpy as np
from ilc_quad.sim_quad_model import QuadModel
from ilc_quad.ilc_gen import PlanarQuadModel
fb = PlanarQuadModel(QuadModel("go1"))
HERE = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]
ok_all = True

def real_errors(path):
    """(ex, ey [cm], eθ [deg]) of the recorded state at t = N dt, in the TO's frame."""
    d = np.load(path, allow_pickle=True)
    cfg = json.loads(str(d["config"])); N = sum(cfg["phases"]); dt = cfg["dt"]
    t, pos, quat, q = d["rec_t"], d["rec_pos"], d["rec_quat"], d["rec_q"]
    yaw0 = math.atan2(2 * (quat[0, 0] * quat[0, 3] + quat[0, 1] * quat[0, 2]),
                      1 - 2 * (quat[0, 2] ** 2 + quat[0, 3] ** 2))
    def planar(i):
        rel = pos[i] - pos[0]
        fwd = math.cos(yaw0) * rel[0] + math.sin(yaw0) * rel[1]
        w, x, y, z = quat[i]
        th = -math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        qp = [np.mean(q[i, [1, 4]]), np.mean(q[i, [2, 5]]), np.mean(q[i, [7, 10]]), np.mean(q[i, [8, 11]])]
        return np.concatenate([[fwd, rel[2], th], qp])
    s0 = planar(0); c0 = fb.com(s0).full().ravel()
    shift = d["x_ref"][0, :2] - c0
    i = min(int(np.searchsorted(t, N * dt - 1e-9)), len(t) - 1)
    s = planar(i); c = fb.com(s).full().ravel() + shift
    g = d["goal"]
    return 100 * (c[0] - g[0]), 100 * (c[1] - g[1]), math.degrees(s[2])

for line in open(os.environ.get("TASKS", os.path.join(HERE, "tasks.txt"))):
    name = line.split()[0]
    f = f"{out}/{name}.json"
    if not os.path.exists(f):
        print(f"{name:12s} MISSING"); ok_all = False; continue
    h = json.load(open(f))["history"]
    fell = [r["trial"] for r in h if r.get("fell")]
    n = len(h)
    ex, ey, et = real_errors(f"{out}/{name}/trial_{n:03d}.npz")
    ex1, ey1, et1 = real_errors(f"{out}/{name}/trial_001.npz")
    ok = not fell and abs(ex) <= 1.0 and abs(ey) <= 2.0 and abs(et) <= 2.0
    ok_all &= ok
    land = subprocess.run(["python3", os.path.join(HERE, "landing_check.py"), f"{out}/{name}", str(n)],
                          capture_output=True, text=True).stdout.strip().splitlines()
    land = land[-1] if land else ""
    m = re.search(r"min pitch after\s+([-+\d.]+).*max rear foot height\s+([-+\d.]+) cm.*creep last 0.6s\s+([-+\d.]+).*compress\s+([\d.]+)", land)
    ls = f"land: min pitch {m.group(1)}, rear up {m.group(2)} cm, creep {m.group(3)} cm/s, compress {m.group(4)} cm" if m else land[:80]
    print(f"{name:12s} {'OK  ' if ok else 'FAIL'} n={n:2d} falls={fell} | trial 1 {ex1:+.1f},{ey1:+.1f},{et1:+.1f} | "
          f"last {ex:+.1f} cm, {ey:+.1f} cm, {et:+.1f}° | {ls}")
print("ALL OK" if ok_all else "NOT ALL OK")
