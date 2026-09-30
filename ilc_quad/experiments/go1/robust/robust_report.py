"""robust_report.py OUT [OUT...]: every run (<task>__<cond>.json) scored on the TRUE state
(rec_true_*, what the robot did, not what the controller measured):
  last   {ex, ey [cm], eθ [deg]} of the whole-body CoM / pitch at t = N dt, last trial
  falls  trials whose true trunk tilt passed 0.6 rad after t = N dt (or the node said so)
  land   worst over trials 1..n after touchdown: nose = lowest pitch [deg] (nose up +),
         rear = highest rear foot above the surface [cm]; creep = cm/s drift in the last
         0.6 s of the last trial
pass = no falls, |ex|<=1, |ey|<=2, |eθ|<=2 (paper Table I, trial 20)."""
import glob, json, math, os, sys, numpy as np, mujoco
from ilc_quad.sim_quad_model import QuadModel
from ilc_quad.ilc_gen import PlanarQuadModel
qm = QuadModel("go1"); fb = PlanarQuadModel(qm)
fg = qm.foot_geom_ids; r_foot = qm.model.geom_size[fg[0]][0]


def tilt(q):
    w, x, y, z = q.T
    return np.arccos(np.clip(1 - 2 * (x ** 2 + y ** 2), -1, 1))


def trial(path):
    d = np.load(path, allow_pickle=True)
    g = lambda k: d[f"rec_true_{k}"] if f"rec_true_{k}" in d.files else d[f"rec_{k}"]
    cfg = json.loads(str(d["config"])); N = sum(cfg["phases"]); dt = cfg["dt"]
    t, pos, quat, q, con = d["rec_t"], g("pos"), g("quat"), g("q"), g("contacts")
    yaw0 = math.atan2(2 * (quat[0, 0] * quat[0, 3] + quat[0, 1] * quat[0, 2]),
                      1 - 2 * (quat[0, 2] ** 2 + quat[0, 3] ** 2))

    def planar(i):
        rel = pos[i] - pos[0]
        fwd = math.cos(yaw0) * rel[0] + math.sin(yaw0) * rel[1]
        w, x, y, z = quat[i]
        th = -math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        qp = [np.mean(q[i, [1, 4]]), np.mean(q[i, [2, 5]]), np.mean(q[i, [7, 10]]),
              np.mean(q[i, [8, 11]])]
        return np.concatenate([[fwd, rel[2], th], qp])

    s0 = planar(0); shift = d["x_ref"][0, :2] - fb.com(s0).full().ravel()
    i = min(int(np.searchsorted(t, N * dt - 1e-9)), len(t) - 1)
    s = planar(i); c = fb.com(s).full().ravel() + shift; goal = d["goal"]
    res = json.loads(str(d["result"]))
    # after the jump window only, like the node: box jumps pitch past 0.6 rad at push-off
    fell = bool(res.get("fell")) or tilt(quat[t >= N * dt]).max() > 0.6
    pitch = np.degrees(np.array([planar(k)[2] for k in range(len(t))]))
    Nc = sum(cfg["phases"][:2])
    fl = np.where((t > Nc * dt + 0.03) & (con.max(1) < 1))[0]      # airborne after takeoff
    nose = rear = creep = np.nan
    if fl.size:
        td = fl[0] + np.argmax(con[fl[0]:].max(1) >= 1)
        nose = pitch[td:].min()
        box = cfg.get("box"); ter = lambda x: box["height"] if box and x >= box["x_front"] else 0.0
        m, dd = qm.model, qm.data; hmax = -1.0
        for k in range(td + np.argmax(con[td:, 2:].min(1) > 5), len(t), 5):
            dd.qpos[0:3], dd.qpos[3:7], dd.qpos[qm.qpos_adr] = pos[k], quat[k], q[k]
            mujoco.mj_kinematics(m, dd)
            for f in (2, 3):
                x = dd.geom_xpos[fg[f]]; hmax = max(hmax, x[2] - r_foot - ter(x[0]))
        rear = 100 * hmax
        k6 = np.searchsorted(t, t[-1] - 0.6)
        fwd = math.cos(yaw0) * (pos[:, 0] - pos[0, 0]) + math.sin(yaw0) * (pos[:, 1] - pos[0, 1])
        creep = 100 * (fwd[-1] - fwd[k6]) / 0.6
    return dict(e=(100 * (c[0] - goal[0]), 100 * (c[1] - goal[1]), math.degrees(s[2])),
                fell=fell, nose=nose, rear=rear, creep=creep)


allok = True
for out in sys.argv[1:]:
    for f in sorted(glob.glob(f"{out}/*.json")):
        name = os.path.basename(f)[:-5]; run = f[:-5]
        files = sorted(glob.glob(f"{run}/trial_*.npz"))
        if not files:
            print(f"{name:34s} NO TRIALS"); allok = False; continue
        tr = [trial(p) for p in files]
        falls = [k + 1 for k, r in enumerate(tr) if r["fell"]]
        ex, ey, et = tr[-1]["e"]; e1 = tr[0]["e"]
        ok = not falls and abs(ex) <= 1 and abs(ey) <= 2 and abs(et) <= 2
        allok &= ok
        nose = np.nanmin([r["nose"] for r in tr]); rear = np.nanmax([r["rear"] for r in tr])
        print(f"{name:34s} {'OK  ' if ok else 'FAIL'} n={len(tr):2d} falls={falls} | "
              f"t1 {e1[0]:+5.1f},{e1[1]:+5.1f},{e1[2]:+5.1f} | last {ex:+5.1f} cm {ey:+5.1f} cm "
              f"{et:+5.1f}° | worst nose {nose:+5.1f}° rear up {rear:4.1f} cm | "
              f"creep {tr[-1]['creep']:+4.1f} cm/s")
print("ALL OK" if allok else "NOT ALL OK")
