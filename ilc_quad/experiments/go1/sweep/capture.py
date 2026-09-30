"""capture.py OUT KEY: per trial, the capture-point margin at the moment all four feet are down
(front foot x minus [CoM x + vx sqrt(h/g)], m, true state; < 0: the CoM's momentum carries it
over the front feet, so the robot must tip onto them -- lift the rear -- to stop), against the
rear-foot unloaded time after it (stick.py). Prints the correlation and a binned table."""
import glob, json, math, os, sys
import numpy as np, mujoco
from ilc_quad.sim_quad_model import QuadModel
from ilc_quad.ilc_gen import PlanarQuadModel
HERE = os.path.dirname(os.path.abspath(__file__))
ns = {}; exec(open(f"{HERE}/stick.py").read().split("\nfor out in sys.argv")[0], ns)
qm = QuadModel("go1"); m, dd = qm.model, qm.data; fg = qm.foot_geom_ids; g = 9.81


def one(p):
    d = np.load(p, allow_pickle=True)
    k = lambda n: d[f"rec_true_{n}"] if f"rec_true_{n}" in d.files else d[f"rec_{n}"]
    t, pos, quat, q, con = d["rec_t"], k("pos"), k("quat"), k("q"), k("contacts")
    cfg = json.loads(str(d["config"])); Nc = sum(cfg["phases"][:2]); dt = cfg["dt"]
    res = json.loads(str(d["result"]))
    air = np.where((t > Nc * dt + 0.03) & (con.max(1) < 1))[0]
    if res.get("fell") or not air.size:
        return None
    after = np.arange(air[0], len(t)); alld = after[con[after].min(1) >= 1]
    if not alld.size:
        return None
    i = alld[0]
    yaw = math.atan2(2 * (quat[0, 0] * quat[0, 3] + quat[0, 1] * quat[0, 2]), 1 - 2 * (quat[0, 2] ** 2 + quat[0, 3] ** 2))
    fwd = lambda v: math.cos(yaw) * v[0] + math.sin(yaw) * v[1]
    dd.qpos[0:3], dd.qpos[3:7], dd.qpos[qm.qpos_adr] = pos[i], quat[i], q[i]
    mujoco.mj_kinematics(m, dd); mujoco.mj_comPos(m, dd)
    com = dd.subtree_com[0]
    front = 0.5 * (fwd(dd.geom_xpos[fg[0]]) + fwd(dd.geom_xpos[fg[1]]))
    rear_z = 0.5 * (dd.geom_xpos[fg[2]][2] + dd.geom_xpos[fg[3]][2])
    j = slice(max(i - 5, 0), i + 1)
    vx = np.polyfit(t[j], [fwd(pp) for pp in pos[j]], 1)[0]
    h = com[2] - rear_z
    return dict(margin=front - (fwd(com) + vx * math.sqrt(h / g)), lever=front - fwd(com), vx=vx, h=h,
                rear=ns["trial"](p)["rear"])


out, key = sys.argv[1:3]
R = [r for p in sorted(glob.glob(f"{out}/*__{key}__*/trial_*.npz")) if "mu05" not in p
     for r in [one(p)] if r is not None]
mg = np.array([r["margin"] for r in R]); rr = np.array([r["rear"] for r in R])
print(f"{len(R)} clean trials; corr(margin, rear unloaded) = {np.corrcoef(mg, rr)[0, 1]:+.2f}; "
      f"lever {np.median([r['lever'] for r in R]):.3f} m, vx {np.median([r['vx'] for r in R]):.2f} m/s, h {np.median([r['h'] for r in R]):.3f} m (medians)")
for lo, hi in ((-1, -0.08), (-0.08, -0.04), (-0.04, 0), (0, 0.04), (0.04, 1)):
    s = (mg >= lo) & (mg < hi)
    if s.any():
        print(f"  margin [{lo:+.2f},{hi:+.2f}) m  n={s.sum():4d}  rear unloaded ms median {np.median(rr[s]):4.0f}  never {np.mean(rr[s] < 10):4.0%}")
