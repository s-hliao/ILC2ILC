"""Post-landing behaviour of recorded trials: pitch after touchdown, rear feet lifting.
python3 landing_check.py RUN_DIR [trial ...]   (default: first, and last)"""
import glob, json, os, sys, numpy as np, mujoco
from ilc_quad.sim_quad_model import QuadModel
run = sys.argv[1]
files = sorted(glob.glob(f"{run}/trial_*.npz"))
nums = [int(t) for t in sys.argv[2:]] or [1, len(files)]
d0 = np.load(files[0], allow_pickle=True); cfg = json.loads(str(d0["config"]))
qm = QuadModel(cfg["robot"], box=cfg.get("box")); m, dd = qm.model, qm.data
box = cfg.get("box"); fg = qm.foot_geom_ids; r = m.geom_size[fg[0]][0]
ter = lambda x: box["height"] if box and x >= box["x_front"] else 0.0
for n in nums:
    d = np.load(f"{run}/trial_{n:03d}.npz", allow_pickle=True); res = json.loads(str(d["result"]))
    t = d["rec_t"] - d["rec_t"][0]; qu = d["rec_quat"]; con = d["rec_contacts"]
    pitch = np.degrees(-2 * np.arctan2(qu[:, 2], qu[:, 0]))          # nose up +
    fl = np.where((t > 0.5) & (con.max(1) < 1))[0]                   # airborne samples
    if not fl.size: print(n, "no flight?"); continue
    td = fl[0] + np.argmax(con[fl[0]:].max(1) >= 1)                   # first touchdown after take-off
    later = np.where((t > t[td] + 0.02) & (con.max(1) < 1))[0]
    airborne_ms = len(later) * np.median(np.diff(t)) * 1000           # all four feet off again
    h = np.zeros((len(t), 4))
    for i in range(td, len(t)):
        dd.qpos[0:3], dd.qpos[3:7], dd.qpos[qm.qpos_adr] = d["rec_pos"][i], qu[i], d["rec_q"][i]
        mujoco.mj_kinematics(m, dd)
        for f in range(4):
            x = dd.geom_xpos[fg[f]]; h[i, f] = x[2] - r - ter(x[0])
    rear_td = td + np.argmax(con[td:, 2:].min(1) > 5)                 # both rear feet down
    after = slice(rear_td, len(t))
    rear_off = (con[after, 2:].min(1) < 5)
    dt = np.median(np.diff(t))
    omega = np.gradient(np.unwrap(np.radians(pitch)), t)
    zb = d["rec_pos"][:, 2]; squat = 100 * (zb[-1] - zb[td:].min())
    xp = d["rec_pos"][:, 0]; creep = 100 * (xp[-1] - xp[np.searchsorted(t, t[-1] - 0.6)]) / 0.6
    print(f"{os.path.basename(run)} trial {n:2d}{' FELL' if res.get('fell') else '     '}: touchdown t={t[td]:.2f} pitch {pitch[td]:+5.1f} deg rate {np.degrees(omega[td]):+6.0f} deg/s"
          f" | min pitch after {pitch[td:].min():+5.1f} (dive {pitch[td:].min() - pitch[td]:+5.1f}) end {pitch[-1]:+5.1f}"
          f" | rear down {t[rear_td] - t[td]:.2f}s after front, then off ground {rear_off.sum() * dt * 1000:4.0f} ms, max rear foot height {100 * h[after, 2:].max():4.1f} cm | all feet off again {airborne_ms:4.0f} ms | creep last 0.6s {creep:+4.1f} cm/s | compress {squat:3.1f} cm")
