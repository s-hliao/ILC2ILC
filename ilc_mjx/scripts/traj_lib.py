"""traj_lib.py: load the exported trajectories (export_trajectories.py) and pose the Go1 model on them.

    T = load("trajectories/gate_loose.npz")        # dict: t, pos, quat, q, contacts, meta (list of dicts)
    sel = select(T, robot="real_r1", iteration=2)   # indices
    m, d = go1_model(box=(x_front, height))         # MuJoCo model (menagerie go1.xml + the jump's box)
    pts = side_points(m, d, T, i, k)                # tick k of jump i: trunk / hips / knees / feet, (x, z)

The Go1 model comes from a mujoco_menagerie checkout: $MUJOCO_MENAGERIE_PATH, else ~/mujoco_menagerie.
"""
import json, os
import numpy as np

LEGS = ["FR", "FL", "RR", "RL"]                  # go1.xml order (the exported q's order)


def load(path):
    z = np.load(path)
    T = {k: z[k] for k in ("t", "pos", "quat", "q", "contacts")}
    T["meta"] = json.loads(str(z["meta"]))
    T["name"] = os.path.splitext(os.path.basename(path))[0]
    return T


def select(T, **kw):
    """indices of the jumps whose meta match every keyword (goal: (x, h) to 3 decimals)."""
    out = []
    for i, m in enumerate(T["meta"]):
        okk = True
        for k, v in kw.items():
            if v is None:
                continue
            if k == "goal":                               # (stored as float32: match within 0.5 mm)
                okk &= bool(np.all(np.abs(np.asarray(m["goal"], float) - np.asarray(v, float)) < 5e-4))
            else:
                okk &= m.get(k) == v
        if okk:
            out.append(i)
    return out


def menagerie():
    for p in (os.environ.get("MUJOCO_MENAGERIE_PATH"), os.path.expanduser("~/mujoco_menagerie"), "/mujoco_menagerie"):
        if p and os.path.exists(os.path.join(p, "unitree_go1", "scene.xml")):
            return p
    raise SystemExit("no mujoco_menagerie checkout found: set MUJOCO_MENAGERIE_PATH")


def go1_model(box=None, scene=True):
    """the menagerie Go1 (scene.xml: floor and light), plus the box (x_front, height): 1 m long and wide, as the CPU
    robot's scene placed it (ilc_quad sim_quad_model)."""
    import mujoco
    spec = mujoco.MjSpec.from_file(os.path.join(menagerie(), "unitree_go1", "scene.xml" if scene else "go1.xml"))
    if box is not None and box[0] is not None and box[1] >= 0.005:
        x_front, h = box
        g = spec.worldbody.add_geom()
        g.name = "box"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.5, 0.5, h / 2]
        g.pos = [x_front + 0.5, 0.0, h / 2]
        g.rgba = [0.85, 0.55, 0.3, 1.0]
    m = spec.compile()
    return m, mujoco.MjData(m)


def set_pose(m, d, T, i, k):
    import mujoco
    k = min(k, T["meta"][i]["n_ticks"] - 1)
    d.qpos[:3] = T["pos"][i, k]
    d.qpos[3:7] = T["quat"][i, k]
    d.qpos[7:19] = T["q"][i, k]
    mujoco.mj_kinematics(m, d)


def side_points(m, d, T, i, k):
    """(x, z) of the trunk's front and rear hips, and per leg hip -> knee -> foot, at tick k of jump i."""
    import mujoco
    set_pose(m, d, T, i, k)
    body = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)][[0, 2]]
    site = lambda n: d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)][[0, 2]]
    legs = {l: np.stack([body(f"{l}_thigh"), body(f"{l}_calf"), site(l)]) for l in LEGS}
    trunk = body("trunk")
    return dict(trunk=trunk, front=(legs["FR"][0] + legs["FL"][0]) / 2, rear=(legs["RR"][0] + legs["RL"][0]) / 2, legs=legs)


def box_of(meta):
    return (meta.get("box_x_front"), meta.get("box_height", 0.0))
