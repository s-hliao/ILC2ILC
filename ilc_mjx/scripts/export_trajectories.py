#!/usr/bin/env python3
"""
export_trajectories.py RUN [RUN ...] [--out DIR]: the hardware stage's recorded jumps, compact and self-contained,
for committing and replaying (animate_hw_stage.py, replay_mujoco.py).

RUN is any directory holding recorded jumps (the CPU "real" robot's episode npz files, found recursively): a deploy.py
hardware stage (RUN/<robot>/it<k>/real/*.npz), fada_adapt.py's adaptation, rma_ctx_eval.py's calibration, or
record_evals.sh's recorded evaluations (RUN/<robot>/*.npz). The robot and the start condition come from the file name. Per jump the ground truth at the robot's
2 ms ticks -- base position, base quaternion (w x y z), the 12 joint angles in MuJoCo's go1.xml order (FR FL RR RL x
hip thigh calf; the recorder's canonical FL FR RL RR order is remapped), the 4 feet's contact flags -- and the goal, the
box (front face x and height in the robot's world frame, as the robot's scene placed it), the robot, the iteration,
the start condition, and the outcome (landing x / z / pitch error, fell, the trial's cost J).
-> DIR/<run name>.npz (float32, compressed) + DIR/<run name>.json (the index, human-readable).
Run from src/ilc_mjx with the ilcmjx env.
"""
import argparse, glob, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--name", default="", help="output name (one RUN only; default: the run directory's name)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trajectories"))
a = ap.parse_args()
sys.path.insert(0, "/home/henry/ilc_ws/src/ilc_quad/scripts")
from dilc_execute import GoalBank  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ilc_mjx import host_path  # noqa: E402

CANON = ["FL", "FR", "RL", "RR"]                 # the recorder's leg order (ilc_quad sim_quad_model.CANONICAL_LEGS)
MODEL = ["FR", "FL", "RR", "RL"]                 # go1.xml's qpos order
PERM = np.concatenate([[3 * CANON.index(l) + j for j in range(3)] for l in MODEL])

os.makedirs(a.out, exist_ok=True)
for run in a.runs:
    run = os.path.abspath(run)
    name = a.name or os.path.basename(run.rstrip("/"))
    eps = sorted(f for f in glob.glob(os.path.join(run, "**", "*.npz"), recursive=True)
                 if os.path.basename(f).startswith("policy_g") and "rec_true_pos" in np.load(f).files)
    if not eps:
        print(f"{run}: no recorded jumps")
        continue
    bank = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "log", "dilc", "plane",
                                       "v10_s2", "snap20", "bank.json")))
    gb = GoalBank(dict(bank, plans=[dict(p, path=host_path(p["path"])) for p in bank["plans"]]))
    sfiles = glob.glob(os.path.join(run, "*", "summary.json"))
    summ = json.load(open(sfiles[0])).get("args", {}) if sfiles else {}
    T = max(len(np.load(f)["rec_t"]) for f in eps)
    pad = lambda x: np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)]) if len(x) < T else x[:T]
    pos, quat, q, con, meta = [], [], [], [], []
    for f in eps:
        z = np.load(f)
        # the file name: policy_g<x>_<h>_<cond>_s<seed>.npz, cond = robot[+perturbation]
        cond = os.path.basename(f)[len("policy_g"):].rsplit("_s", 1)[0].split("_", 2)[2]
        robot = cond.split("+")[0]
        parts = [p for p in f.split(os.sep) if p.startswith("it") and p[2:].isdigit()]
        it = int(parts[-1][2:]) if parts else -1
        g = [float(v) for v in z["goal"]]
        _, box, _ = gb.reference(np.array(g))
        n = len(z["rec_t"])
        pos.append(pad(np.asarray(z["rec_true_pos"], np.float32)))
        quat.append(pad(np.asarray(z["rec_true_quat"], np.float32)))
        q.append(pad(np.asarray(z["rec_true_q"], np.float32)[:, PERM]))
        con.append(pad(np.asarray(z["rec_true_contacts"], np.float32)[:, [CANON.index(l) for l in MODEL]] > 0.5))
        info = np.asarray(z["info"], float)
        meta.append(dict(robot=robot, cond=cond, iteration=it, n_ticks=int(n), goal=g, box_x_front=box["x_front"] if box else None,
                         box_height=box["height"] if box else 0.0, ex=float(info[0]), ez=float(info[1]), eth=float(info[2]),
                         fell=bool(z["fell"]), J=float(0.01 * np.asarray(z["rc"])[:, 2].sum()),
                         train_cond=summ.get("train_cond") or "nominal", file=os.path.relpath(f, run)))
    t = (np.arange(T) * float(np.median(np.diff(np.load(eps[0])["rec_t"])))).astype(np.float32)
    out = os.path.join(a.out, f"{name}.npz")
    np.savez_compressed(out, t=t, pos=np.stack(pos), quat=np.stack(quat), q=np.stack(q), contacts=np.stack(con),
                        meta=np.array(json.dumps(meta)))
    json.dump(dict(run=name, source=run, n=len(meta), ticks=int(T), dt=float(np.median(np.diff(t))),
                   joint_order="go1.xml qpos: FR FL RR RL x (hip, thigh, calf)", quat="w x y z",
                   recipe={k: summ.get(k) for k in ("member", "goals", "iters", "gate", "train_cond", "robots") if k in summ},
                   jumps=meta), open(os.path.join(a.out, f"{name}.json"), "w"), indent=1)
    print(f"{name}: {len(meta)} jumps x {T} ticks -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
