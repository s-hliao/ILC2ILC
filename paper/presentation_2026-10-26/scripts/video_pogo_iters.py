#!/usr/bin/env python3
"""
video_pogo_iters.py (2026-10-26): the pogo's hardware stage (servo leg, sim-to-sim), iteration by iteration, on ONE
stick and ONE training goal, with every earlier iteration kept on screen as a ghost in its ramp colour.
The stage's networks are flown again exactly as hw_stage.py flew them (log/pogo_servo/runs/v3/<arm>_curve: it1..it3,
the final networks; the same fixed start perturbation; T = 300 periods = 3 s).
  top     the stick hopping (camera follows the current iteration; earlier iterations' bodies at the same instant
          as faded ghosts, their paths in their colours)
  bottom  apex height per hop | forward speed per hop (every iteration's series, the current one growing; target
          dashed, success band shaded) | the batch over iterations (sticks on target of the 6 training goals,
          and this goal's apex RMS)
    python video_pogo_iters.py ROBOT [--arm nom_s0] [--goal I] [--procs 16]
"""
import json
import os
import sys

import numpy as np

POGO = "/home/henry/ilc_ws/src/ilc_pogo"
SERVO = "/home/henry/ilc_ws/log/pogo_servo"
sys.path.insert(0, POGO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import common as C                                            # noqa: E402
import policies                                               # noqa: E402
import real_pogo as rp                                        # noqa: E402
import iter_style as S                                        # noqa: E402

RUNS = f"{SERVO}/runs/v3"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pogo")
TRAJ, META = None, None
STEP = 3                                                      # control periods per frame (DT 0.01 -> 1x real time at 33 fps)


def fly_stage(arm, robot):
    A = json.load(open(f"{RUNS}/{arm}_curve/args.json"))
    base = os.path.join(SERVO, A["policy"])
    G = C.TRAIN_GOALS
    pp = rp.robot_params(robot)
    R = A["robots"] or rp.DEFAULT_ROBOTS
    ri = R.index(robot)
    x0 = rp.start_states(np.tile(G, (len(R), 1)), A["train_pert"], np.random.default_rng(A["seed"]))[ri * len(G):(ri + 1) * len(G)]
    out = []
    for k in range(A["iters"] + 1):
        f = base if k == 0 else (f"{RUNS}/{arm}_curve/it{k}/{robot}_policy.npz" if k < A["iters"] else f"{RUNS}/{arm}_curve/{robot}_policy.npz")
        res = rp.simulate(policies.as_policy(f), G, x0, {kk: np.full(len(G), v) for kk, v in pp.items()}, A["T"])
        m = [C.hop_metrics(res["x"][i], G[i], res["failed_end"][i]) for i in range(len(G))]
        out.append(dict(x=res["x"], m=m, fell=res["failed_end"], l0=pp["l0"], phi_off=pp["phi_off"]))
        print(f"it{k}: success {sum(bool(q['success']) for q in m)}/6", flush=True)
    return out


def hops(xs):
    """per hop (apex to apex): apex height (with the ballistic rise left), the index of the apex, mean forward speed"""
    vz, st = xs[:, 3], xs[:, 5]
    idx = np.where((vz[:-1] > 0) & (vz[1:] <= 0) & (st[1:] < 0.5))[0] + 1
    ap = xs[idx, 1] + 0.5 * xs[idx, 3] ** 2 / C.GRAV
    sp = np.r_[np.nan, np.diff(xs[idx, 0]) / (np.diff(idx) * C.DT)] if len(idx) else np.zeros(0)
    return idx, ap, sp


def spring_xy(p0, p1, n=9, w=0.025):
    d = np.asarray(p1) - np.asarray(p0)
    L = np.linalg.norm(d) + 1e-9
    v = np.array([-d[1], d[0]]) / L
    s = np.linspace(0, 1, 2 * n + 1)
    off = np.zeros_like(s)
    off[1:-1:2], off[2:-1:2] = w, -w
    core = (s > 0.15) & (s < 0.85)
    pts = np.asarray(p0)[None] + s[:, None] * d[None] + (off * core)[:, None] * v[None]
    return pts[:, 0], pts[:, 1]


def frame(job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.rc()
    i, k, t, phase, path = job
    robot, gi, K = META["robot"], META["goal"], len(TRAJ) - 1
    g = C.TRAIN_GOALS[gi]
    T = TRAJ[k]["x"].shape[1] - 1
    j = T if t is None else min(t * STEP, T)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], left=0.05, right=0.985, top=0.79, bottom=0.07, hspace=0.32, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    xs = TRAJ[k]["x"][gi]
    X, Z = xs[j, 0], xs[j, 1]
    ax.set_xlim(X - 2.2, X + 1.4)
    ax.set_ylim(-0.03, 0.95)
    ax.set_aspect("equal")
    ax.axhline(0, color=S.INK2, lw=1.2)
    ax.axhspan(g[1] - 0.02, g[1] + 0.02, color=S.TOL, zorder=0)
    ax.axhline(g[1], color=S.INK, ls="--", lw=1.0)
    ax.text(X + 1.38, g[1] + 0.03, f"apex target {g[1]:.2f} m (±2 cm)", ha="right", fontsize=10, color=S.INK2)
    for kk in range(k):                                       # earlier iterations: path + ghost body at the same instant
        xp = TRAJ[kk]["x"][gi]
        jj = min(j, xp.shape[0] - 1)
        ax.plot(xp[:jj + 1, 0], xp[:jj + 1, 1], color=S.color(kk, K), lw=1.6, alpha=0.9, zorder=2)
        ax.plot([xp[jj, 0]], [xp[jj, 1]], "o", color=S.color(kk, K), ms=13, alpha=0.75, zorder=2)
        ax.text(xp[jj, 0], xp[jj, 1] + 0.05, f"π{S.sub(kk)}", ha="center", fontsize=10, color=S.color(kk, K))
    ax.plot(xs[:j + 1, 0], xs[:j + 1, 1], color=S.color(k, K), lw=2.6, zorder=3)
    st, th = xs[j, 5], xs[j, 6]
    ph = th + TRAJ[k]["phi_off"]
    fp = (xs[j, 4], 0.0) if st > 0.5 else (X + TRAJ[k]["l0"] * np.sin(ph), Z - TRAJ[k]["l0"] * np.cos(ph))
    ax.plot(*spring_xy((X, Z), fp), color=S.INK, lw=1.8, zorder=4)
    ax.plot([X], [Z], "o", color=S.color(k, K), mec=S.INK, ms=17, zorder=5)
    ax.text(X, Z + 0.07, f"π{S.sub(k)}", ha="center", fontsize=12, weight="bold", zorder=6)
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_ylabel("height (m)")
    ax.set_xlabel("forward position (m)")
    ax.set_title(f"stick '{robot.replace('real_', '')}'  -  goal: hop at {g[0]:.1f} m/s with apex {g[1]:.2f} m    "
                 f"(t = {j * C.DT:.2f} s of {T * C.DT:.0f} s)", loc="left")

    # apex / speed per hop
    a1, a2, a3 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    a1.axhspan(g[1] - 0.02, g[1] + 0.02, color=S.TOL)
    a1.axhline(g[1], color=S.INK, ls="--", lw=1.0)
    a2.axhspan(g[0] - 0.1, g[0] + 0.1, color=S.TOL)
    a2.axhline(g[0], color=S.INK, ls="--", lw=1.0)
    for kk in range(k + 1):
        xp = TRAJ[kk]["x"][gi]
        idx, ap, sp = hops(xp)
        upto = idx <= j if kk == k else np.ones(len(idx), bool)
        lw, a = (2.8, 1.0) if kk == k else (1.6, 0.85)
        n = np.arange(1, len(idx) + 1)
        a1.plot(n[upto], ap[upto], "-o", color=S.color(kk, K), lw=lw, ms=5 if kk == k else 3.5, alpha=a)
        a2.plot(n[upto], sp[upto], "-o", color=S.color(kk, K), lw=lw, ms=5 if kk == k else 3.5, alpha=a)
        if upto.any():
            a1.text(n[upto][-1] + 0.25, ap[upto][-1], f"π{S.sub(kk)}", fontsize=10, color=S.color(kk, K), va="center")
    a1.set_title("apex height, hop by hop", loc="left")
    a1.set_xlabel("hop")
    a1.set_ylabel("apex (m)")
    a2.set_title("forward speed, hop by hop", loc="left")
    a2.set_xlabel("hop")
    a2.set_ylabel("speed (m/s)")
    nh = max(len(hops(TRAJ[kk]["x"][gi])[0]) for kk in range(K + 1))
    for a_ in (a1, a2):
        a_.set_xlim(0.5, nh + 1.5)
    a1.set_ylim(min(g[1] - 0.12, min(np.nanmin(hops(TRAJ[kk]["x"][gi])[1]) for kk in range(K + 1)) - 0.01),
                max(g[1] + 0.06, max(np.nanmax(hops(TRAJ[kk]["x"][gi])[1]) for kk in range(K + 1)) + 0.01))

    # the batch over iterations
    done = k + 1 if phase == "pause" else k
    succ = [sum(bool(q["success"]) for q in TRAJ[kk]["m"]) for kk in range(K + 1)]
    for kk in range(K + 1):
        if kk < done:
            a3.bar(kk, succ[kk], color=S.color(kk, K), width=0.7)
            a3.text(kk, succ[kk] + 0.12, f"{succ[kk]}/6", ha="center", fontsize=11)
        else:
            a3.bar(kk, 6, color="none", ec=S.GRID, width=0.7, ls=":")
    a3.set_xticks(range(K + 1))
    a3.set_xticklabels([f"π{S.sub(kk)}" for kk in range(K + 1)])
    a3.set_ylim(0, 7)
    a3.set_yticks(range(0, 7, 2))
    a3.set_title("training goals on target (of 6)", loc="left")
    a3.set_xlabel("iteration (6 real trials each)")
    res = None
    if phase == "pause":
        m = TRAJ[k]["m"][gi]
        ok = "FELL" if TRAJ[k]["fell"][gi] else ("on target" if m["success"] else "off target")
        ra = m["rms_apex"]
        res = (f"π{S.sub(k)} on this goal: apex RMS {100 * ra:.1f} cm, speed error {m['v_err']:+.2f} m/s  ({ok})"
               if ra is not None else f"π{S.sub(k)} on this goal: {ok}")
    S.header(fig, f"Pogo stick: DistILC hardware stage, iteration by iteration  ({robot.replace('real_', '')} stick, sim-to-sim)",
             k, K, phase, res)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    global TRAJ, META
    robot = sys.argv[1]
    arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "nom_s0"
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 16
    TRAJ = fly_stage(arm, robot)
    if "--goal" in sys.argv:
        gi = int(sys.argv[sys.argv.index("--goal") + 1])
    else:                                                     # the goal pi_0 misses worst that the last network makes
        K = len(TRAJ) - 1
        cand = [i for i in range(6) if TRAJ[K]["m"][i]["success"]] or list(range(6))
        err = lambda i: 9.0 if TRAJ[0]["fell"][i] or TRAJ[0]["m"][i]["rms_apex"] is None else TRAJ[0]["m"][i]["rms_apex"]
        gi = max(cand, key=err)
    META = dict(robot=robot, goal=gi)
    print("goal", gi, C.TRAIN_GOALS[gi], flush=True)
    os.makedirs(OUT, exist_ok=True)
    n_fly = [(TRAJ[k]["x"].shape[1] - 1) // STEP + 1 for k in range(len(TRAJ))]
    plan = S.frame_plan(n_fly, len(TRAJ) - 1)
    mp4 = f"{OUT}/pogo_iters_{robot}_{arm}.mp4"
    n = S.render(frame, plan, mp4, procs, stills=[(0, mp4[:-4] + "_pi0.png"), (len(TRAJ) - 1, mp4[:-4] + "_final.png")])
    print(mp4, n, "frames", flush=True)


if __name__ == "__main__":
    main()
