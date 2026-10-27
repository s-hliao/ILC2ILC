#!/usr/bin/env python3
"""
video_pogo_goals.py (2026-10-26): goal conditioning -- ONE network π_k flies all 6 training goals at once (6 lanes,
the same stick), iteration by iteration, no graphs. Each lane: the stick hopping toward its own goal (speed v*, apex
h*: target line and band), earlier iterations' paths as ghosts in their ramp colours; each iteration ends in a pause
with every goal's verdict. Same re-flights as video_pogo_iters.py (servo leg, log/pogo_servo/runs/v3/<arm>_curve).
    POGO_LEG=servo python video_pogo_goals.py ROBOT [--arm nom_s0] [--procs 16]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter_style as S                                        # noqa: E402
import video_pogo_iters as V                                  # noqa: E402

C = V.C
TRAJ, ROBOT = None, None


def frame(job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.rc()
    i, k, t, phase, path = job
    K = len(TRAJ) - 1
    G = C.TRAIN_GOALS
    T = TRAJ[k]["x"].shape[1] - 1
    j = T if t is None else min(t * V.STEP, T)
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(3, 2, left=0.035, right=0.99, top=0.80, bottom=0.04, hspace=0.42, wspace=0.06)
    order = np.argsort(G[:, 0])                               # lanes by goal speed
    xmax = max(TRAJ[kk]["x"][:, :, 0].max() for kk in range(K + 1)) + 0.4
    for li, gi in enumerate(order):
        ax = fig.add_subplot(gs[li % 3, li // 3])
        g = G[gi]
        xs = TRAJ[k]["x"][gi]
        ax.set_xlim(-0.6, xmax)
        ax.set_ylim(-0.02, 0.92)
        ax.set_aspect("equal")
        ax.axhline(0, color=S.INK2, lw=1.0)
        ax.axhspan(g[1] - 0.02, g[1] + 0.02, color=S.TOL, zorder=0)
        ax.axhline(g[1], color=S.INK, ls="--", lw=0.9)
        # the goal speed as a moving marker: where a stick at exactly v* would be now
        ax.plot([xs[0, 0] + g[0] * j * C.DT], [0.0], marker="^", ms=11, color=S.OK, zorder=6, clip_on=False)
        for kk in range(k):
            xp = TRAJ[kk]["x"][gi]
            jj = min(j, xp.shape[0] - 1)
            ax.plot(xp[:jj + 1, 0], xp[:jj + 1, 1], color=S.color(kk, K), lw=1.2, alpha=0.75, zorder=2)
        ax.plot(xs[:j + 1, 0], xs[:j + 1, 1], color=S.color(k, K), lw=2.2, zorder=3)
        X, Z, st, th = xs[j, 0], xs[j, 1], xs[j, 5], xs[j, 6]
        ph = th + TRAJ[k]["phi_off"]
        fp_ = (xs[j, 4], 0.0) if st > 0.5 else (X + TRAJ[k]["l0"] * np.sin(ph), Z - TRAJ[k]["l0"] * np.cos(ph))
        ax.plot(*V.spring_xy((X, Z), fp_, w=0.03), color=S.INK, lw=1.4, zorder=4)
        ax.plot([X], [Z], "o", color=S.color(k, K), mec=S.INK, ms=12, zorder=5)
        ax.set_yticks([])
        ax.set_xticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        lab = f"goal {li + 1}:  hop at {g[0]:.1f} m/s,  apex {g[1]:.2f} m"
        if phase == "pause":
            m = TRAJ[k]["m"][gi]
            if TRAJ[k]["fell"][gi]:
                verdict, col = "  ✗ fell", S.BAD
            elif m["success"]:
                verdict, col = f"  ✓ on target (apex rms {100 * m['rms_apex']:.1f} cm, speed {m['v_err']:+.2f} m/s)", S.OK
            else:
                ra = m["rms_apex"]
                verdict = (f"  ✗ off (apex rms {100 * ra:.1f} cm, speed {m['v_err']:+.2f} m/s)" if ra is not None else "  ✗ off")
                col = S.BAD
            ax.set_title(lab, loc="left", fontsize=12.5)
            ax.text(0.0, -0.04, verdict.strip(), transform=ax.transAxes, ha="left", va="top", fontsize=12.5, color=col, weight="bold")
        else:
            ax.set_title(lab, loc="left", fontsize=12.5)
    res = None
    if phase == "pause":
        n_ok = sum(bool(m["success"]) for m in TRAJ[k]["m"])
        res = f"one network π{S.sub(k)}, six goals:  {n_ok} of 6 on target"
    S.header(fig, f"Pogo stick: one goal-conditioned network flies every goal, iteration by iteration  "
                  f"({ROBOT.replace('real_', '')} stick, sim-to-sim)", k, K, phase, res)
    fig.text(0.99, 0.845, "dashed line + band: apex target ±2 cm     green ▲: where a stick at the goal speed would be",
             ha="right", va="center", fontsize=11, color=S.INK2)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    global TRAJ, ROBOT
    ROBOT = sys.argv[1]
    arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "nom_s0"
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 16
    TRAJ = V.fly_stage(arm, ROBOT)
    os.makedirs(V.OUT, exist_ok=True)
    n_fly = [(TRAJ[k]["x"].shape[1] - 1) // V.STEP + 1 for k in range(len(TRAJ))]
    plan = S.frame_plan(n_fly, len(TRAJ) - 1)
    mp4 = f"{V.OUT}/pogo_goals_{ROBOT}_{arm}.mp4"
    n = S.render(frame, plan, mp4, procs, stills=[(0, mp4[:-4] + "_pi0.png"), (len(TRAJ) - 1, mp4[:-4] + "_final.png")])
    print(mp4, n, "frames", flush=True)


if __name__ == "__main__":
    main()
