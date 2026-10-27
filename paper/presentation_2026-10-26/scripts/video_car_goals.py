#!/usr/bin/env python3
"""
video_car_goals.py (2026-10-26): goal conditioning on the car -- ONE network π_k drives 6 goals at once (rows: the
square and the figure-8 track; columns: drift angle 0 / 18 / 25 degrees), iteration by iteration, no graphs. Each
panel: the goal's reference line (dashed), earlier iterations' paths as ghosts in their ramp colours, the car of the
current one; each iteration ends in a pause with every goal's RMS lateral error or crash. From car_fly_goals.py.
    python video_car_goals.py CAR [--speed 3] [--procs 12]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iter_style as S                                        # noqa: E402
import video_car_iters as V                                   # noqa: E402

OUT = V.OUT
D = None
SPEED = 3.0
TRACK = {"square2fast": "rounded-square track", "figfast": "figure-8"}


def frame(job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.rc()
    i, k, t, phase, path = job
    K, dt = D["K"], D["dt"]
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 3, left=0.01, right=0.99, top=0.80, bottom=0.05, hspace=0.22, wspace=0.03)
    n_ok, n_all = 0, 0
    for gi, g in enumerate(D["goals"]):
        ax = fig.add_subplot(gs[gi // 3, gi % 3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        xy = D["plan"][g]
        ax.plot(np.r_[xy[:, 0], xy[:1, 0]], np.r_[xy[:, 1], xy[:1, 1]], color=S.INK, lw=1.0, ls="--", zorder=2)
        runs = D["runs"][g]
        for kk in range(k):
            xp = runs[kk]["x"]
            ax.plot(xp[:, 0], xp[:, 1], color=S.color(kk, K), lw=1.0, alpha=0.85, zorder=3)
            if runs[kk]["failed"]:
                ax.plot([xp[-1, 0]], [xp[-1, 1]], "x", color=S.BAD, ms=10, mew=2, zorder=7)
        cur = runs[k]
        x = cur["x"]
        j = len(x) - 1 if t is None else min(int(t * SPEED / (S.FPS * dt)), len(x) - 1)
        ax.plot(x[:j + 1, 0], x[:j + 1, 1], color=S.color(k, K), lw=2.0, zorder=4)
        b = V.car_poly(*x[j, :3], L=0.36, W=0.2)
        ax.fill(b[:, 0], b[:, 1], color=S.color(k, K), ec=S.INK, lw=1.0, zorder=6)
        pad = 0.35
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
        trk, ang = g.replace("mocap_", "").split("_b")
        ax.set_title(f"goal {gi + 1}: {TRACK.get(trk, trk)}, drift angle {int(ang)}°", loc="left", fontsize=12.5)
        if phase == "pause":
            n_all += 1
            if cur["failed"]:
                txt, col = f"✗ crashed ({cur['crash'] or 'off track'}) after {cur['laps']:.1f} laps", S.BAD
                ax.plot([x[j, 0]], [x[j, 1]], "x", color=S.BAD, ms=16, mew=3, zorder=8)
            else:
                n_ok += 1
                txt, col = f"RMS lateral error {cur['rms']:.1f} cm", S.INK
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=12.5, weight="bold", color=col,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=S.GRID, alpha=0.9), zorder=10)
    res = None
    if phase == "pause":
        ok = [D["runs"][g][k]["rms"] for g in D["goals"] if not D["runs"][g][k]["failed"]]
        res = (f"one network π{S.sub(k)}, six goals:  {len(ok)} of 6 completed" +
               (f",  mean RMS lateral error {np.mean(ok):.1f} cm" if ok else ""))
    S.header(fig, f"Drift car (F1TENTH): one goal-conditioned network drives every goal, iteration by iteration  "
                  f"({D['car']}, sim-to-sim)", k, K, phase, res)
    fig.text(0.99, 0.845, f"dashed: the goal's reference line     {SPEED:g}x real time", ha="right", va="center",
             fontsize=11, color=S.INK2)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    global D, SPEED
    car = sys.argv[1]
    SPEED = float(sys.argv[sys.argv.index("--speed") + 1]) if "--speed" in sys.argv else 3.0
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 12
    tag = car.replace(":", "").replace("=", "")
    z = np.load(f"{OUT}/goals_trials_{tag}.npz")
    meta = json.loads(str(z["meta"]))
    goals = list(dict.fromkeys(m["plan"] for m in meta))
    K = max(m["it"] for m in meta)
    runs = {g: [None] * (K + 1) for g in goals}
    for i, m in enumerate(meta):
        runs[m["plan"]][m["it"]] = dict(x=z[f"x_{i}"], failed=bool(m["failed"]), crash=m.get("crash", ""), laps=m["laps"],
                                        rms=None if m["failed"] else 100 * m["rms_ey"])
    D = dict(goals=goals, K=K, runs=runs, dt=float(z["dt"]), car=car.replace("real_", "").replace("ax:mu=", "friction x"),
             plan={g: z[f"plan__{g}__xy"] for g in goals})
    for g in goals:
        print(g, ["crash" if r["failed"] else f"{r['rms']:.1f}" for r in runs[g]], flush=True)
    n_fly = [int(np.ceil(max(len(runs[g][k]["x"]) for g in goals) * D["dt"] * S.FPS / SPEED)) + 1 for k in range(K + 1)]
    plan = S.frame_plan(n_fly, K)
    mp4 = f"{OUT}/car_goals_{tag}.mp4"
    n = S.render(frame, plan, mp4, procs, stills=[(0, mp4[:-4] + "_pi0.png"), (K, mp4[:-4] + "_final.png")])
    print(mp4, n, "frames", flush=True)


if __name__ == "__main__":
    main()
